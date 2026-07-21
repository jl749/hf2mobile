import tempfile
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set

import onnx
import onnx.external_data_helper  # NOTE: `import onnx` alone does not expose this submodule
import onnx_ir as ir
import onnxscript.optimizer
from onnx_ir.passes.common import InlinePass, LiftConstantsToInitializersPass, RemoveUnusedNodesPass

from hf2hw.constant import ONNX_DOMAIN_NAME

# ===================== onnx/ ===================== #


def onnx_to_function(
    submodule_onnx_path: str,
    function_name: str,
    domain: str = ONNX_DOMAIN_NAME,
) -> onnx.FunctionProto:
    """
    Load a standalone submodule ONNX and convert its graph to a FunctionProto.

    FunctionProto does not accept graph-level initializers.
    Hence, initializers are inlined as `Constant` nodes prepended to the function body.
    """
    m = onnx.load(submodule_onnx_path, load_external_data=True)

    const_nodes = [
        onnx.helper.make_node("Constant", inputs=[], outputs=[init.name], value=init, name=f"const_{init.name}")
        for init in m.graph.initializer
    ]
    submodule_nodes = list(m.graph.node)

    func = onnx.helper.make_function(
        domain=domain,
        fname=function_name,
        inputs=[i.name for i in m.graph.input],
        outputs=[o.name for o in m.graph.output],
        nodes=const_nodes + submodule_nodes,
        opset_imports=list(m.opset_import),
    )
    return func


def ensure_opset_imports(model: onnx.ModelProto, new_imports) -> None:
    """Add opset imports from a function into the model if not already present."""
    existing = {(o.domain, o.version) for o in model.opset_import}
    existing_domains = {o.domain for o in model.opset_import}
    for op in new_imports:
        if op.domain in existing_domains:
            continue  # do not clash with existing version pins
        if (op.domain, op.version) in existing:
            continue
        model.opset_import.append(op)


def update_opset(model: onnx.ModelProto, domain: str, version: int) -> None:
    """
    If `onnx.OperatorSetIdProto(domain, version)` does not exist under opset_import add it.
    If opset_import exists but the version is lower upgrade it.
    """
    for oi in model.opset_import:
        if oi.domain == domain:
            if oi.version < version:
                oi.version = version  # NOTE: version up the opset
            return
    model.opset_import.append(onnx.helper.make_opsetid(domain, version))


def drop_attributes(node: onnx.NodeProto, names_to_drop: Set[str]) -> None:
    """Drop AttributeProtos by their names from the passed NodeProto"""
    keep = [a for a in node.attribute if a.name not in names_to_drop]
    del node.attribute[:]
    node.attribute.extend(keep)


# ===================== onnx/fusion ===================== #


def get_scalar(val: ir.Value) -> float | None:
    """Extract a scalar float from a constant-valued IR Value, or None."""
    try:
        cv = val.const_value
        if cv is None:
            return None
        return float(cv.numpy().flat[0])
    except Exception:
        return None


def get_fwd_dict(nodes: Sequence[onnx.NodeProto]) -> Dict[str, List[onnx.NodeProto]]:
    consumers: Dict[str, List[onnx.NodeProto]] = {}
    for node in nodes:
        for edge_name in node.input:
            consumers.setdefault(edge_name, []).append(node)
    return consumers


def get_bwd_dict(nodes: Sequence[onnx.NodeProto]) -> Dict[str, onnx.NodeProto]:
    return {out: n for n in nodes for out in n.output}


# ===================== onnx/postprocess ===================== #


def set_vi_axis(vi: onnx.ValueInfoProto, axis: int, value: int | str) -> None:
    """Set a dim on `vi`: an `int` writes a fixed `dim_value`, a `str` a symbolic `dim_param`."""
    dim = vi.type.tensor_type.shape.dim[axis]
    if isinstance(value, str):
        dim.ClearField("dim_value")
        dim.dim_param = value
    elif isinstance(value, int):
        dim.ClearField("dim_param")
        dim.dim_value = value
    else:
        raise TypeError(f"`{value=}` must be int (dim_value) or str (dim_param), got {type(value).__name__}.")


def get_vi_axis(vi: onnx.ValueInfoProto, axis: int) -> int | str | None:
    """Return a dim of `vi`: `int` for a fixed `dim_value`, `str` for a symbolic `dim_param`, None if unset."""
    dim = vi.type.tensor_type.shape.dim[axis]
    if dim.HasField("dim_param"):
        return dim.dim_param
    if dim.HasField("dim_value"):
        return dim.dim_value
    return None


def drop_vi_by_name(vis: Sequence[onnx.ValueInfoProto], names: Set[str]) -> None:
    """Remove named entries from the given `vis` inplace (NOTE: assumes they have no consumers)."""
    keep = [vi for vi in vis if vi.name not in names]
    del vis[:]
    vis.extend(keep)


def update_node_attribute(node: onnx.NodeProto, attribute_name: str, value: Any):
    """Update node attribute using the new value."""
    attr = next((a for a in node.attribute if a.name == attribute_name), None)
    if attr is None:
        node.attribute.append(onnx.helper.make_attribute(attribute_name, value))
    elif attr.i != value:
        attr.i = value


# ===================== GENERAL ===================== #
def save_onnx(model: onnx.ModelProto, save_path: str | PathLike):
    data_path = Path(f"{save_path}.data")
    data_path.unlink(missing_ok=True)
    onnx.save(
        model,
        save_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path.name,
        size_threshold=1024,
    )
    onnx.external_data_helper.load_external_data_for_model(model, str(data_path.resolve().parent))


def optimize_onnx(model: onnx.ModelProto):
    """
    pure-Python onnx_ir passes
    onnx.inliner/onnxoptimizer round-trip the full model through GB-scale protobuf C++ (de)serialization
    which could be the source of silent nondeterministic weight corruption

    - inline local functions + drop unused ones
    - Constant to initializers
    - clean dead nodes + drop unused initializers
    """
    model_ir = ir.from_proto(model)
    InlinePass()(model_ir)
    onnxscript.optimizer.fold_constants(model_ir)
    LiftConstantsToInitializersPass()(model_ir)
    RemoveUnusedNodesPass()(model_ir)
    model = ir.to_proto(model_ir)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        temp_onnx_path = temp_dir.joinpath("delete_me.onnx")
        save_onnx(model, str(temp_onnx_path))
        onnx.shape_inference.infer_shapes_path(str(temp_onnx_path), check_type=True, strict_mode=False)
        model = onnx.load(str(temp_onnx_path))
    return model
