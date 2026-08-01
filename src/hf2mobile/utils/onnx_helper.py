"""
`onnx_ir` centered ONNX helpers.

The pipeline holds graphs as `ir.Model` and touches `onnx.ModelProto` only where a C++ API demands one.
Two reasons:

* memory — `ir.load` mmaps external tensors, so a multi-GB graph costs page cache the kernel can evict rather than heap.
  `onnx.load(load_external_data=True)` reads every weight into RAM.
* use-def — `ir.Value` knows its producer and its consumers.
  so graph surgery is a local edit rather than a rebuild of fwd/bwd dictionaries.

The one trap worth naming: `ir.Function` accepts initializers, but serialization **silently drops them**
  `FunctionProto` has no initializer field and nothing raises.
  Anything that becomes a function has to carry its weights as `Constant` nodes (see `graph_to_function`).
"""

import shutil
from os import PathLike
from pathlib import Path
from typing import Any, List, Sequence, Set

import numpy as np
import onnx_ir as ir
import onnxscript.optimizer
from onnx_ir.passes.common import (
    InlinePass,
    LiftConstantsToInitializersPass,
    RemoveUnusedNodesPass,
    ShapeInferencePass,
)

from hf2mobile.constant import ONNX_DOMAIN_NAME

# ===================== onnx/ ===================== #


def graph_to_function(
    submodule_onnx_path: str | PathLike,
    function_name: str,
    domain: str = ONNX_DOMAIN_NAME,
) -> ir.Function:
    """
    Load a standalone submodule ONNX and convert its graph to an `ir.Function`.

    Functions cannot carry initializers — `FunctionProto` has no field for them and `ir.to_proto` drops them without complaint.
    Every initializer is re-emitted as a `Constant` node at the head of the body.
    """
    model_ir = load_onnx_ir(submodule_onnx_path)
    graph = model_ir.graph

    const_nodes = []
    for init in tuple(graph.initializers.values()):
        out = ir.Value(name=init.name, type=init.type, shape=init.shape)
        const_nodes.append(
            ir.Node(
                "",
                "Constant",
                inputs=[],
                attributes={"value": ir.AttrTensor("value", init.const_value)},
                outputs=[out],
                name=f"const_{init.name}",
            )
        )
        ir.convenience.replace_all_uses_with(init, out)
        graph.initializers.pop(init.name)

    prepend_nodes_to_graph(graph, const_nodes)  # const nodes go first when toposort
    return ir.Function(domain=domain, name=function_name, graph=graph, attributes=())


def update_opset(model_ir: ir.Model, domain: str, version: int) -> None:
    """Pin `domain` to at least `version`, never downgrading an existing pin."""
    if model_ir.opset_imports.get(domain, -1) < version:
        model_ir.opset_imports[domain] = version


def drop_attributes(node: ir.Node, names_to_drop: Set[str]) -> None:
    """Drop attributes by name from `node`."""
    for name in names_to_drop:
        node.attributes.pop(name, None)


# ===================== onnx/fusion ===================== #


def get_scalar(val: ir.Value) -> float | None:
    """Extract a scalar float from a constant-valued IR Value, or None."""
    try:
        const_val = val.const_value
        if const_val is None:
            return None
        const_val_npy = const_val.numpy()
        if const_val_npy.size != 1:
            return None
        return float(const_val_npy.flat[0])
    except Exception:
        return None


def get_const_tensor(val: ir.Value | None) -> ir.TensorProtocol | None:
    """
    The constant tensor behind `val`, whether it arrived as an initializer or a `Constant` node.

    IMPORTANT: `Value.const_value` is set for initializers but not for a `Constant` node's output.
    """
    if val is None:
        return None
    if val.const_value is not None:
        return val.const_value  # `val` was initializer
    producer = val.producer()
    if producer is not None and producer.op_type == "Constant":
        attr = producer.attributes.get("value")
        if attr is not None:
            return attr.value  # `val` was Constant
    return None


# ===================== onnx/postprocess ===================== #


def set_value_axis(value: ir.Value, axis: int, dim: int | str) -> None:
    """Set one axis of `value`'s shape: an `int` is a fixed dim, a `str` a symbolic one."""
    if value.shape is None:
        raise ValueError(f"`{value.name}` has no shape to set axis {axis} on.")
    dims = list(value.shape)
    dims[axis] = dim
    value.shape = ir.Shape(dims)


def get_value_shape(value: ir.Value) -> List[int | str | None] | None:
    """
    `value`'s shape(symbolic or int) in plain Python, or None when it has no shape at all.

    e.g.
    >>> get_value_shape(logits) # [1, 'L', 262144]
    """
    if value.shape is None:
        return None
    return [dim.value if isinstance(dim, ir.SymbolicDim) else int(dim) for dim in value.shape]


def drop_graph_io_by_name(io: list[ir.Value], names: Set[str]) -> None:
    """
    Remove named entries from a graph's inputs/outputs in place (assumes they have no consumers).

    e.g.
    >>> drop_graph_io_by_name(graph.input, {"attention_mask"})
    """
    keep = [v for v in io if v.name not in names]
    del io[:]
    io.extend(keep)


# TODO: support various attribute dtype
def update_node_attribute(node: ir.Node, attribute_name: str, value: Any) -> None:
    """Set an int attribute on `node`, adding it when absent."""
    node.attributes[attribute_name] = ir.AttrInt64(attribute_name, int(value))


def make_constant_node(name: str, array: np.ndarray) -> ir.Node:
    """A `Constant` node wrapping `array`, its output named `name`."""
    tensor = ir.tensor(array, name=name)
    out = ir.Value(name=name, type=ir.TensorType(tensor.dtype), shape=ir.Shape(tensor.shape))
    return ir.Node(
        "",
        "Constant",
        inputs=[],
        attributes={"value": ir.AttrTensor("value", tensor)},
        outputs=[out],
        name=f"const_{name}",
    )


def append_node_input(node: ir.Node, value: ir.Value) -> None:
    """
    Append `value` to `node`'s inputs.

    `Node.inputs` is an immutable tuple — the IR owns it because every entry carries a use-def link.
    `resize_inputs` grows it with a `None` slot and `replace_input_with` registers the usage.
    """
    node.resize_inputs(len(node.inputs) + 1)
    node.replace_input_with(len(node.inputs) - 1, value)


def prepend_nodes_to_graph(graph: ir.Graph | ir.Function, nodes: Sequence[ir.Node]) -> None:
    """Insert `nodes` at the head of `graph`, keeping their relative order."""
    if not nodes:
        return
    first = next(iter(graph), None)
    if first is None:
        graph.extend(nodes)
    else:
        for node in nodes:
            graph.insert_before(first, node)


# ===================== GENERAL ===================== #


def load_onnx_ir(onnx_path: str | PathLike) -> ir.Model:
    """Load an ONNX as an `ir.Model`, mmapping its external tensors instead of reading them into RAM."""
    return ir.load(Path(onnx_path).resolve())


def save_onnx_ir(model_ir: ir.Model, save_path: str | PathLike) -> None:
    """
    Save an `ir.Model` with its tensors externalized into a sibling `{save_path}.data`.

    Writing directly to `{save_path}.data` is *correct* (`ir.save` notices the destination is the file)
    but that buffering tensors form the file is a full copy of the weights in RAM,
    which is the whole thing this path exists to avoid (measured 1.60 GB vs 0.35 GB on fp32 gemma-3-270m).
    Staging to a fresh file lets it stream tensor by tensor instead, and the rename is atomic.
    Do NOT `unlink` the destination first: `ir.save` re-opens each source tensor by path, so deleting it raises `FileNotFoundError`.
        `.{save name}.stage_dir/{save name}` -> {save_path}
        `.{save name}.stage_dir/{save name}.data` -> {save_path}.data
    """
    save_path = Path(save_path)
    stage_dir = save_path.parent.joinpath(f".{save_path.name}.stage_dir")
    shutil.rmtree(stage_dir, ignore_errors=True)  # in case it exists (clean mkdir)
    stage_dir.mkdir(parents=True)
    try:
        staged_onnx_path = stage_dir.joinpath(save_path.name)
        ir.save(model_ir, staged_onnx_path, external_data=f"{save_path.name}.data", size_threshold_bytes=1024)
        staged_data = Path(f"{staged_onnx_path}.data")
        staged_onnx_path.rename(save_path)  # move onnx to `save_path`
        if staged_data.is_file():
            staged_data.rename(f"{save_path}.data")  # (if exist) move data to `save_path`
        else:
            Path(f"{save_path}.data").unlink(missing_ok=True)  # original data not needed anymore
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)


def optimize_onnx(model_ir: ir.Model) -> None:
    """
    Flatten and clean `model_ir` in place.

    **pure-Python onnx_ir passes**
    onnx.inliner/onnxoptimizer round-trip the full model through GB-scale protobuf C++ (de)serialization.
    This could be the source of silent nondeterministic weight corruption.

    - inline local functions + drop unused ones
    - Constant to initializers
    - clean dead nodes + drop unused initializers
    - infer shapes

    `ShapeInferencePass` swaps the big initializers out for graph inputs before it serializes to proto for the C++ inferencer,
    so the weights are never copied (no disk round-trip).
    """
    InlinePass()(model_ir)
    onnxscript.optimizer.fold_constants(model_ir)
    LiftConstantsToInitializersPass()(model_ir)
    RemoveUnusedNodesPass()(model_ir)
    ShapeInferencePass(check_type=True, strict_mode=False, data_prop=False)(model_ir)
