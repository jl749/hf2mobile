"""Merge standalone submodule ONNX files into a case graph as local functions.

Each placeholder plugin node in ``case{i}.onnx`` (e.g. ``CustomAttention``) is
rewritten into a call to a ``FunctionProto`` whose body is the standalone
attention ONNX. The function name reuses the per-instance ``torchlib_op_name``
already stamped on the placeholder, so wiring is 1:1.

NOTE: weights from the standalone ONNX are inlined as ``Constant`` nodes inside
the FunctionProto (FunctionProto has no graph-level initializers). This grows
file size — optimize later by hoisting weights to the parent model.

Norm fusion is *not* performed here: subblock norms are already fused inside
``submodules/attention.py`` at export time, and main-graph norm fusion is done
explicitly by ``SubgraphExporterInterface._merge_subgraphs_into_main_graph``.
"""

from typing import Dict, List

import onnx
from onnx import FunctionProto, ModelProto, helper

from ...constant import ONNX_DOMAIN_NAME
from ...utils.logger import logger

_PLACEHOLDER_ATTR = "torchlib_op_name"


def onnx_to_function(
    submodule_onnx_path: str,
    function_name: str,
    domain: str = ONNX_DOMAIN_NAME,
) -> FunctionProto:
    """Load a standalone submodule ONNX and convert its graph to a FunctionProto.

    Initializers are inlined as ``Constant`` nodes prepended to the function body,
    since FunctionProto does not accept graph-level initializers.
    """
    m = onnx.load(submodule_onnx_path, load_external_data=True)

    g = m.graph
    init_nodes = [
        helper.make_node("Constant", inputs=[], outputs=[init.name], value=init, name=f"const_{init.name}")
        for init in g.initializer
    ]
    body_nodes = list(g.node)

    func = helper.make_function(
        domain=domain,
        fname=function_name,
        inputs=[i.name for i in g.input],
        outputs=[o.name for o in g.output],
        nodes=init_nodes + body_nodes,
        opset_imports=list(m.opset_import),
    )
    logger.debug(
        f"onnx_to_function: {submodule_onnx_path} → fn {function_name!r} "
        f"({len(init_nodes)} const(s) + {len(body_nodes)} node(s), "
        f"{len(func.input)} in / {len(func.output)} out)"
    )
    return func


def merge_subgraphs_into_model(
    case_path: str,
    torchlib_op_to_submodule_path: Dict[str, str],
    domain: str = ONNX_DOMAIN_NAME,
    out_path: str | None = None,
) -> str:
    """Rewrite placeholder nodes in ``case_path`` to call inlined FunctionProtos.

    A placeholder node is any node in ``domain`` carrying a ``torchlib_op_name``
    attribute. For each such node whose ``torchlib_op_name`` is present in
    ``torchlib_op_to_submodule_path``:
        1. Build a FunctionProto named after ``torchlib_op_name`` from its
           standalone ONNX (once per unique name).
        2. Rewrite the node's ``op_type`` to ``torchlib_op_name`` (domain unchanged).
        3. Append the FunctionProto to ``model.functions``.

    Nodes whose ``torchlib_op_name`` is not in the mapping (e.g. a different case,
    or an unimplemented plugin type) are left untouched.

    Args:
        case_path: existing ``case{i}.onnx`` to patch.
        torchlib_op_to_submodule_path: ``{torchlib_op_name: standalone_onnx_path}``.
        domain: ONNX domain for both placeholder nodes and emitted functions.
        out_path: target path; defaults to ``case_path`` (in-place).

    Returns:
        Path of the saved model.
    """
    out_path = out_path or case_path
    model: ModelProto = onnx.load(case_path, load_external_data=True)

    rewritten = 0
    appended_fn_names: List[str] = []
    seen_fn_names = {(f.domain, f.name) for f in model.functions}

    for node in model.graph.node:
        if node.domain != domain:
            continue
        attr = next((a for a in node.attribute if a.name == _PLACEHOLDER_ATTR), None)
        if attr is None:
            continue
        torchlib_op_name = attr.s.decode() if isinstance(attr.s, (bytes, bytearray)) else str(attr.s)

        sub_path = torchlib_op_to_submodule_path.get(torchlib_op_name)
        if sub_path is None:
            logger.warning(
                f"merge_subgraphs_into_model: no standalone ONNX provided for "
                f"{torchlib_op_name!r}; leaving placeholder in place."
            )
            continue

        # Build & register the function (skip if a same-named function is already present).
        if (domain, torchlib_op_name) not in seen_fn_names:
            func = onnx_to_function(sub_path, function_name=torchlib_op_name, domain=domain)
            _ensure_opset_imports(model, func.opset_import)
            model.functions.append(func)
            seen_fn_names.add((domain, torchlib_op_name))
            appended_fn_names.append(torchlib_op_name)

        # Validate input/output arity before mutating the node.
        func = next(f for f in model.functions if f.name == torchlib_op_name and f.domain == domain)
        if len(node.input) != len(func.input):
            raise ValueError(
                f"Input count mismatch for {torchlib_op_name!r}: "
                f"placeholder has {len(node.input)} inputs, function has {len(func.input)}."
            )
        if len(node.output) != len(func.output):
            raise ValueError(
                f"Output count mismatch for {torchlib_op_name!r}: "
                f"placeholder has {len(node.output)} outputs, function has {len(func.output)}."
            )

        # Rewrite placeholder into a function call (domain stays the same).
        node.op_type = torchlib_op_name
        # Drop placeholder-only attributes so the node is a clean function call.
        _clear_attributes(node, names={_PLACEHOLDER_ATTR})
        rewritten += 1

    onnx.save(model, out_path)
    logger.info(
        f"merge_subgraphs_into_model: {case_path} → {out_path} "
        f"({rewritten} node(s) rewritten, {len(appended_fn_names)} function(s) appended)"
    )
    return out_path


def _ensure_opset_imports(model: ModelProto, new_imports) -> None:
    """Add opset imports from a function into the model if not already present."""
    existing = {(o.domain, o.version) for o in model.opset_import}
    existing_domains = {o.domain for o in model.opset_import}
    for op in new_imports:
        if op.domain in existing_domains:
            continue  # do not clash with existing version pins
        if (op.domain, op.version) in existing:
            continue
        model.opset_import.append(op)


def _clear_attributes(node, names) -> None:
    keep = [a for a in node.attribute if a.name not in names]
    del node.attribute[:]
    node.attribute.extend(keep)


__all__ = ["onnx_to_function", "merge_subgraphs_into_model"]
