from typing import Dict, List

import onnx

from ...constant import ONNX_DOMAIN_NAME
from ...utils import drop_attributes, ensure_opset_imports
from ...utils.logger import logger

_PLACEHOLDER_ATTR = "torchlib_op_name"


def _onnx_to_function(
    submodule_onnx_path: str,
    function_name: str,
    domain: str = ONNX_DOMAIN_NAME,
) -> onnx.FunctionProto:
    """
    Load a standalone submodule ONNX and convert its graph to a FunctionProto.

    FunctionProto does not accept graph-level initializers.
    Hence, iInitializers are inlined as `Constant` nodes prepended to the function body.
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
    logger.debug(
        f"onnx_to_function: {submodule_onnx_path} → fn {function_name!r} "
        f"({len(const_nodes)} const(s) + {len(submodule_nodes)} node(s), "
        f"{len(func.input)} in / {len(func.output)} out)"
    )
    return func


def merge_subgraphs_into_model(
    case_path: str,
    torchlib_op2subgraph_path: Dict[str, str],
    domain: str = ONNX_DOMAIN_NAME,
    output_path: str | None = None,
) -> str:
    """
    Replace custom plugin nodes under the main ONNX graph (`cast_path`) using `torchlib_op2subgraph_path`.
    Each subgraph will be represented in FunctionProto.
    `subgraph_path = torchlib_op2subgraph_path[plugin_node.attribute["torchlib_op_name"]]`

    Args:
        case_path: existing `case{i}.onnx` to patch
        torchlib_op2subgraph_path: `{torchlib_op_name: standalone_onnx_path}`
        domain: ONNX domain for both placeholder nodes and emitted functions
        output_path: save path if provided. if None -> overwrite `case_path`
    Returns:
        merged onnx path
    """
    output_path = output_path or case_path
    model: onnx.ModelProto = onnx.load(case_path, load_external_data=True)

    rewritten = 0
    appended_fn_names: List[str] = []
    seen_fn_names = {(f.domain, f.name) for f in model.functions}

    for node in model.graph.node:
        if node.domain != domain:
            continue
        _attr = next((a for a in node.attribute if a.name == _PLACEHOLDER_ATTR), None)
        if _attr is None:
            continue

        # NOTE: from this point => `node = {...torchlib registered plugin node...}`
        torchlib_op_name = _attr.s.decode() if isinstance(_attr.s, (bytes, bytearray)) else str(_attr.s)

        subgraph_path = torchlib_op2subgraph_path.get(torchlib_op_name, None)
        if subgraph_path is None:
            logger.warning(
                f"merge_subgraphs_into_model: no standalone ONNX provided for "
                f"{torchlib_op_name!r}; leaving placeholder in place."
            )
            continue

        # build and register the function
        if (domain, torchlib_op_name) not in seen_fn_names:
            func: onnx.FunctionProto = _onnx_to_function(subgraph_path, function_name=torchlib_op_name, domain=domain)
            ensure_opset_imports(model, func.opset_import)
            model.functions.append(func)
            seen_fn_names.add((domain, torchlib_op_name))
            appended_fn_names.append(torchlib_op_name)

        # validate input/output arity before mutating the node.
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

        # rewrite placeholder into a function call (domain stays the same).
        node.op_type = torchlib_op_name
        # drop placeholder-only attributes so the node is a clean function call.
        drop_attributes(node, names_to_drop={_PLACEHOLDER_ATTR})  # TODO: is this step required?
        rewritten += 1

    onnx.save(model, output_path)
    logger.info(
        f"merge_subgraphs_into_model: {case_path} → {output_path} "
        f"({rewritten} node(s) rewritten, {len(appended_fn_names)} function(s) appended)"
    )
    return output_path


__all__ = ["merge_subgraphs_into_model"]
