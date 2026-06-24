import contextvars
import inspect
from abc import ABC
from typing import Any, Dict, List, Optional, Tuple

import onnx_ir
import torch
from torch.onnx._internal.exporter import _core as _onnx_core

from .tensor_metadata import TensorSpec, INPUT_SPECS_TYPE, OUTPUT_SPECS_TYPE
from .inspect import FwdSpec, sig2fwdspecs, _flatten_call_args
from utils.constant import CUSTOM_LIB_NAME, CUSTOM_LIB, ONNX_DOMAIN_NAME, _CACHE_PARAMS
from utils.py_helper import check_parent_field


# Case index threaded by `export_graphs` so each module's `plugin_forward`
# picks the right per-case op.  `None` means generation/inference (no export
# in progress) — `plugin_forward` falls straight through to `orig_cls.forward`.
export_case: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "export_case", default=None
)


def _filter_fwdspecs_by_input(
    fwdspecs: List[FwdSpec],
    input_spec: INPUT_SPECS_TYPE,
) -> List[FwdSpec]:
    """Keep only fwdspecs whose param actually appears as a real tensor input.

    `unique_ios()` already drops empty `TensorSpec` placeholders from each
    captured `input_spec`, so a key being absent here means "this param had no
    real tensor in this case".  Filtering by presence is what makes the case-0
    schema have no `past_key_values` while the case-1 schema does.
    """
    out: List[FwdSpec] = []
    for fs in fwdspecs:
        if fs.name not in input_spec:
            continue
        v = input_spec[fs.name]
        if v is None:
            continue
        if isinstance(v, TensorSpec) and v.is_empty:
            continue
        out.append(fs)
    return out


def _kv_specs_from_input(input_spec: INPUT_SPECS_TYPE) -> Optional[Tuple[TensorSpec, TensorSpec]]:
    """If a KV cache param was captured as a populated (K, V) pair, return it."""
    for key in _CACHE_PARAMS:
        kv = input_spec.get(key)
        if (
            isinstance(kv, (list, tuple))
            and len(kv) == 2
            and all(isinstance(t, TensorSpec) and not t.is_empty for t in kv)
        ):
            return kv[0], kv[1]
    return None


def _augment_output_with_kv(
    output_spec: OUTPUT_SPECS_TYPE,
    input_spec: INPUT_SPECS_TYPE,
) -> OUTPUT_SPECS_TYPE:
    """For cases that consume a KV cache, append (K, V) to the op's outputs.

    The K/V specs are sourced from the captured input cache shapes — the
    abstract impl will emit tensors of those shapes, and the ONNX node will
    expose the updated cache as additional outputs.
    """
    kv = _kv_specs_from_input(input_spec)
    if kv is None:
        return output_spec
    return tuple(output_spec) + kv


def _make_plugin_forward(
    orig_cls: Any,
    module: torch.nn.Module,
):
    """Build a replacement ``forward`` that dispatches per export case.

    - ``export_case`` is None → call the original forward unchanged.
    - ``export_case`` is i    → flatten args via the case-i fwdspecs and call
      the registered op stored on ``module._case_ops[i]``.

    For cases where the op was augmented with KV outputs, the result is
    repackaged into ``(attn_out, (K, V))`` so the ONNX node's KV outputs
    survive downstream consumption.
    """
    sig = inspect.signature(orig_cls.forward)
    params_no_self = [p for n, p in sig.parameters.items() if n != "self"]
    sig_wo_self = sig.replace(parameters=params_no_self)

    def plugin_forward(self_mod, *args, **kwargs):
        case = export_case.get()
        if case is None:
            return orig_cls.forward(module, *args, **kwargs)

        case_ops = getattr(self_mod, "_case_ops", None)
        if case_ops is None or case >= len(case_ops):
            raise RuntimeError(
                f"No op registered for case {case} on {orig_cls.__name__}. "
                f"Available cases: {0 if case_ops is None else len(case_ops)}"
            )
        op, case_specs, has_kv = case_ops[case]

        bound = sig_wo_self.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        flat = _flatten_call_args(case_specs, bound.arguments)
        result = op(id(self_mod), *flat)

        if has_kv:
            # result = (attn_out, K, V[, ...]) → repackage to (attn_out, (K, V))
            attn_out = result[0]
            k_v = (result[1], result[2])
            return attn_out, k_v

        # Match the original `(output, weights)` 2-tuple convention if applicable
        ret_ann = sig.return_annotation
        if isinstance(ret_ann, type) and issubclass(ret_ann, torch.Tensor):
            return result
        if isinstance(result, tuple):
            return result + (None,) * (2 - len(result))
        return result, None

    return plugin_forward


class PluginRegisterInterface(ABC):
    def __init__(self):
        self._registered_torchlib_opname: set = set()
        self.custom_onnx_translation: Dict[Any, Any] = {}
        check_parent_field(self, "_id2module")
        check_parent_field(self, "_module2name")

    @staticmethod
    def create_schema_str(op_name: str, fwdspecs: List[FwdSpec], num_outputs: int) -> str:
        parts = ["int module_id"]
        for fs in fwdspecs:
            if fs.kind == "tensor":
                parts.append(f"Tensor {fs.name}")
            elif fs.kind == "optional_tensor":
                parts.append(f"Tensor? {fs.name}")
            elif fs.kind == "tuple_tensor":
                for i in range(fs.count):
                    parts.append(f"Tensor {fs.name}_{i}")
        ret = "Tensor" if num_outputs == 1 else f"({', '.join('Tensor' for _ in range(num_outputs))})"
        return f"{op_name}({', '.join(parts)}) -> {ret}"

    def _register_torchlib_op(
        self,
        torchlib_op_name: str,
        onnx_op_name: str,
        fwdspecs: List[FwdSpec],
        input_spec: INPUT_SPECS_TYPE,
        output_spec: OUTPUT_SPECS_TYPE,
    ) -> None:
        """Register an export-only op (schema + fake impl + ONNX translation).

        No CPU kernel is registered: these ops exist purely to be captured by
        ``torch.onnx.export``.  The trace path goes through the fake impl
        (shape inference) and the ONNX translation (node emission); eager
        dispatch on real CPU tensors will deliberately raise.
        """
        if torchlib_op_name in self._registered_torchlib_opname:
            return
        self._registered_torchlib_opname.add(torchlib_op_name)

        # Append (K, V) to outputs when this case consumes a populated KV cache
        output_spec = _augment_output_with_kv(output_spec, input_spec)

        _leaves, _ = torch.utils._pytree.tree_flatten(output_spec)
        flat_os = [v for v in _leaves if isinstance(v, TensorSpec) and not v.is_empty]
        num_outputs = len(flat_os)
        assert num_outputs > 0, (
            f"`output_spec` for {torchlib_op_name} flattened to zero TensorSpec leaves."
        )

        fwdspecs = [fs.resolve_unknown(input_spec) for fs in fwdspecs]
        schema = self.create_schema_str(torchlib_op_name, fwdspecs, num_outputs)
        CUSTOM_LIB.define(schema)

        @torch.library.register_fake(f"{CUSTOM_LIB_NAME}::{torchlib_op_name}")
        def _abstract_impl(module_id, *flat_tensors):
            device = next(t for t in flat_tensors if t is not None).device
            outs = tuple(
                torch.empty(ts.shape, dtype=ts.torch_dtype, device=device)
                for ts in flat_os
            )
            return outs[0] if num_outputs == 1 else outs

        def _onnx_translation(module_id, *flat_tensors):
            inputs = [t for t in flat_tensors if t is not None]
            node = onnx_ir.Node(
                domain=ONNX_DOMAIN_NAME,
                op_type=onnx_op_name,
                inputs=inputs,
                attributes=[
                    onnx_ir.AttrString("torchlib_op_name", torchlib_op_name),
                ],
                num_outputs=num_outputs,
            )
            _onnx_core.current_tracer.nodes.append(node)
            return node.outputs[0] if num_outputs == 1 else tuple(node.outputs)

        op_overload = getattr(getattr(torch.ops.hf_module2plugin, torchlib_op_name), "default")
        self.custom_onnx_translation[op_overload] = _onnx_translation

    def register_plugins(self):
        """Patch each plugin module's ``forward`` and register one op per case.

        For every module in ``get_plugin_modules()``:
            - Pull ``unique_ios()`` for that ``(cls_name, module_name)`` — for
              CausalLM this is 2 cases (prefill, decode); generically N cases.
            - Per case, filter the signature's fwdspecs against the captured
              ``input_spec`` so case 0 (no KV) and case 1 (with KV) get
              distinct schemas.
            - Register the op (schema + fake + ONNX translation).
            - Attach ``_case_ops`` to the module so ``plugin_forward`` can
              dispatch by ``export_case.get()``.
        """
        name2modules_dict = self.get_plugin_modules(plugin_suffix=self.plugin_suffix)
        for suffix, modules in name2modules_dict.items():
            onnx_op_name = f"Custom{suffix}"
            if not modules:
                continue  # TODO: logger warning
            cls_names = set(m.__class__.__name__ for m in modules)
            if len(cls_names) > 1:
                raise RuntimeError(
                    f"More than one class candidate for suffix '{suffix}': {cls_names}"
                )

            for orig_m in modules:
                orig_cls = orig_m.__class__
                name = self._module2name[orig_m]
                sig = inspect.signature(orig_cls.forward)
                full_fwdspecs = sig2fwdspecs(sig)

                unique_io_cases = self.plugin_ios[
                    f"{orig_cls.__name__}::{name}"
                ].unique_ios()
                orig_m._trace_metadata = unique_io_cases

                case_ops: List[Tuple[Any, List[FwdSpec], bool]] = []
                for i, (input_spec, output_spec) in enumerate(unique_io_cases):
                    case_fwdspecs = _filter_fwdspecs_by_input(full_fwdspecs, input_spec)
                    case_fwdspecs = [fs.resolve_unknown(input_spec) for fs in case_fwdspecs]
                    has_kv = _kv_specs_from_input(input_spec) is not None
                    # Share the torchlib op across module instances of the same
                    # class (same shapes → same schema), distinguishing only by
                    # case index.
                    torchlib_op_name = f"{orig_cls.__name__}_case{i}"
                    self._register_torchlib_op(
                        torchlib_op_name=torchlib_op_name,
                        onnx_op_name=onnx_op_name,
                        fwdspecs=case_fwdspecs,
                        input_spec=input_spec,
                        output_spec=output_spec,
                    )
                    op = getattr(torch.ops.hf_module2plugin, torchlib_op_name)
                    case_ops.append((op, case_fwdspecs, has_kv))

                orig_m._case_ops = case_ops

                traceable_fwd = _make_plugin_forward(orig_cls=orig_cls, module=orig_m)
                orig_m.__class__ = type(
                    f"Traceable{orig_cls.__name__}",
                    (orig_cls,),
                    {"forward": traceable_fwd},
                )


__all__ = [
    "CUSTOM_LIB_NAME",
    "PluginRegisterInterface",
    "export_case",
]
