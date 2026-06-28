from collections import defaultdict
import contextvars
import inspect
from abc import ABC
from typing import Any, Dict, List, Optional, Tuple, Set

import onnx_ir
import torch
from torch.onnx._internal.exporter import _core as _onnx_core

from .tensor_metadata import TensorSpec, INPUT_SPECS_TYPE, OUTPUT_SPECS_TYPE, get_kv_specs_from_input_specs
from .inspect import FwdSpec, apply_input_specs2fwd_specs, sig2fwdspecs, fwdspecs2args
from constant import CUSTOM_LIB_NAME, CUSTOM_LIB, ONNX_DOMAIN_NAME, _NON_HASHABLE_PARAMS
from utils.py_helper import check_parent_field


# Case index threaded by `export_graphs` so each module's `plugin_forward`
# picks the right per-case op.  `None` means generation/inference (no export
# in progress) — `plugin_forward` falls straight through to `orig_cls.forward`.
export_case: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "export_case", default=None
)


def _make_plugin_forward(
    orig_cls: Any,
    module: torch.nn.Module,
):
    """Build a replacement ``forward`` that dispatches per export case.

    - ``export_case`` is None → call the original forward unchanged.
    - ``export_case`` is i    → flatten args via the case-i fwd_specs and call
      the registered op stored on ``module._case_ops[i]``.

    For cases where the op was augmented with KV outputs, the result is
    repackaged into ``(attn_out, (K, V))`` so the ONNX node's KV outputs
    survive downstream consumption.
    """
    sig = inspect.signature(orig_cls.forward)
    params_no_self = [p for n, p in sig.parameters.items() if n != "self"]
    sig_wo_self = sig.replace(parameters=params_no_self)

    def _plugin_forward(self_module, *args, **kwargs):
        case = export_case.get()
        if case is None:
            return orig_cls.forward(module, *args, **kwargs)

        # NOTE: unwrap _trace_metadata from `self_module`
        trace_metadata = getattr(self_module, "_trace_metadata", None)
        if trace_metadata is None:
            raise RuntimeError(
                f"No trace metadata has been registered for `{case=}` on `{orig_cls.__name__}`(id={id(module)})."
                f"Pleae check the logics under `PluginRegisterInterface.register_plugins(...)`."
            )
        elif case >= len(trace_metadata):
            raise RuntimeError(
                f"No op registered for case `{case=}` on `{orig_cls.__name__}`(id={id(module)}). "
                f"Number of the observed trace cases: {len(trace_metadata)}"
            )
        _metadata = trace_metadata[case]
        op = _metadata["op"]
        case_fwd_specs = _metadata["fwd_specs"]
        has_kv = _metadata["has_kv"]

        bound = sig_wo_self.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        bound_args = dict(bound.arguments)

        # KV-cache params arrive as `transformers.Cache` objects but the op
        # signature expects (K, V) tensors.  Pull this layer's K, V out before
        # flattening.
        cache_param_name = None
        for fs in case_fwd_specs:
            if fs.name in _NON_HASHABLE_PARAMS:
                val = bound_args.get(fs.name)
                if val is not None and hasattr(val, "layers"):
                    layer_idx = getattr(self_module, "layer_idx", 0)
                    layer_cache = val.layers[layer_idx]
                    bound_args[fs.name] = (layer_cache.keys, layer_cache.values)
                    cache_param_name = fs.name
                break

        flat = fwdspecs2args(case_fwd_specs, bound_args)
        result = op(id(self_module), *flat)

        if has_kv:
            # result = (attn_out, K, V[, ...]) → repackage to (attn_out, (K, V))
            attn_out = result[0]
            k_new, v_new = result[1], result[2]
            # Write back so the outer wrapper (which reads cache.layers[i].keys/.values
            # after model.forward returns) sees the *new* K, V flowing to the graph
            # outputs.  Without this the K, V get DCE'd from the final ONNX graph.
            if cache_param_name is not None:
                cache = bound.arguments.get(cache_param_name)
                if cache is not None and hasattr(cache, "layers"):
                    layer_idx = getattr(self_module, "layer_idx", 0)
                    cache.layers[layer_idx].keys = k_new
                    cache.layers[layer_idx].values = v_new
            return attn_out, (k_new, v_new)

        # Match the original `(output, weights)` 2-tuple convention if applicable
        ret_ann = sig.return_annotation
        if isinstance(ret_ann, type) and issubclass(ret_ann, torch.Tensor):
            return result
        if isinstance(result, tuple):
            return result + (None,) * (2 - len(result))
        return result, None

    return _plugin_forward


class PluginRegisterInterface(ABC):
    def __init__(self):
        self.onnxop2torchlibop: Dict[str, Set[str]] = defaultdict(set)
        self.torchlibop2onnxop: Dict[str, str] = {}
        self.custom_onnx_translation: Dict[Any, Any] = {}
        check_parent_field(self, "_id2module")
        check_parent_field(self, "_module2name")

    @staticmethod
    def _create_schema_str(torchlib_op_name: str, fwd_specs: List[FwdSpec], num_outputs: int) -> str:
        """
        Create a torch library function schema string in order to define a new custom operator
        Args:
            torchlib_op_name: custom op name
            fwd_specs: actual fwd signatures returned by 
                `apply_input_specs2fwd_specs(fwd_specs, `ModuleIOSpec.unique_ios()[i][0]`)
            num_outputs: length of the `ModuleIOSpec.unique_ios()` filtered output_specs
        Returns:
            schema string
        """
        parts = ["int module_id"]
        for fs in fwd_specs:
            if fs.kind == "tensor":
                parts.append(f"Tensor {fs.name}")
            elif fs.kind == "optional_tensor":
                parts.append(f"Tensor? {fs.name}")
            elif fs.kind == "tuple_tensor":
                for i in range(fs.count):
                    parts.append(f"Tensor {fs.name}_{i}")
        ret = "Tensor" if num_outputs == 1 else f"({', '.join('Tensor' for _ in range(num_outputs))})"
        return f"{torchlib_op_name}({', '.join(parts)}) -> {ret}"

    def _register_custom_op(
        self,
        torchlib_op_name: str,
        onnx_op_name: str,
        fwd_specs: List[FwdSpec],
        output_specs: OUTPUT_SPECS_TYPE,
    ) -> None:
        """
        Register an export-only op (schema + fake impl + ONNX translation).
        No CPU kernel is registered: these ops exist purely to be captured by `torch.onnx.export`.  
        The trace path goes through the fake impl (shape inference) and the ONNX translation (node emission)

        Args:
            torchlib_op_name: custom op name that defines random `torch.nn.Module`
            onnx_op_name: custom onnx op_type name to map `torchlib_op_name`
            fwd_specs: actual fwd signatures returned by 
                `apply_input_specs2fwd_specs(fwd_specs, `ModuleIOSpec.unique_ios()[i][0]`)
            output_specs: module outputs observed by the hook
                `_, output_specs = ModuleIOSpec.unique_ios()[i]`
        """
        _err_msg = f"While registering {torchlib_op_name},"

        _onnx_op_name_mapped = self.torchlibop2onnxop.get(torchlib_op_name, None)
        if _onnx_op_name_mapped:
            assert _onnx_op_name_mapped == onnx_op_name, f"{_err_msg} we noticed `{torchlib_op_name=}` is already mapped to `onnx_op_name={_onnx_op_name_mapped}`. However, user is trying to map it again to `{onnx_op_name=}`"
            return
        
        leaves, _ = torch.utils._pytree.tree_flatten(output_specs)
        flat_os: List[TensorSpec] = [v for v in leaves]
        num_outputs = len(flat_os)

        assert all(isinstance(spec, TensorSpec) for spec in flat_os), f"{_err_msg} we detected non `TesnorSpec` object under `output_specs`. This is likely a bug please report it with repro codes to the dev."
        assert all(spec.is_empty is False for spec in flat_os), f"{_err_msg} we detected `output_specs` is containing empty `TensorSpec`. Please call `ModuleIOSpec.unique_ios()` before running `_register_custom_op` in order to obtain None filtered `TensorSpec` observations."
        assert num_outputs > 0, (
            f"{_err_msg} we noticed `output_specs` is containing 0 `TensorSpec`. In order to trace the ONNX at least one output `TensorSpec` is required."
        )

        schema = self._create_schema_str(torchlib_op_name, fwd_specs, num_outputs)
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
            # TODO: set attributes by reading self._id2module attributes
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

        # e.g. <OpOverload(op='hf_module2plugin.Qwen3Attention____model__layers__0__self_attn____case0', overload='default')>
        op_overload = getattr(getattr(torch.ops.hf_module2plugin, torchlib_op_name), "default")
        self.custom_onnx_translation[op_overload] = _onnx_translation

        # NOTE: mark both `torchlib_op_name` and `onnx_op_name` as registered
        self.torchlibop2onnxop[torchlib_op_name] = onnx_op_name
        self.onnxop2torchlibop[onnx_op_name].add(torchlib_op_name)

    def register_plugins(self):
        """Patch each plugin module's ``forward`` and register one op per case.

        For every module in ``get_plugin_modules()``:
            - Pull ``unique_ios()`` for that ``(cls_name, module_name)`` — for
              CausalLM this is 2 cases (prefill, decode); generically N cases.
            - Per case, filter the signature's fwd_specs against the captured
              ``sinput_spec`` so case 0 (no KV) and case 1 (with KV) get
              distinct schemas.
            - Register the op (schema + fake + ONNX translation).
            - Attach ``_case_ops`` to the module so ``plugin_forward`` can
              dispatch by ``export_case.get()``.
        """
        name2modules_dict = self.get_plugin_modules(plugin_suffix=self.plugin_suffix)
        if isinstance(name2modules_dict, list):
            name2modules_dict = {self.plugin_suffix[0]: name2modules_dict}
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
                sig = inspect.signature(orig_cls.forward)
                fwd_specs: List[FwdSpec] = sig2fwdspecs(sig)

                name = self._module2name[orig_m]
                unique_io_cases = self.plugin_ios[f"{orig_cls.__name__}::{name}"].unique_ios()

                trace_metadata: List[Dict[str, Any]] = []
                for i, (input_specs, output_specs) in enumerate(unique_io_cases):
                    case_fwd_specs = apply_input_specs2fwd_specs(fwd_specs, input_specs)
                    torchlib_op_name = f"{orig_cls.__name__}____{name.replace('.', '__')}____case{i}"

                    # NOTE: in case KV caches are observed under `input_specs` append KV to `output_specs` (ONNX tracing purpose)
                    _kv_specs = get_kv_specs_from_input_specs(input_specs)
                    output_specs = output_specs + _kv_specs

                    self._register_custom_op(
                        torchlib_op_name=torchlib_op_name,
                        onnx_op_name=onnx_op_name,
                        fwd_specs=case_fwd_specs,
                        output_specs=output_specs,
                    )

                    # e.g. <OpOverloadPacket(op='hf_module2plugin.Qwen3Attention____model__layers__0__self_attn____case0')>
                    op = getattr(torch.ops.hf_module2plugin, torchlib_op_name)
                    trace_metadata.append({
                        "op": op, 
                        "fwd_specs": case_fwd_specs, 
                        "unique_io_specs": (input_specs, output_specs),
                        "has_kv": len(_kv_specs) > 0
                    })
                orig_m._trace_metadata = trace_metadata

                breakpoint()
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
