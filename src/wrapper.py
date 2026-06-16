# TODO: as we support more architectures expand this module into submodule
import inspect
from collections import defaultdict
from typing import Sequence, Dict, Any, Tuple, List

import torch
import transformers

from utils import convert_dtype

class CausalLMWrapper:
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        model_inputs: Dict[str, Any],
        plugin_suffix: Sequence[str] = ("Attention", "RotaryEmbedding"),  # TODO: make enum object
        **generate_kwargs,
    ):
        self.model = model  # TODO: make sure .generate method exist (e.g. only allow ForCausalLM)
        self.plugin_suffix = plugin_suffix
        
        # Maps memory addresses (module instances) to hierarchical string names
        self._module2name = {mod: name for name, mod in model.named_modules()}
        
        # Container to store captured input shapes/dtypes
        self._plugin_inputs: Dict[str, Dict[str, List[Dict[str, tuple]]]] = defaultdict(lambda: defaultdict(list))
        
        # Internal storage to manage active hook attachment states
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []

        # Pipeline begins here ...
        self.apply_plugin_wrappers()
        self._attach_hooks()
        self.model.generate(**model_inputs, **generate_kwargs)  # type: ignore[reportAttributeAccessIssue]
        self._detach_hooks()

    @property
    def captured_plugin_inputs(self) -> Dict[str, Dict[str, List[Dict[str, tuple]]]]:
        return self._plugin_inputs

    def apply_plugin_wrappers(self):
        """Force nn.Module inputs to be flattened out"""
        HF_CONFIG = self.model.config

        if "Attention" in self.plugin_suffix:
            KV_CACHE_VAR_NAME = "past_key_values"
            
            # ===== Register DynamicCache Pytree Node ===== #
            def cache_flatten(past_kv):
                flat_tensors = tuple(e for cache in past_kv.layers for e in (cache.keys, cache.values))
                metadata = {"num_layers": len(past_kv.layers)}
                return flat_tensors, metadata
            def cache_unflatten(flat_tensors, metadata):
                if any(t is None for t in flat_tensors):
                    past_kv = transformers.DynamicCache(config=HF_CONFIG)
                past_kv = transformers.DynamicCache()
                for i in range(0, len(flat_tensors), 2):
                    layer_idx = i // 2
                    k = flat_tensors[i]
                    v = flat_tensors[i+1]
                    past_kv.update(k, v, layer_idx)
                return past_kv
            torch.utils._pytree.register_pytree_node(
                transformers.DynamicCache,
                flatten_fn=cache_flatten,
                unflatten_fn=cache_unflatten
            )

            # ===== Make Attention to take traceable tuple and parse to DynamicCache ===== #
            attn_layers = self.get_plugin_modules(plugin_suffix="Attention")
            _candidate_attn_cls = set(m.__class__.__name__ for m in attn_layers)
            if len(_candidate_attn_cls) > 1:
                raise RuntimeError(f"More than one class candidate detected for suffix `Attention`: {_candidate_attn_cls}")
            for module in attn_layers:
                attn_cls = module.__class__
                forward_sig = inspect.signature(module.forward)
                def attn_forward(self, *_args, **_kwargs):
                    """Custom Attn forward"""
                    bound_args = forward_sig.bind_partial(*_args, **_kwargs)
                    bound_args.apply_defaults()
                    past_kv = bound_args.arguments.get(KV_CACHE_VAR_NAME, None)

                    outputs = attn_cls.forward(self, **bound_args.arguments)
                    assert isinstance(outputs, tuple) and len(outputs) == 2, "Only support the latest transformers"

                    return outputs[0], past_kv  # WARNING: replaced `attn_weight` to `past_kv`. May cause problem

                TraceableAttnCls = type(
                    f"Traceable{attn_cls.__name__}",
                    (attn_cls,), 
                    {"forward": attn_forward}
                )
                module.__class__ = TraceableAttnCls

    def get_plugin_modules(
        self,
        *,
        plugin_suffix: str | Sequence[str] | None = None
    ) -> Dict[str, List[torch.nn.Module]] | List[torch.nn.Module]:
        """Filter out the plugin nn.Module user defined with `plugin_suffix`"""
        plugin_suffix = plugin_suffix or self.plugin_suffix
        if isinstance(plugin_suffix, str):
            plugin_suffix = [plugin_suffix]
        suffix2modules = {}
        for suffix in plugin_suffix:
            module_list = []
            for m in (m for m in self.model.modules() if suffix in m.__class__.__name__):
                module_list.append(m)
            suffix2modules[suffix] = module_list
        if len(suffix2modules) == 1:
            suffix2modules = next(iter(suffix2modules.values()))
        return suffix2modules

    def _observation_hook(self, module: torch.nn.Module, hook_args: Tuple[Any, ...], *args):

        _module_name = self._module2name[module]
        if len(args) == 2:
            # Layout is: (module, input_args, input_kwargs, output)
            actual_args = hook_args
            actual_kwargs = args[0]
        else:
            # Layout is: (module, input_args, input_kwargs)
            actual_args = hook_args
            actual_kwargs = args[0] if args else {}
        sig = inspect.signature(module.forward)
        bound_args = sig.bind(*actual_args, **actual_kwargs)
        bound_args.apply_defaults()
        name2val = bound_args.arguments
        name2val = name2val["_kwargs"] if "_kwargs" in name2val else name2val  # custom forward under `apply_plugin_wrappers`

        _module_inputs = dict()
        for param_name, value in name2val.items():
            if value is None:
                _module_inputs[param_name] = (None, None)
                continue
            if isinstance(value, (int, float, bool)):
                _module_inputs[param_name] = (value, convert_dtype(torch.tensor(value)))
            elif isinstance(value, torch.Tensor):
                _module_inputs[param_name] = (tuple(value.shape), convert_dtype(value))
            elif isinstance(value, transformers.DynamicCache):
                layer_idx = module.layer_idx
                flat_cache, _ = torch.utils._pytree.tree_flatten(value)
                k = flat_cache[layer_idx*2]
                v = flat_cache[layer_idx*2+1]
                _module_inputs[param_name] = ((tuple(k.shape), convert_dtype(k)), (tuple(v.shape), convert_dtype(v)))
            else:
                try:
                    flat_tensors, spec = torch.utils._pytree.tree_flatten(value)
                    _ = torch.utils._pytree.tree_unflatten(((tuple(t.shape), convert_dtype(t)) for t in flat_tensors), spec)
                    _module_inputs[param_name] = _
                except Exception:
                    raise RuntimeError(f"Unknown `{param_name}={value}` when inspecting the hook at `{module.__class__.__name__}`")

        self._plugin_inputs[module.__class__.__name__][_module_name].append(_module_inputs)

    def _attach_hooks(self):
        """Attaches the forward hooks to inspect plugin inputs. Capture name, shape and dtype"""
        self._plugin_inputs.clear()
        self._hook_handles = []
        for module in (m for ml in self.get_plugin_modules().values() for m in ml):
            handle = module.register_forward_hook(self._observation_hook, with_kwargs=True)
            self._hook_handles.append(handle)
    def _detach_hooks(self):
        """Detach the attached forward hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
