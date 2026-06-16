import time
import inspect
from collections import defaultdict
from typing import Sequence, Dict, Any, Tuple, List, Set

import torch
import transformers

class TokenSpeedStreamer(transformers.generation.streamers.BaseStreamer):
    def __init__(self):
        self.token_count = 0
        self.start_time: float | None = None

    def put(self, value):
        """Called every time the model generates a new token/batch."""
        if self.start_time is None:
            self.start_time = time.perf_counter()
            return

        num_tokens = value.numel()
        self.token_count += num_tokens
        
        elapsed = time.perf_counter() - self.start_time
        if elapsed > 0:
            current_speed = self.token_count / elapsed
            print(f"\rGenerated: {self.token_count} tokens | Speed: {current_speed:.2f} tok/sec", end="")

    def end(self):
        """Called when generation finishes."""
        if self.start_time is None:
            raise RuntimeError("self.start_time was never set!")
        elapsed = time.perf_counter() - self.start_time
        final_speed = self.token_count / elapsed
        print(f"\n\n✨ Done! Final Average Speed: {final_speed:.2f} tok/sec")


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
        self.MODULE2NAME = {mod: name for name, mod in model.named_modules()}
        
        # Container to store captured input shapes/dtypes
        self.NAME2INPUT: Dict[str, Set[tuple]] = defaultdict(set)
        
        # Internal storage to manage active hook attachment states
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []

        # Pipeline begins here ...
        self.apply_plugin_wrappers()
        self._attach_hooks()
        self.model.generate(**model_inputs, **generate_kwargs)  # type: ignore[reportAttributeAccessIssue]
        self._detach_hooks()

    def apply_plugin_wrappers(self):
        """Force nn.Module inputs to be flattened out"""
        HF_CONFIG = self.model.config

        if "Attention" in self.plugin_suffix:
            KV_CACHE_VAR_NAME = "past_key_values"
            
            def unwrap_cache(past_kv):
                assert isinstance(past_kv, transformers.DynamicCache)
                return tuple((c.keys, c.values) for c in past_kv.layers)
            def wrap_cache(traceable_tuple):
                past_kv = transformers.DynamicCache()
                for layer_idx, (key_states, value_states) in enumerate(traceable_tuple):
                    past_kv.update(key_states, value_states, layer_idx)
                return past_kv

            # ===== Make Attention to take traceable tuple and parse to DynamicCache ===== #
            attn_layers = self.get_plugin_modules(plugin_suffix="Attention")
            _candidate_attn_cls = set(m.__class__.__name__ for m in attn_layers)
            if len(_candidate_attn_cls) > 1:
                raise RuntimeError(f"More than one class candidate detected for suffix `Attention`: {_candidate_attn_cls}")
            for module in attn_layers:
                attn_cls = module.__class__
                forward_sig = inspect.signature(module.forward)
                def attn_forward(self, *_args, **_kwargs):
                    bound_args = forward_sig.bind_partial(*_args, **_kwargs)
                    bound_args.apply_defaults()
                    traceable_tuple = bound_args.arguments.get(KV_CACHE_VAR_NAME, None)
                    if traceable_tuple is None:
                        past_kv = None
                    else:  # traceable_tuple -> Cache
                        if any(e is None for tup in traceable_tuple for e in tup):
                            past_kv = transformers.DynamicCache(config=HF_CONFIG)
                        else:
                            past_kv = wrap_cache(traceable_tuple)
                        bound_args.arguments[KV_CACHE_VAR_NAME] = past_kv

                    outputs = attn_cls.forward(self, **bound_args.arguments)
                    assert isinstance(outputs, tuple), "Only support the latest transformers"

                    self._output_kv_tobe_popped = unwrap_cache(past_kv)

                    return outputs

                TraceableAttnCls = type(
                    f"Traceable{attn_cls.__name__}",
                    (attn_cls,), 
                    {"forward": attn_forward}
                )
                module.__class__ = TraceableAttnCls
            # ===== Make DecoderLayer to pass traceable tuple to Attention module ===== #
            decode_layers = self.get_plugin_modules(plugin_suffix="DecoderLayer")
            _candidate_decode_cls = set(m.__class__.__name__ for m in decode_layers)
            if len(_candidate_decode_cls) > 1:
                raise RuntimeError(f"More than one class candidate detected for suffix `DecoderLayer`: {_candidate_decode_cls}")
            for module in decode_layers:
                decoder_cls = module.__class__
                forward_sig = inspect.signature(module.forward)
                def decoder_forward(self, *_args, **_kwargs):
                    bound_args = forward_sig.bind_partial(*_args, **_kwargs)
                    bound_args.apply_defaults()
                    past_kv = bound_args.arguments.get(KV_CACHE_VAR_NAME, None)
                    if past_kv is not None:  # Cache -> traceable_tuple
                        bound_args.arguments[KV_CACHE_VAR_NAME] = unwrap_cache(past_kv)

                    outputs = decoder_cls.forward(self, **bound_args.arguments)

                    traceable_tuple = getattr(self.self_attn, "_output_kv_tobe_popped")
                    delattr(self.self_attn, "_output_kv_tobe_popped")
                    if any(e is None for tup in traceable_tuple for e in tup):
                        past_kv = transformers.DynamicCache(config=HF_CONFIG)
                    else:
                        past_kv = wrap_cache(traceable_tuple)

                    return outputs

                TraceableDecoderCls = type(
                    f"Traceable{decoder_cls.__name__}",
                    (decoder_cls,), 
                    {"forward": decoder_forward}
                )
                module.__class__ = TraceableDecoderCls

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
        _module_name = self.MODULE2NAME[module]
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
        if "_kwargs" in name2val:  # custom forward under `apply_plugin_wrappers`
            name2val = name2val["_kwargs"]
            name2val.pop("kwargs")

        _module_inputs = set()
        for param_name, value in name2val.items():
            if value is None:
                _module_inputs.add((param_name, None, None))
                continue
            if isinstance(value, (int, float)):
                value = torch.tensor(value)
            if isinstance(value, torch.Tensor):
                _module_inputs.add((param_name, tuple(value.shape), value.dtype))
            elif isinstance(value, (list, tuple)) and all(isinstance(x, torch.Tensor) for x in value):
                _module_inputs.add((param_name, ((tuple(x.shape), x.dtype) for x in value if isinstance(x, torch.Tensor))))

        self.NAME2INPUT[_module_name] = _module_inputs

    def _attach_hooks(self):
        """Attaches the forward hooks to inspect plugin inputs. Capture name, shape and dtype"""
        self.NAME2INPUT.clear()
        self._hook_handles = []
        for module in (m for ml in self.get_plugin_modules().values() for m in ml):
            handle = module.register_forward_hook(self._observation_hook, with_kwargs=True)
            self._hook_handles.append(handle)
    def _detach_hooks(self):
        """Detach the attached forward hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()


def main():
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
    ).cpu()

    prompt = "Give me a short introduction to large language model./think"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True  # by default
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    model_wrapper = CausalLMWrapper(
        model, 
        model_inputs,
        plugin_suffix=("Attention", "RotaryEmbedding")
    )
    # for layer_path, tensor_meta in list(inspector.NAME2INPUT.items())[:2]:
    #     print(f"\n📍 Layer: {layer_path}")
    #     for param, attributes in tensor_meta.items():
    #         print(f"   └── Param: {param:<15} -> Meta: {attributes}")
    # output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    # print(tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n"))


if __name__ == "__main__":
    main()
