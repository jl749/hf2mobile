import time

import torch
import transformers

from hf2hw.constant import KV_CACHE_PARAM_NAME


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


_CACHE_PYTREE_REGISTERED = False


def register_dynamic_cache_pytree() -> None:
    """
    Register DynamicCache as a pytree whose direct children are per-layer (K, V) pairs.
    tree_flatten(cache) -> leaves [K0, V0, K1, V1, ...]
    tree_unflatten([K0, V0, K1, V1, ...]) -> DynamicCache
    """
    global _CACHE_PYTREE_REGISTERED
    if _CACHE_PYTREE_REGISTERED:
        return

    def _flatten(cache: transformers.DynamicCache):
        return [(layer.keys, layer.values) for layer in cache.layers], len(cache.layers)

    def _unflatten(pairs, num_layers):
        return transformers.DynamicCache(ddp_cache_data=list(pairs))

    try:
        torch.utils._pytree.register_pytree_node(transformers.DynamicCache, _flatten, _unflatten)
    except ValueError:
        raise RuntimeError("Pytree already registered for Dynamic cache (likely newer transformers)")
    _CACHE_PYTREE_REGISTERED = True


def update_input_cache(input_dict: dict, layer_idx: int) -> str | None:
    """
    `Attentino.forward` parameter `past_key_values: Cache` contains every global cache even after pytree flattening
    This function updates `input_dict` so that KV_CACHE_PARAM_NAME("past_key_values") only contains the local cache index
    Args:
        input_dict: .forward inputs with default entries
            `dict(sig.bound_partial(*args, **kwargs).apply_defaults().arguments())`
        layer_idx: in order to extract the local cache from `transformers.Cache` object
            we need to know the exact layer_idx (local cache location)
    Returns:
        updated input KV cache param name or None
    """
    cache_param_name = None
    value = input_dict.get(KV_CACHE_PARAM_NAME, None)
    if isinstance(value, transformers.Cache):
        try:
            flat, _ = torch.utils._pytree.tree_flatten(value)
        except:
            raise RuntimeError(
                "Cannot flatten Cache object. Please make sure `register_dynamic_cache_pytree` is called in advance."
            )
        kv_cache_tuple = (flat[2 * layer_idx], flat[2 * layer_idx + 1])
        input_dict[KV_CACHE_PARAM_NAME] = kv_cache_tuple
        cache_param_name = KV_CACHE_PARAM_NAME
    return cache_param_name


__all__ = ["TokenSpeedStreamer", "register_dynamic_cache_pytree", "update_input_cache"]
