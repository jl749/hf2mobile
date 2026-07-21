import torch
import transformers
import transformers.models.gemma3.modeling_gemma3 as modeling_gemma3

from .causallm import CausalLMExporter


class Gemma3RotaryEmbeddingPinned(torch.nn.Module):
    """
    Single layer_type view over Gemma3's shared dual-frequency rotary module.

    `Gemma3RotaryEmbedding.forward(x, position_ids, layer_type)`
    selects different `{layer_type}_inv_freq` base on layer_type(full or sliding).
    (computed using `rope_theta` for full layers, `rope_local_base_freq` for sliding layers).

    The tracing framework registers plugins by module instance ...
    For every `layer_type` register N Gemma3RotaryEmbeddingPinned child modules under Gemma3RopeDispatch.
    Each child module will expose plain `forward(x, position_ids)` signature.
    """

    def __init__(self, rotary_emb: torch.nn.Module, layer_type: str):
        super().__init__()
        self.config = rotary_emb.config
        self._layer_type = layer_type

        # NOTE: keep the shared module out of the module tree by storing it under Tuple
        #   prevents `Gemma3RotaryEmbedding` torchlib registration
        self._shared = (rotary_emb,)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        return self._shared[0].forward(x, position_ids, self._layer_type)


class Gemma3RopeDispatch(torch.nn.Module):
    """
    Drop-in `model.rotary_emb` replacement routing each `layer_type` to its pinned child.

    The class name intentionally avoids the "RotaryEmbedding" plugin suffix: only the
    pinned children (one per unique layer type) are traced and exported as subgraphs.
    """

    def __init__(self, rotary_emb: torch.nn.Module, layer_types: list):
        super().__init__()
        for layer_type in dict.fromkeys(layer_types):
            self.add_module(layer_type, Gemma3RotaryEmbeddingPinned(rotary_emb, layer_type))

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, layer_type: str):
        return getattr(self, layer_type)(x, position_ids)


class Gemma3ForCausalLMExporter(CausalLMExporter):
    def __init__(self, model: transformers.PreTrainedModel):
        # NOTE: split the shared dual-frequency rotary module before TracerInterface builds its module maps.
        text_model = model.model
        text_model.rotary_emb = Gemma3RopeDispatch(text_model.rotary_emb, model.config.layer_types).eval()

        # NOTE: transformers/integrations/sdpa_attention.py::`use_gqa_in_sdpa()` returns False if `attention_mask != None`.
        #   This materializes `Expand` onnx nodes(expand KV) on K/V that `AttentionIdentifier` cannot walk past.
        #   Hence, Null the sliding mask that every attention module traces with `.forward(..., attention_mask=None)`
        #   forcing `F.scaled_dot_product_attentio(..., enable_gqa=True)` and native onnx Attention node when exported.
        #   later re-applying the attention mask with `attach_sliding_window_mask_onnx` function.
        modeling_gemma3.create_sliding_window_causal_mask = lambda *args, **kwargs: None

        super().__init__(model, plugin_suffix=("Attention", "RotaryEmbedding"))


__all__ = ["Gemma3ForCausalLMExporter"]
