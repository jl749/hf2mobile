import transformers
import transformers.models.qwen2.modeling_qwen2 as modeling_qwen2

from .causallm import CausalLMExporter


class Qwen2ForCausalLMExporter(CausalLMExporter):
    def __init__(self, model: transformers.PreTrainedModel):
        # NOTE: transformers/integrations/sdpa_attention.py::`use_gqa_in_sdpa()` returns False if `attention_mask != None`.
        #   This materializes `Expand` onnx nodes(expand KV) on K/V that `AttentionIdentifier` cannot walk past.
        #   Hence, Null the sliding mask that every attention module traces with `.forward(..., attention_mask=None)`
        #   forcing `F.scaled_dot_product_attentio(..., enable_gqa=True)` and native onnx Attention node when exported.
        #   later re-applying the attention mask with `attach_sliding_window_mask_onnx` function.
        modeling_qwen2.create_sliding_window_causal_mask = lambda *a, **k: None

        super().__init__(model, plugin_suffix=("Attention", "RotaryEmbedding"))


__all__ = ["Qwen2ForCausalLMExporter"]
