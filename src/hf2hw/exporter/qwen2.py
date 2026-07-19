import transformers

from .causallm import CausalLMExporter


class Qwen2ForCausalLMExporter(CausalLMExporter):
    def __init__(self, model: transformers.PreTrainedModel):
        super().__init__(model, plugin_suffix=("Attention", "RotaryEmbedding"))


__all__ = ["Qwen2ForCausalLMExporter"]
