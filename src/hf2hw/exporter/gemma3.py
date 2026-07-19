import torch
import transformers
import transformers.models.gemma3.modeling_gemma3 as modeling_gemma3

from .causallm import CausalLMExporter


class Gemma3RotaryEmbeddingPinned(torch.nn.Module):
    """Single-layer_type view over Gemma3's shared dual-frequency rotary module.

    Gemma3 computes RoPE with one `Gemma3RotaryEmbedding` called once per layer type —
    `forward(x, position_ids, layer_type)` selects the `{layer_type}_inv_freq` buffer
    (global `rope_theta` for full layers, `rope_local_base_freq` for sliding layers).
    The tracing framework keys plugins by module instance and only traces tensor args,
    so the shared module is split into one pinned child per layer type, each exposing the
    plain `forward(x, position_ids)` signature every other supported arch has.
    """

    def __init__(self, rotary_emb: torch.nn.Module, layer_type: str):
        super().__init__()
        self.config = rotary_emb.config
        self._layer_type = layer_type
        self._shared = (rotary_emb,)  # tuple: keeps the shared module out of the module tree
        #   (a registered `Gemma3RotaryEmbedding` child would also match plugin_suffix)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        return self._shared[0].forward(x, position_ids, self._layer_type)


class Gemma3RopeDispatch(torch.nn.Module):
    """Drop-in `model.rotary_emb` replacement routing each `layer_type` to its pinned child.

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

        # NOTE: sliding mask is attached during postprocessing. keep it causal initially for easier tracing.
        modeling_gemma3.create_causal_mask = lambda *args, **kwargs: None
        modeling_gemma3.create_sliding_window_causal_mask = lambda *args, **kwargs: None

        super().__init__(model, plugin_suffix=("Attention", "RotaryEmbedding"))


__all__ = ["Gemma3ForCausalLMExporter"]
