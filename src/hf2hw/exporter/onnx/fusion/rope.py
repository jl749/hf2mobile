"""
RotaryEmbedding fusion (fuse_rope)
───────────────────────────────────
Pattern (rotate_half RoPE from torch.onnx.export):

    Mul(x, cos_us)                         # x * cos
    Slice(x, 0, half, axis=-1)             # x[:half]
    Slice(x, half, end, axis=-1)           # x[half:]
    Neg(x_second)                          # -x[half:]
    Concat(neg, x_first, axis=-1)          # rotate_half(x)
    Mul(rotated, sin_us)                   # rotate_half(x) * sin
    Add(x_cos, x_sin)                      # output

Replaced with (custom domain):
    RotaryEmbedding(x, cos_us, sin_us)

cos_us / sin_us are typically Unsqueeze outputs from a shared Unsqueeze node
(added by PyTorch to broadcast over the heads dimension).  The replacement looks
through those Unsqueeze nodes to consume the raw cos/sin graph inputs directly
(the Unsqueeze nodes become dead code).  The rule fires once for Q and once for K.
Afterwards the cos/sin graph-input VIs are halved on the last axis: the opset-23
RotaryEmbedding takes head_dim//2 caches, not the traced full-head_dim tensors.
"""

from typing import Tuple

import onnx
import onnx_ir as ir
from onnxscript import rewriter
from onnxscript.rewriter import pattern

from hf2hw.utils.logger import logger
from hf2hw.utils.onnx_helper import get_vi_axis, set_vi_axis, update_opset


def _unwrap_unsqueeze(val: ir.Value) -> ir.Value | None:
    """Return the input of a preceding Unsqueeze node, or None."""
    node = val.producer()
    if node is not None and node.op_type == "Unsqueeze":
        return node.inputs[0]
    return None


def _rope_rule() -> pattern.RewriteRule:
    """rotate_half RoPE → opset-23 RotaryEmbedding(X, cos_cache, sin_cache).

    The pattern binds ``cos_us`` / ``sin_us`` to the Unsqueeze outputs that
    PyTorch inserts to broadcast over heads.  The replacement looks through
    those Unsqueeze nodes to recover the raw ``(N, L, E)`` cos/sin caches,
    which match the standard ONNX opset-23 RotaryEmbedding signature directly
    (``position_ids`` is optional and omitted).  Both Unsqueeze nodes become
    dead code after both Q and K fusions fire.

    The Slice axis is stored as an absolute index (e.g. 3), not -1, so no
    axis condition is applied — the structural pattern is distinctive enough.
    """

    def pat(
        op: pattern.OpsetPatternBuilder,
        x: pattern.Var,
        cos_us: pattern.Var,
        sin_us: pattern.Var,
        start_a: pattern.Var,
        half_a: pattern.Var,
        end_a: pattern.Var,
        axes_a: pattern.Var,
        steps_a: pattern.Var,
    ):
        x_cos = op.Mul(x, cos_us)
        x_first = op.Slice(x, start_a, half_a, axes_a, steps_a)
        x_second = op.Slice(x, half_a, end_a, axes_a, steps_a)
        neg_sec = op.Neg(x_second)
        rotated = op.Concat(neg_sec, x_first, axis=-1)
        rot_sin = op.Mul(rotated, sin_us)
        return op.Add(x_cos, rot_sin)

    def repl(
        op: pattern.RewriterContext,
        x: ir.Value,
        cos_us: ir.Value,
        sin_us: ir.Value,
        start_a: ir.Value,
        half_a: ir.Value,
        end_a: ir.Value,
        axes_a: ir.Value,
        steps_a: ir.Value,
    ):
        cos = _unwrap_unsqueeze(cos_us) or cos_us
        sin = _unwrap_unsqueeze(sin_us) or sin_us
        return op.RotaryEmbedding(x, cos, sin)

    return pattern.RewriteRule(pat, repl)


def _halve_cache_input_vis(model: onnx.ModelProto) -> None:
    """
    Halve the last axis of the graph-input VIs consumed as RotaryEmbedding cos/sin caches.

    The torch rotate_half convention traces cos/sin at full `head_dim`,
    so the subgraph's cos/sin graph inputs are declared `(1, L, head_dim)`.
    The fused opset-23 `RotaryEmbedding` instead takes `head_dim//2` caches for both cos/sin caches
    so the declared width must follow: `(1, L, head_dim)` -> `(1, L, head_dim//2)`.
    Returns the number of VIs rewritten.
    """
    cache_names = set()
    for node in model.graph.node:
        if node.op_type == "RotaryEmbedding" and len(node.input) >= 3:
            cache_names.update(node.input[1:3])

    for vi in model.graph.input:
        if vi.name not in cache_names:
            continue
        last_axis = len(vi.type.tensor_type.shape.dim) - 1
        width = get_vi_axis(vi, last_axis)
        if isinstance(width, int) and width % 2 == 0:
            set_vi_axis(vi, last_axis, width // 2)
        else:
            logger.warning(f"fuse_rope: cannot halve cache input {vi.name!r} last axis ({width!r}); left as-is.")


def fuse_rope(model: onnx.ModelProto) -> Tuple[onnx.ModelProto, int]:
    """
    Fuse rotate_half RoPE subgraphs into opset-23 RotaryEmbedding nodes.

    Each explicit subgraph — `x*cos + rotate_half(x)*sin` — is replaced by a
    single standard `RotaryEmbedding(X, cos_cache, sin_cache)` node (opset 23, default ONNX domain).
    The rule fires independently for Q and K, so two fusions are expected per attention layer.
    The Unsqueeze nodes that PyTorch inserts to broadcast cos/sin over heads are looked through
    so the replacement receives the raw `(N, L, E)` caches; the Unsqueeze nodes become dead code.

    Args:
        model: input ONNX model.
    Returns:
        `(patched_model, num_fusions)`
    """
    rule_set = pattern.RewriteRuleSet([_rope_rule()])
    model_ir = ir.from_proto(model)
    new_model_ir = rewriter.rewrite(model_ir, pattern_rewrite_rules=rule_set)
    new_model: onnx.ModelProto = ir.to_proto(new_model_ir)

    n_fused = sum(1 for n in new_model.graph.node if n.op_type == "RotaryEmbedding")
    if n_fused:
        update_opset(new_model, domain="", version=23)
        _halve_cache_input_vis(new_model)
        logger.debug(f"fuse_rope: fused {n_fused} RotaryEmbedding node(s)")
    return new_model, n_fused


__all__ = ["fuse_rope"]
