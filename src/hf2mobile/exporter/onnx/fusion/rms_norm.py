"""
RMSNorm fusion (fuse_rms_norm)
──────────────────────────────
Pattern (Qwen3RMSNorm expansion from torch.onnx.export):

    Cast(x, to=FLOAT)                      # promote to f32
    → Pow(x_f32, 2)
    → ReduceMean(sq, axes=[-1], keepdims=1)
    → Add(mean, epsilon)
    → Sqrt
    → Reciprocal                           # = rsqrt
    → Mul(x_f32, rsqrt)
    → Cast(normed, to=<original dtype>)    # demote back
    → Mul(scale, normed_cast)              # affine scale

Replaced with:
    RMSNormalization(x, scale, epsilon=eps, stash_type=FLOAT, axis=-1)

Three rule variants cover fp16, bf16, and fp32 (no-cast) models.
"""

from typing import Tuple

import onnx
import onnx_ir as ir
from onnxscript import rewriter
from onnxscript.rewriter import pattern

from hf2mobile.utils.logger import logger
from hf2mobile.utils.onnx_helper import get_scalar, update_opset


def _cast_rule(cast_back_to: int) -> pattern.RewriteRule:
    """Rule for non-fp32 models: Cast-in → … → Cast-out → Mul(scale)."""

    def pat(
        op: pattern.OpsetPatternBuilder,
        x: pattern.Var,
        scale: pattern.Var,
        pow_exp: pattern.Var,
        axes: pattern.Var,
        epsilon: pattern.Var,
    ):
        x_f32 = op.Cast(x, to=onnx.TensorProto.FLOAT)
        sq = op.Pow(x_f32, pow_exp)
        mean_sq = op.ReduceMean(sq, axes)
        added = op.Add(mean_sq, epsilon)
        rms = op.Sqrt(added)
        rsqrt = op.Reciprocal(rms)
        normed = op.Mul(x_f32, rsqrt)
        normed_bc = op.Cast(normed, to=cast_back_to)
        return op.Mul(scale, normed_bc)

    def repl(
        op: pattern.RewriterContext,
        x: ir.Value,
        scale: ir.Value,
        pow_exp: ir.Value,
        axes: ir.Value,
        epsilon: ir.Value,
    ):
        eps = get_scalar(epsilon) or 1e-6
        return op.RMSNormalization(x, scale, epsilon=eps, stash_type=onnx.TensorProto.FLOAT, axis=-1)

    def cond(
        context: "MatchContext",
        x: ir.Value,
        scale: ir.Value,
        pow_exp: ir.Value,
        axes: ir.Value,
        epsilon: ir.Value,
    ):
        p = get_scalar(pow_exp)
        return p is not None and abs(p - 2.0) < 1e-6

    return pattern.RewriteRule(pat, repl, cond)


def _fp32_rule() -> pattern.RewriteRule:
    """Rule for fp32 models: no surrounding Cast nodes."""

    def pat(
        op: pattern.OpsetPatternBuilder,
        x: pattern.Var,
        scale: pattern.Var,
        pow_exp: pattern.Var,
        axes: pattern.Var,
        epsilon: pattern.Var,
    ):
        sq = op.Pow(x, pow_exp)
        mean_sq = op.ReduceMean(sq, axes)
        added = op.Add(mean_sq, epsilon)
        rms = op.Sqrt(added)
        rsqrt = op.Reciprocal(rms)
        normed = op.Mul(x, rsqrt)
        return op.Mul(scale, normed)

    def repl(
        op: pattern.RewriterContext,
        x: ir.Value,
        scale: ir.Value,
        pow_exp: ir.Value,
        axes: ir.Value,
        epsilon: ir.Value,
    ):
        eps = get_scalar(epsilon) or 1e-6
        return op.RMSNormalization(x, scale, epsilon=eps, stash_type=onnx.TensorProto.FLOAT, axis=-1)

    def cond(
        context: "MatchContext",
        x: ir.Value,
        scale: ir.Value,
        pow_exp: ir.Value,
        axes: ir.Value,
        epsilon: ir.Value,
    ):
        p = get_scalar(pow_exp)
        return p is not None and abs(p - 2.0) < 1e-6

    return pattern.RewriteRule(pat, repl, cond)


def _gemma_cast_rule(cast_back_to: int) -> pattern.RewriteRule:
    """
    Rule for Gemma-style norms:
    `(1 + weight)` scale is applied in fp32 BEFORE the cast back to original activation dtype
    unlike normal RMSNorm (e.g. torch.nn.functional.rms_norm).

    Cast(x, f32) → ...RMSNorm... → Mul(x_f32) → Mul(scale_f32) → Cast(back)
    """

    def pat(
        op: pattern.OpsetPatternBuilder,
        x: pattern.Var,
        scale: pattern.Var,
        pow_exp: pattern.Var,
        axes: pattern.Var,
        epsilon: pattern.Var,
    ):
        x_f32 = op.Cast(x, to=onnx.TensorProto.FLOAT)
        sq = op.Pow(x_f32, pow_exp)
        mean_sq = op.ReduceMean(sq, axes)
        added = op.Add(mean_sq, epsilon)
        rms = op.Sqrt(added)
        rsqrt = op.Reciprocal(rms)
        normed = op.Mul(x_f32, rsqrt)
        scaled = op.Mul(normed, scale)
        return op.Cast(scaled, to=cast_back_to)

    def repl(
        op: pattern.RewriterContext,
        x: ir.Value,
        scale: ir.Value,
        pow_exp: ir.Value,
        axes: ir.Value,
        epsilon: ir.Value,
    ):
        eps = get_scalar(epsilon) or 1e-6
        # keep the `(1 + weight)` scale in fp32 (HF applies it BEFORE the cast-back). opset-23
        #   RMSNormalization ties X and scale to the same dtype (shape inference fails otherwise),
        #   so upcast X to fp32 too and cast the fp32 result back to the activation dtype.
        x_f32 = op.Cast(x, to=onnx.TensorProto.FLOAT)
        y_f32 = op.RMSNormalization(x_f32, scale, epsilon=eps, stash_type=onnx.TensorProto.FLOAT, axis=-1)
        return op.Cast(y_f32, to=cast_back_to)

    def cond(
        context: "MatchContext",
        x: ir.Value,
        scale: ir.Value,
        pow_exp: ir.Value,
        axes: ir.Value,
        epsilon: ir.Value,
    ):
        p = get_scalar(pow_exp)
        return p is not None and abs(p - 2.0) < 1e-6

    return pattern.RewriteRule(pat, repl, cond)


_RMS_RULE_SET = pattern.RewriteRuleSet(
    [
        _cast_rule(onnx.TensorProto.FLOAT16),
        _cast_rule(onnx.TensorProto.BFLOAT16),
        _gemma_cast_rule(onnx.TensorProto.FLOAT16),
        _gemma_cast_rule(onnx.TensorProto.BFLOAT16),
        _fp32_rule(),
    ]
)


def fuse_rms_norm(model: onnx.ModelProto) -> Tuple[onnx.ModelProto, int]:
    """
    Fuse expanded RMSNorm subgraphs into opset-23 RMSNormalization nodes.

    Args:
        model: input ONNX model (modified in-place via onnx_ir round-trip).

    Returns:
        `(patched_model, num_fusions)`
    """
    model_ir = ir.from_proto(model)
    new_model_ir = rewriter.rewrite(model_ir, pattern_rewrite_rules=_RMS_RULE_SET)
    new_model: onnx.ModelProto = ir.to_proto(new_model_ir)

    n_fused = sum(1 for n in new_model.graph.node if n.op_type == "RMSNormalization")
    if n_fused > 0:
        update_opset(new_model, domain="", version=23)
        logger.debug(f"fuse_rms_norm: fused {n_fused} RMSNormalization node(s)")
    return new_model, n_fused


__all__ = ["fuse_rms_norm"]
