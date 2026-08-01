from typing import Dict, List, Optional

import onnx_ir as ir
import transformers

from hf2mobile.constant import _NORM_OPS
from hf2mobile.utils.onnx_helper import get_const_tensor

# TODO: eager mode won't create Attention ONNX use fuser under submodules/attention.py


class AttentionIdentifier:
    """
    Structural identification of a traced Attention `ir.Function`.

    e.g. (Qwen3 generation case)
      {args_0} ─ MatMul ─ Reshape(4d) ─ RMSNorm ─ Transpose ─ RotaryEmbedding ─────┐
      {args_0} ─ MatMul ─ Reshape(4d) ─ RMSNorm ─ Transpose ─ RotaryEmbedding ── Concat ── {prev_K}
      {args_0} ─ MatMul ─ Reshape(4d) ─ Transpose ────────────────────────────── Concat ── {prev_V}
                                                                                   |
                                                                             Attention(Q,K,V)
                                                                 Transpose ─ Reshape(3d) ─ MatMul -> {OUT}
    """

    _PASSTHROUGH_OPS = ("Cast", "Identity", "Add", "Mul")

    def __init__(self, func: ir.Function, hf_config: transformers.PreTrainedConfig):
        self.hf_config = hf_config
        self.func = func

        _attns = [n for n in func if n.op_type == "Attention"]
        if len(_attns) != 1:
            raise ValueError(f"`{func.name}` holds {len(_attns)} Attention node(s); expected exactly 1.")
        self.attention: ir.Node = _attns[0]

        # NOTE: the following _walk_bwd and _walk_fwd calls are assuming certain graph topologies in advacne
        # [bwd]
        #   Q, K, V MatMul weights are not merged
        #   RotaryEmbedding must have been fused in advance
        #   (OPTIONAL) RMSNormalization, SimplifiedLayerNormalization, LayerNormalization must have been fused in advance
        # [fwd]
        #   {O -> Trnaspose -> Reshape} must be a pathological tree
        self._q: Dict[str, ir.Node] = self._walk_bwd(self.attention.inputs[0])
        self._k: Dict[str, ir.Node] = self._walk_bwd(self.attention.inputs[1])
        self._v: Dict[str, ir.Node] = self._walk_bwd(self.attention.inputs[2])
        self._o: Dict[str, ir.Node] = self._walk_fwd(self.attention.outputs[0])
        for name, attn_info in (("_q", self._q), ("_k", self._k), ("_v", self._v)):
            if "matmul" not in attn_info or "reshape" not in attn_info:
                raise ValueError(
                    f"[FuncProto.name={func.name}] `self._{name}` attn_info is missing its projection MatMul/Reshape."
                )
        if "matmul" not in self._o or "reshape" not in self._o:
            raise ValueError(
                f"[FuncProto.name={func.name}] `{func.name}` `self._o` attn_info is missing its o_proj MatMul/Reshape."
            )

    @classmethod
    def is_attention_func(cls, func: ir.Function) -> bool:
        return sum(1 for n in func if n.op_type == "Attention") == 1

    # ================ fwd bwd inspector ================ #
    @staticmethod
    def _parent_node(value: ir.Value | None) -> ir.Node | None:
        """parent node from edge, None at a graph input / dangling edge."""
        return value.producer() if value is not None else None

    def _walk_bwd(self, value: ir.Value | None) -> Dict[str, ir.Node]:
        """
        Bwdtrace Attention QKV branches from the attention.input[0~2] edges.
        Return dictionary containing references to ...
            `matmul` (Q, K or V)
            `concat` (past_KV + cur_KV (IF EXIST))
            `rope`   (assumes RotaryEmbedding has already been fused)
            `norm`   (assums RMSNormalization, SimplifiedLayerNormalization, LayerNormalization are fused already (IF EXIST))
            `transpose`, `reshape` (3d -> 4d head split)
        """
        parents: Dict[str, ir.Node] = {}
        while True:
            node = self._parent_node(value)
            if node is None:
                break  # NOTE: graph input reached (end of the loop)

            if node.op_type in ("MatMul", "Gemm"):
                parents["matmul"] = node
                break  # NOTE: MM reached (end of the loop)
            elif node.op_type == "Concat":
                parents["concat"] = node
                value = next((v for v in node.inputs if self._parent_node(v) is not None), None)
                if value is None:
                    break  # NOTE: concat reached but has no parents (end loop)
            elif node.op_type == "RotaryEmbedding":
                parents["rope"] = node
                value = node.inputs[0]
            elif node.op_type in _NORM_OPS:
                parents["norm"] = node
                value = node.inputs[0]
            elif node.op_type == "Reshape":
                parents.setdefault("reshape", node)
                value = node.inputs[0]
            elif node.op_type == "Transpose":
                parents["transpose"] = node
                value = node.inputs[0]
            elif node.op_type in self._PASSTHROUGH_OPS:
                value = next((v for v in node.inputs if self._parent_node(v) is not None), node.inputs[0])
            else:
                break  # NOTE: no more case to cover exit the loop
        return parents

    def _walk_fwd(self, value: ir.Value | None) -> Dict[str, ir.Node]:
        """
        Fwdtrace Attention block nodes from sftmx((QK.T)/sqrt(d))V.outputs[0]
        Return dictionary containing references to ...
            `matmul` (O)
            `transpose`, `reshape` (4d -> 3d head merge)
        """
        children: Dict[str, ir.Node] = {}
        while True:
            if value is None:
                break
            nexts = [use.node for use in value.uses()]
            if len(nexts) != 1:
                break  # NOTE: graph output (or a branch) reached — end of the loop
            node = nexts[0]
            if node.op_type in ("MatMul", "Gemm"):
                children["matmul"] = node
                break
            elif node.op_type == "Reshape":
                children["reshape"] = node
            elif node.op_type == "Transpose":
                children["transpose"] = node
            elif node.op_type not in self._PASSTHROUGH_OPS:
                break
            value = node.outputs[0]  # NOTE: always follow outputs[0]
        return children

    # ================ shape references ================ #
    @property
    def query4dShape(self) -> List[int]:
        return [1, -1, self.num_heads, self.head_dim]

    @property
    def key4dShape(self) -> List[int]:
        return [1, -1, self.kv_num_heads, self.head_dim]

    @property
    def value4dShape(self) -> List[int]:
        return [1, -1, self.kv_num_heads, self.head_dim]

    @property
    def out3dShape(self) -> List[int]:
        return [1, -1, self.num_heads * self.head_dim]

    # ================ node references ================ #
    @property
    def queryMM(self) -> ir.Node:
        return self._q["matmul"]

    @property
    def keyMM(self) -> ir.Node:
        return self._k["matmul"]

    @property
    def valueMM(self) -> ir.Node:
        return self._v["matmul"]

    @property
    def query4dReshape(self) -> ir.Node:
        return self._q["reshape"]

    @property
    def key4dReshape(self) -> ir.Node:
        return self._k["reshape"]

    @property
    def value4dReshape(self) -> ir.Node:
        return self._v["reshape"]

    @property
    def queryNorm(self) -> Optional[ir.Node]:
        return self._q.get("norm")

    @property
    def keyNorm(self) -> Optional[ir.Node]:
        return self._k.get("norm")

    @property
    def valueNorm(self) -> Optional[ir.Node]:
        return self._v.get("norm")

    @property
    def queryRope(self) -> Optional[ir.Node]:
        return self._q.get("rope")

    @property
    def keyRope(self) -> Optional[ir.Node]:
        return self._k.get("rope")

    @property
    def keyConcat(self) -> Optional[ir.Node]:
        return self._k.get("concat")

    @property
    def valueConcat(self) -> Optional[ir.Node]:
        return self._v.get("concat")

    @property
    def out3dReshape(self) -> ir.Node:
        return self._o["reshape"]

    @property
    def outMM(self) -> ir.Node:
        return self._o["matmul"]

    # ================ derived attention hyper-params ================ #
    def _get_shape_tensor_from_reshape(self, reshape_node: ir.Node) -> ir.TensorProtocol | None:
        assert (
            reshape_node.op_type == "Reshape"
        ), f"`{reshape_node.op_type}` is not an allowed op_type for `_get_shape_tensor_from_reshape`"
        return get_const_tensor(reshape_node.inputs[1])

    @property
    def head_dim(self) -> int:
        """Last element of the Q head-split Reshape, e.g. [1, L, H, head_dim] -> head_dim"""
        shape_tensor = self._get_shape_tensor_from_reshape(self.query4dReshape)
        head_dim = int(shape_tensor.numpy().ravel()[-1])
        if self.hf_config is not None:
            config_head_dim = getattr(
                self.hf_config, "head_dim", self.hf_config.hidden_size // self.hf_config.num_attention_heads
            )
            head_dim = head_dim if head_dim <= 0 else config_head_dim  # NOTE: in case -1 under ONNX reshape
            assert head_dim == config_head_dim
        return head_dim

    def _mm_out_features(self, mm: ir.Node) -> int:
        """Out-features of a projection MatMul/Gemm: weight is `[in, out]` (`[out, in]` if transB)."""
        trans_b = int(mm.attributes.get_int("transB", 0)) if mm.op_type == "Gemm" else 0
        for value in mm.inputs:
            tensor = get_const_tensor(value)
            if tensor is not None and len(tensor.shape) == 2:
                return int(tensor.shape[0] if trans_b else tensor.shape[1])
        raise ValueError(f"`{self.func.name}` {mm.name!r} has non 2D Constant weight input.")

    @property
    def num_heads(self) -> int:
        num_heads = self._mm_out_features(self.queryMM) // self.head_dim
        if self.hf_config is not None:
            assert num_heads == self.hf_config.num_attention_heads
        return num_heads

    @property
    def kv_num_heads(self) -> int:
        num_heads = self._mm_out_features(self.keyMM) // self.head_dim
        if self.hf_config is not None:
            assert num_heads == self.hf_config.num_key_value_heads
        return num_heads


__all__ = ["AttentionIdentifier"]
