from typing import Dict, List, Optional

import onnx
import transformers

from hf2mobile.constant import _NORM_OPS
from hf2mobile.utils.onnx_helper import get_bwd_dict, get_fwd_dict

# TODO: eager mode won't create Attention ONNX use fuser under submodules/attention.py
# TODO: use onnx_ir instead of using bwd_dict, fwd_dict


class AttentionIdentifier:
    """Structural identification of a traced Attention FunctionProto.

    e.g. (Qwen3 generation case)
      {args_0} ─ MatMul ─ Reshape(4d) ─ RMSNorm ─ Transpose ─ RotaryEmbedding ─────┐
      {args_0} ─ MatMul ─ Reshape(4d) ─ RMSNorm ─ Transpose ─ RotaryEmbedding ── Concat ── {prev_K}
      {args_0} ─ MatMul ─ Reshape(4d) ─ Transpose ────────────────────────────── Concat ── {prev_V}
                                                                                   |
                                                                             Attention(Q,K,V)
                                                                 Transpose ─ Reshape(3d) ─ MatMul -> {OUT}
    """

    _PASSTHROUGH_OPS = ("Cast", "Identity", "Add", "Mul")

    def __init__(self, func: onnx.FunctionProto, hf_config: transformers.PreTrainedConfig):
        self.hf_config = hf_config

        self.func = func
        self.fwd_dict: Dict[str, List[onnx.NodeProto]] = get_fwd_dict(func.node)
        self.bwd_dict: Dict[str, onnx.NodeProto] = get_bwd_dict(func.node)
        self.edge2const: Dict[str, onnx.TensorProto] = {
            n.output[0]: attr.t
            for n in func.node
            if n.op_type == "Constant" and n.output
            for attr in n.attribute
            if attr.name == "value"
        }

        _attns = [n for n in func.node if n.op_type == "Attention"]
        if len(_attns) != 1:
            raise ValueError(f"`{func.name}` holds {len(_attns)} Attention node(s); expected exactly 1.")
        self.attention: onnx.NodeProto = _attns[0]

        # NOTE: the following _walk_bwd and _walk_fwd calls are assuming certain graph topologies in advacne
        # [bwd]
        #   Q, K, V MatMul weights are not merged
        #   RotaryEmbedding must have been fused in advance
        #   (OPTIONAL) RMSNormalization, SimplifiedLayerNormalization, LayerNormalization must have been fused in advance
        # [fwd]
        #   {O -> Trnaspose -> Reshape} must be a pathological tree
        self._q = self._walk_bwd(self.attention.input[0])
        self._k = self._walk_bwd(self.attention.input[1])
        self._v = self._walk_bwd(self.attention.input[2])
        self._o = self._walk_fwd(self.attention.output[0])
        for tag, path in (("Q", self._q), ("K", self._k), ("V", self._v)):
            if "matmul" not in path or "reshape" not in path:
                raise ValueError(f"`{func.name}` {tag} path is missing its projection MatMul/Reshape.")
        if "matmul" not in self._o or "reshape" not in self._o:
            raise ValueError(f"`{func.name}` output path is missing its o_proj MatMul/Reshape.")

    @classmethod
    def is_attention_func(cls, func: onnx.FunctionProto) -> bool:
        return sum(1 for n in func.node if n.op_type == "Attention") == 1

    # ================ fwd bwd inspector ================ #
    def _walk_bwd(self, edge: str) -> Dict[str, onnx.NodeProto]:
        """
        Using `slef.bwd_dict` backtrace Attention block nodes on Q, K or V branch.
        Return dictionary containing references to ...
            `matmul` (Q, K or V)
            `concat` (past_KV + cur_KV (IF EXIST))
            `rope`   (assumes RotaryEmbedding has already been fused)
            `norm`   (assums RMSNormalization, SimplifiedLayerNormalization, LayerNormalization are fused already (IF EXIST))
            `transpose`, `reshape` (3d -> 4d head split)
        NodeProtos
        """
        parents: Dict[str, onnx.NodeProto] = {}
        while True:
            node = self.bwd_dict.get(edge)
            if node is None:
                break  # NOTE: graph input reached (end of the loop)

            if node.op_type in ("MatMul", "Gemm"):
                parents["matmul"] = node
                break  # NOTE: MM reached (end of the loop)
            elif node.op_type == "Concat":
                parents["concat"] = node
                edge = next((_ for _ in node.input if _ in self.bwd_dict), None)
                if edge is None:
                    break  # NOTE: concat reached but does not have parent node
            elif node.op_type == "RotaryEmbedding":
                parents["rope"] = node
                edge = node.input[0]
            elif node.op_type in _NORM_OPS:
                parents["norm"] = node
                edge = node.input[0]
            elif node.op_type == "Reshape":
                parents.setdefault("reshape", node)
                edge = node.input[0]
            elif node.op_type == "Transpose":
                parents["transpose"] = node
                edge = node.input[0]
            elif node.op_type in self._PASSTHROUGH_OPS:
                edge = next((_ for _ in node.input if _ in self.bwd_dict), node.input[0])
            else:
                break  # NOTE: no more case to cover exit the loop
        return parents

    def _walk_fwd(self, edge: str) -> Dict[str, onnx.NodeProto]:
        """
        Using `slef.fwd_dict` forwardtrace Attention block nodes from sftmx((QK.T)/sqrt(d))V.output[0]
        Return dictionary containing references to ...
            `matmul` (O)
            `transpose`, `reshape` (4d -> 3d head merge)
        NodeProtos
        """
        children: Dict[str, onnx.NodeProto] = {}
        while True:
            nexts = self.fwd_dict.get(edge, [])
            if len(nexts) != 1:
                break  # NOTE: graph output reached (end of the loop)
            assert (
                len(nexts) == 1
            ), f"`{edge}` directs to multiple NodeProtos `{[n.name for n in nexts]}`. Please pass an edge name that does not branch out (pathological)."
            node = nexts[0]
            if node.op_type in ("MatMul", "Gemm"):
                children["matmul"] = node
                break
            elif node.op_type == "Reshape":
                children["reshape"] = node
            elif node.op_type == "Transpose":
                children["transpose"] = node
            elif node.op_type not in _PASSTHROUGH_OPS:
                break
            edge = node.output[0]  # NOTE: always follow output[0]
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
    def queryMM(self) -> onnx.NodeProto:
        return self._q["matmul"]

    @property
    def keyMM(self) -> onnx.NodeProto:
        return self._k["matmul"]

    @property
    def valueMM(self) -> onnx.NodeProto:
        return self._v["matmul"]

    @property
    def query4dReshape(self) -> onnx.NodeProto:
        return self._q["reshape"]

    @property
    def key4dReshape(self) -> onnx.NodeProto:
        return self._k["reshape"]

    @property
    def value4dReshape(self) -> onnx.NodeProto:
        return self._v["reshape"]

    @property
    def queryNorm(self) -> Optional[onnx.NodeProto]:
        return self._q.get("norm")

    @property
    def keyNorm(self) -> Optional[onnx.NodeProto]:
        return self._k.get("norm")

    @property
    def valueNorm(self) -> Optional[onnx.NodeProto]:
        return self._v.get("norm")

    @property
    def queryRope(self) -> Optional[onnx.NodeProto]:
        return self._q.get("rope")

    @property
    def keyRope(self) -> Optional[onnx.NodeProto]:
        return self._k.get("rope")

    @property
    def keyConcat(self) -> Optional[onnx.NodeProto]:
        return self._k.get("concat")

    @property
    def valueConcat(self) -> Optional[onnx.NodeProto]:
        return self._v.get("concat")

    @property
    def out3dReshape(self) -> onnx.NodeProto:
        return self._o["reshape"]

    @property
    def outMM(self) -> onnx.NodeProto:
        return self._o["matmul"]

    # ================ derived attention hyper-params ================ #
    def _get_shape_tp_from_reshape(self, reshape_node: onnx.NodeProto) -> onnx.TensorProto | None:
        assert (
            reshape_node.op_type == "Reshape"
        ), f"`{reshape_node.op_type}` is not an allowed op_type for `_get_shape_tp_from_reshape`"
        return self.edge2const.get(reshape_node.input[1], None)

    @property
    def head_dim(self) -> int:
        """Last element of the Q head-split Reshape, e.g. [1, L, H, head_dim] -> head_dim"""
        shape_tp = self._get_shape_tp_from_reshape(self.query4dReshape)
        head_dim = int(onnx.numpy_helper.to_array(shape_tp).ravel()[-1])
        if self.hf_config is not None:
            config_head_dim = getattr(
                self.hf_config, "head_dim", self.hf_config.hidden_size // self.hf_config.num_attention_heads
            )
            head_dim = head_dim if head_dim <= 0 else config_head_dim
            assert head_dim == config_head_dim
        return head_dim

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

    def _mm_out_features(self, mm: onnx.NodeProto) -> int:
        """Out-features of a projection MatMul/Gemm: weight is `[in, out]` (`[out, in]` if transB)."""
        trans_b = next((a.i for a in mm.attribute if a.name == "transB"), 0) if mm.op_type == "Gemm" else 0
        for edge_name in mm.input:
            tp = self.edge2const.get(edge_name, None)
            if tp is not None and len(tp.dims) == 2:
                return int(tp.dims[0] if trans_b else tp.dims[1])
        raise ValueError(f"`{self.func.name}` {mm.name!r} has non 2D Constant weight input.")


__all__ = ["AttentionIdentifier"]
