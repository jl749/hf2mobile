# TODO: class that takes an Attention FuncProto as an input (called AttentionIdentifier)
#   it should be able to locate Q, K, V matmul and Q, K, V reshape node and return reference to NodeProto using property (e.g. queryMM, keyMM, valueMM, query4dReshape, key4dReshape, ..., o3dReshape)
#   initializer has bwd_dict that maps edge name to parent nodeproto which later can be used to locate QKV MMs and QKVOReshape
#   starting from attention node we can backtrace because Attention node inputs are in Q,K,V order


# TODO: fuse_group_query_attention(model: onnx.ModelProto) call which constructs ORT GQA plugin within the graph based on Attention topology
#   starting model example: /home/simp/GIT/hf-quantizer/2026-07-14_01-23-48__ORT__Qwen-Qwen3-0.6B/case2.onnx
#   please fuse RotaryEmbedding nodes into GQA, (Qwen3RotaryEmbedding____model__rotary_emb____case2 subgraph is no longer needed as we will be passing seqlens_k and total_sequence_length directly)
#   use Gather.input[0] on both cos,sine embed nodes in order to set cos_cache and sin_cache input on the GQA node
