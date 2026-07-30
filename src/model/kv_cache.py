import torch


class KVCache:
    """Per-request key/value cache for fast autoregressive decoding.

    Stores the keys and values computed for every past token, one entry per
    layer, so that generating a new token only requires computing K/V for that
    single token instead of re-running the whole sequence.

    This is REQUEST state, not model state: one KVCache is created per
    generation call and threaded through the forward pass, never stored on a
    module. Two concurrent requests hold two separate caches and cannot corrupt
    each other.

    Cache tensors are stored UNEXPANDED (n_kv_heads, not n_heads). The GQA
    repeat_interleave happens after the cache is read, inside attention, so the
    cache keeps the GQA memory saving: for V3, 3 KV heads stored rather than 12.
    Keys are stored with RoPE already applied; the expansion and the attention
    are what happen downstream.

    Attributes:
        k_cache: list of length n_layers. Entry i is layer i's accumulated keys,
                 shape [B, tokens_so_far, n_kv_heads, d_k], or None before the
                 first update.
        v_cache: same, for values.
        length:  number of tokens currently cached. NOT bumped here; gpt.forward
                 advances it once per step (by seq_len) after the layer loop, so
                 it moves once per token rather than once per layer.
    """

    def __init__(self, n_layers):
        self.k_cache = [None] * n_layers
        self.v_cache = [None] * n_layers
        self.length = 0

    def update(self, layer_idx, k_new, v_new):
        """Append this step's K/V for one layer and return the full accumulated K/V.

        On the first call for a layer (prefill), stores k_new/v_new as-is. On
        later calls (decode), concatenates along the sequence dimension (dim=1).

        Args:
            layer_idx: which layer's cache to update.
            k_new: keys for the new token(s), [B, seq_new, n_kv_heads, d_k],
                   RoPE already applied, NOT GQA-expanded.
            v_new: values for the new token(s), same shape.

        Returns:
            (k, v) for this layer over ALL tokens so far (past + new), each
            [B, tokens_so_far, n_kv_heads, d_k]. This is what attention attends
            against, before the GQA expansion.
        """
        if self.k_cache[layer_idx] is None:
            self.k_cache[layer_idx] = k_new
            self.v_cache[layer_idx] = v_new
        else:
            self.k_cache[layer_idx] = torch.cat([self.k_cache[layer_idx], k_new], dim=1)
            self.v_cache[layer_idx] = torch.cat([self.v_cache[layer_idx], v_new], dim=1)

        return self.k_cache[layer_idx], self.v_cache[layer_idx]