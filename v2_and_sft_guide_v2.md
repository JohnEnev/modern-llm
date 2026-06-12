# V2 Modules + SFT Guide
*Offline reference: questions → answers → try it (skeleton with hints) → full solution → interview Q&A*

---

## Status

**Done:**
- ✅ Muon optimizer (`src/optim/muon.py`)

**V2 remaining (this doc):**
- ⬜ EMA — `train.py`
- ⬜ QK-Norm + learnable scale — `src/model/attention.py`
- ⬜ GQA — `src/model/attention.py`
- ⬜ Differential Attention — `src/model/attention.py`
- ⬜ mHC Hyper-Connections — `src/model/hyper_connection.py`
- ⬜ train.py integration

**SFT (this doc):**
- ⬜ SFT dataset class — `src/data/sft_dataset.py`
- ⬜ SFT training script — `sft.py`

---

## Part 1: EMA (Exponential Moving Average)

### Concept

During training, weights fluctuate step to step. EMA keeps a smoothed shadow copy:
```
ema_weights = decay * ema_weights + (1 - decay) * current_weights
```
After training, the EMA model has seen a weighted average of all recent checkpoints — like taking the center of mass of where the model has been. Gives 0.02-0.05 lower val loss for free.

**decay=0.9995** means the EMA weights weight the last ~2700 steps roughly equally (half-life ≈ 1386 steps).

**Key rule:** Use EMA for eval/inference. Keep training the original model normally.

### Questions

**Q1:** If decay=0.9995 and you train for 20,000 steps, roughly how many recent steps does the EMA weight equally?

**Q2:** Why use EMA for inference but NOT continue training from it?

**Q3:** Theoretically, why should the average of past weights be better than the final weights?

### Answers

**A1:** Half-life = log(0.5)/log(0.9995) ≈ 1386 steps. The EMA roughly averages over the last ~2700 steps (two half-lives). Recent steps count more, older steps decay exponentially.

**A2:** EMA weights lag behind the training trajectory. If you trained FROM EMA, you'd lose the optimizer's momentum state (Adam's m and v buffers) which corresponds to the TRAINING model, not EMA. Resuming from EMA discards all that learned optimization history.

**A3:** Adam finds weights that minimize training loss, but the loss surface is non-convex and noisy. The optimizer oscillates around a basin without perfectly settling. Averaging reduces the variance of this oscillation — you're averaging many points near the true minimum, which by Jensen's inequality lands closer to the minimum than any individual noisy point.

### Try It

```python
# Add to train.py

class EMA:
    """Exponential Moving Average of model weights.
    Use ema.model for evaluation, not for training.
    """
    
    def __init__(self, model: torch.nn.Module, decay: float = 0.9995):
        self.decay = decay
        
        # TODO:
        # 1. Create a deep copy of model for EMA weights
        #    Hint: import copy; self.model = copy.deepcopy(model)
        #
        # 2. Set EMA model to eval mode — it's never trained directly
        #    Hint: self.model.eval()
        #
        # 3. Freeze all EMA parameters — updated manually, not by gradients
        #    Hint: for param in self.model.parameters(): param.requires_grad_(False)
        pass
    
    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        """Update EMA weights after each optimizer step."""
        # TODO:
        # For each pair of (ema_param, train_param):
        #   ema_param = decay * ema_param + (1 - decay) * train_param
        #   Hint: ema_param.data.mul_(self.decay).add_(train_param.data, alpha=1-self.decay)
        pass
    
    def state_dict(self):
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
```

### Full Solution

```python
class EMA:
    """Exponential Moving Average of model weights.
    Use ema.model for evaluation, not for training.
    """
    
    def __init__(self, model: torch.nn.Module, decay: float = 0.9995):
        self.decay = decay
        import copy
        self.model = copy.deepcopy(model)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
    
    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        """Update EMA weights after each optimizer step."""
        for ema_param, train_param in zip(self.model.parameters(), model.parameters()):
            ema_param.data.mul_(self.decay).add_(train_param.data, alpha=1 - self.decay)
    
    def state_dict(self):
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
```

**Integration in train.py:**

```python
# After creating model, before training loop:
ema = EMA(model, decay=config.ema_decay)

# Inside training loop, after optimizer.step():
ema.update(model)

# In eval block — use EMA model, not training model:
if step > 0 and step % config.eval_interval == 0:
    val_loss = evaluate(ema.model, val_dataset, config, device, dtype)
    for prompt in EVAL_PROMPTS:
        sample = generate_sample(ema.model, enc, device, prompt=prompt)
    model.train()  # EMA stays in eval, only training model switches

# In checkpoint:
checkpoint["ema_weights"] = ema.state_dict()
```

**Add to TrainConfig:**
```python
use_ema: bool = True
ema_decay: float = 0.9995
```

### Interview Q&A

**"Why not just use the last checkpoint?"**
The last checkpoint is wherever the optimizer landed at step N — sensitive to the noise of that single step. EMA is the average of where the optimizer has been for ~2700 steps. Much more stable.

**"Does EMA help with SFT too?"**
Yes, but less so. SFT is short (1-3 epochs) and you WANT the model to change behavior quickly. Lower decay (0.999) or skip entirely for short SFT runs.

---

## Part 2: QK-Norm + Learnable Scale

### Concept

Q and K vectors can grow to arbitrary magnitude during training. Large dot products make softmax output extremely sharp (attending to one position) or collapse (all equal). QK-Norm normalizes Q and K to unit length, then scales by a learnable per-head parameter — separating direction (what to attend) from magnitude (how sharply).

**Standard:** `scores = Q @ K.T / sqrt(d_k)`
**QK-Norm:** `scores = F.normalize(Q) * scale @ F.normalize(K).T`

### Questions

**Q1:** After QK-Norm, Q and K are unit vectors. What is the range of their dot products? What does this imply for softmax temperature?

**Q2:** Why a learnable scale rather than fixed sqrt(d_k)?

**Q3:** What value should scale be initialized to for a smooth transition?

**Q4:** What happens to gradient magnitude through attention after QK-Norm?

### Answers

**A1:** Dot product of two unit vectors = cos(θ) ∈ [-1, 1]. Attention logits are ALWAYS in [-1, 1] regardless of training duration. Temperature is now entirely controlled by the learnable scale.

**A2:** Different heads specialize differently — some use sharp local attention (syntax), others use diffuse broad attention (semantics). A fixed scale can't adapt to different optimal sharpness per head.

**A3:** Pre-norm, Q·K ≈ sqrt(d_k) for random init. Initialize scale = sqrt(d_k) = 8 for d_k=64. This way the transition doesn't shock the model if adding mid-training. For from-scratch training, 1.0 is fine too.

**A4:** QK-Norm projects gradients onto the tangent space of the unit sphere, capping large gradients from large Q/K vectors. It provides built-in gradient clipping for the attention computation independently of the training loop's grad clip.

### Try It

```python
# Modifications to MultiHeadAttention.__init__:

def __init__(self, d_model, n_heads, dropout=0.0, max_seq_len=2048,
             use_flash=True, use_qk_norm=True):
    super().__init__()
    # ... existing init ...
    
    # TODO: Add QK-Norm parameters
    self.use_qk_norm = use_qk_norm
    if use_qk_norm:
        # Learnable scale: one per head, initialized to sqrt(d_k)
        # Shape: [n_heads] — each head independently controls its attention sharpness
        # Hint: nn.Parameter(torch.ones(n_heads) * (self.d_k ** 0.5))
        self.qk_scale = ???

# Modifications to MultiHeadAttention.forward, after computing q, k, v
# and BEFORE applying RoPE:

    if self.use_qk_norm:
        # TODO:
        # 1. Normalize Q to unit length along the d_k dimension
        #    Hint: F.normalize(q, dim=-1)
        #    Shape: [batch, seq, n_heads, d_k]
        #
        # 2. Multiply Q by per-head scale
        #    Hint: self.qk_scale.view(1, 1, -1, 1) to broadcast correctly
        #
        # 3. Normalize K to unit length (no scale on K)
        q = ???
        k = ???
```

### Full Solution

```python
# In __init__:
self.use_qk_norm = use_qk_norm
if use_qk_norm:
    self.qk_scale = nn.Parameter(torch.ones(n_heads) * (self.d_k ** 0.5))

# In forward, after computing q, k, v, before RoPE:
if self.use_qk_norm:
    q = F.normalize(q, dim=-1) * self.qk_scale.view(1, 1, -1, 1)
    k = F.normalize(k, dim=-1)

# Also: when using Flash Attention with QK-Norm, pass scale=1.0
# since QK-Norm already handles the scaling:
if self.use_flash:
    attn_output = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=True,
        scale=1.0 if self.use_qk_norm else None,
    )
```

**Add to GPTConfig:**
```python
use_qk_norm: bool = True
```

### Interview Q&A

**"Is QK-Norm equivalent to a temperature parameter?"**
Partially. A global temperature scales all heads uniformly. QK-Norm with per-head scale is N_heads separate temperatures, each learned independently. Also, QK-Norm bounds the logit range to [-scale, scale] which temperature doesn't.

**"Does QK-Norm replace the 1/sqrt(d_k) scaling?"**
Yes — after QK-Norm, dot products are already in [-scale, scale]. Pass `scale=1.0` to Flash Attention to avoid double-scaling.

---

## Part 3: GQA (Grouped Query Attention)

### Concept

Standard MHA: 16 Q heads, 16 K heads, 16 V heads. During inference, K and V are cached for every previous token — the KV cache. With GQA, multiple Q heads share one K/V head:

```
Standard MHA:     16 Q, 16 K, 16 V
GQA (4 groups):   16 Q,  4 K,  4 V  → 4x smaller KV cache
```

At 350M, the win is modest. At 7B+ serving many users, KV cache becomes the memory bottleneck — GQA is essential.

### Questions

**Q1:** With d_model=1024, n_heads=16, n_kv_heads=4: what are the W_q, W_k, W_v shapes? How does this change parameter count?

**Q2:** Q has shape [batch, seq, 16, 64] but K has shape [batch, seq, 4, 64]. How do you compute attention?

**Q3:** At what model scale does GQA start to matter most?

**Q4:** LLaMA-2 7B uses 32 Q, 8 KV. LLaMA-2 70B uses 64 Q, 8 KV. Why does the ratio increase with model size?

### Answers

**A1:**
- W_q: [1024, 16×64] = [1024, 1024] — unchanged
- W_k: [1024, 4×64] = [1024, 256] — 4x smaller
- W_v: [1024, 4×64] = [1024, 256] — 4x smaller
- Savings: ~1.57M params. <0.5% of 350M — negligible. Win is at inference, not training.

**A2:** Expand (repeat) K/V heads to match Q heads. Each K/V head is repeated 4 times:
```python
k = k.repeat_interleave(n_rep, dim=2)  # 4 → 16 heads
```

**A3:** Matters at 3-7B+ in production serving. At 350M with 1024 context, KV cache is ~100MB — trivial. At 7B with 4096 context and 100 concurrent users: ~160GB — catastrophic without GQA.

**A4:** Larger models have more capacity per head (128-dim KV at 70B vs 64-dim at 7B). Richer representations can be shared more aggressively across Q heads without quality loss.

### Try It

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads=None, dropout=0.0,
                 max_seq_len=2048, use_flash=True, use_qk_norm=True):
        super().__init__()
        
        # TODO: GQA setup
        # 1. Store n_heads and n_kv_heads
        #    If n_kv_heads is None, default to n_heads (standard MHA)
        #
        # 2. Compute n_rep = n_heads // n_kv_heads
        #    This is how many Q heads share each K/V head
        #    Assert n_heads % n_kv_heads == 0
        #
        # 3. W_q: projects to n_heads * d_k
        #    W_k, W_v: project to n_kv_heads * d_k (smaller!)
        self.n_heads = n_heads
        self.n_kv_heads = ???
        self.n_rep = ???
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, n_heads * self.d_k,      bias=False)
        self.W_k = nn.Linear(d_model, ???,                      bias=False)
        self.W_v = nn.Linear(d_model, ???,                      bias=False)
        self.W_o = nn.Linear(d_model, d_model,                  bias=False)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        q = self.W_q(x).view(batch_size, seq_len, self.n_heads,    self.d_k)
        k = self.W_k(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k)
        v = self.W_v(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k)
        
        # QK-Norm (if enabled) ...
        # RoPE ...
        
        # TODO: GQA expansion
        # If n_rep > 1, expand K and V to match Q heads
        # Hint: k = k.repeat_interleave(self.n_rep, dim=2)
        if self.n_rep > 1:
            k = ???
            v = ???
        
        # Transpose and attention as usual...
```

### Full Solution

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads=None, dropout=0.0,
                 max_seq_len=2048, use_flash=True, use_qk_norm=True):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.n_rep = n_heads // self.n_kv_heads
        assert n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.d_k = d_model // n_heads
        self.d_model = d_model
        self.dropout = dropout
        self.use_flash = use_flash
        
        self.W_q = nn.Linear(d_model, n_heads * self.d_k,           bias=False)
        self.W_k = nn.Linear(d_model, self.n_kv_heads * self.d_k,  bias=False)
        self.W_v = nn.Linear(d_model, self.n_kv_heads * self.d_k,  bias=False)
        self.W_o = nn.Linear(d_model, d_model,                      bias=False)
        
        self.rope_cache = RoPECache(self.d_k, max_seq_len)
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.qk_scale = nn.Parameter(torch.ones(n_heads) * (self.d_k ** 0.5))
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        q = self.W_q(x).view(batch_size, seq_len, self.n_heads,    self.d_k)
        k = self.W_k(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k)
        v = self.W_v(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k)
        
        if self.use_qk_norm:
            q = F.normalize(q, dim=-1) * self.qk_scale.view(1, 1, -1, 1)
            k = F.normalize(k, dim=-1)
        
        freqs = self.rope_cache.get_freqs(seq_len)
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)
        
        # GQA: expand K/V to match Q heads
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        if self.use_flash:
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
                scale=1.0 if self.use_qk_norm else None,
            )
        else:
            scale = 1.0 if self.use_qk_norm else (self.d_k ** -0.5)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1
            )
            attn_scores = attn_scores + causal_mask
            attn_weights = F.softmax(attn_scores, dim=-1)
            if self.attn_dropout:
                attn_weights = self.attn_dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(attn_output)
```

**Add to GPTConfig:**
```python
n_kv_heads: int = 4
```

### Interview Q&A

**"Does GQA hurt quality much?"**
At 350M: ~0.02-0.03 higher loss. At 7B+ with enough data: essentially no degradation. LLaMA-2 70B uses GQA and is competitive with MHA models of similar size.

**"GQA vs MQA?"**
MQA = extreme case: one shared K/V for ALL Q heads. GQA generalizes with G groups. MQA is cheapest, GQA finds a better quality-efficiency tradeoff. MQA used in PaLM, GQA became standard with LLaMA-2+.

---

## Part 4: Differential Attention

### Concept

Standard attention computes ONE softmax map per head — it assigns non-trivial weight to irrelevant tokens (attention noise). **Differential Attention** computes TWO maps and subtracts:

```
Attention_diff = softmax(Q1 @ K1.T / scale) - λ * softmax(Q2 @ K2.T / scale)
```

Like a differential amplifier: common-mode noise (both maps attend similarly to irrelevant tokens) cancels; signal (tokens the maps disagree about) amplifies. **λ** is learnable, initialized small and layer-dependent.

**Parameter count unchanged:** split d_k into two halves (d_k/2 per pair), so total Q/K params = 2 × d_k/2 = d_k. Same as standard.

### Questions

**Q1:** Show total parameter count per head is the same for standard vs differential attention.

**Q2:** λ init = 0.8 - 0.6 * exp(-0.3 * layer_idx). At layer 0: λ≈0.2, at layer 23: λ≈0.74. Why smaller in early layers?

**Q3:** Why apply GroupNorm after the differential attention output?

**Q4:** Give a concrete example of when differential attention improves output quality.

### Answers

**A1:**
Standard: Q+K+V = d_model×d_k + d_model×d_k + d_model×d_k = 3d_model×d_k
Differential: (Q1+Q2)+(K1+K2)+V = 2×d_model×(d_k/2) + 2×d_model×(d_k/2) + d_model×d_k = 3d_model×d_k ✓

**A2:** Early layers learn low-level features (syntax, local patterns) where most tokens are somewhat relevant — aggressive cancellation would damage useful signal. Later layers learn high-level semantics where most context IS noise relative to the key few tokens — stronger cancellation is beneficial.

**A3:** After differential subtraction, the attention map can have any mean. If both maps are similar, output is near zero — causing vanishing gradients downstream. GroupNorm ensures consistent output scale regardless of cancellation strength.

**A4:** Long-document QA: "What is the capital of France?" with 500 irrelevant sentences. Standard attention must learn to suppress 499 sentences. Differential: both maps attend broadly, but only the relevant sentence causes a DIFFERENCE — noise cancels, Paris is more reliably predicted.

### Try It

```python
# src/model/attention.py — add DifferentialAttention class

import math

class DifferentialAttention(nn.Module):
    """Differential Attention: cancels attention noise via map subtraction.
    
    Reference: https://arxiv.org/abs/2410.05258
    """
    
    def __init__(self, d_model, n_heads, layer_idx=0, dropout=0.0,
                 max_seq_len=2048, use_flash=True):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.d_k_half = self.d_k // 2
        self.dropout = dropout
        
        # TODO: projections
        # W_q and W_k output FULL d_k per head (both Q1+Q2 and K1+K2 packed together)
        # W_v outputs full d_k per head as usual
        # Hint: nn.Linear(d_model, n_heads * self.d_k, bias=False) for each
        self.W_q = ???
        self.W_k = ???
        self.W_v = ???
        self.W_o = ???
        
        # TODO: λ parameters — two pairs of learnable vectors for computing λ
        # λ = exp(λ_q1 · λ_k1) - exp(λ_q2 · λ_k2) + λ_init
        # Each is a vector of size d_k_half, initialized with small random values
        # Hint: nn.Parameter(torch.randn(self.d_k_half) * 0.1)
        lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_idx)
        self.lambda_init = lambda_init
        self.lambda_q1 = ???
        self.lambda_k1 = ???
        self.lambda_q2 = ???
        self.lambda_k2 = ???
        
        # TODO: GroupNorm for output stabilization
        # GroupNorm(num_groups, num_channels) — one group per head, channels = d_model
        # Hint: nn.GroupNorm(n_heads, d_model)
        self.norm = ???
        
        self.rope_cache = RoPECache(self.d_k_half, max_seq_len)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        k = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        v = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        
        # TODO: split Q and K into two halves
        # Hint: q1, q2 = q.chunk(2, dim=-1)  — each [batch, seq, n_heads, d_k/2]
        q1, q2 = ???
        k1, k2 = ???
        
        # TODO: apply RoPE to each half separately
        freqs = self.rope_cache.get_freqs(seq_len)
        q1 = apply_rope(q1, freqs)
        # ... same for q2, k1, k2
        
        # Transpose all to [batch, n_heads, seq, d_k/2]
        q1 = q1.transpose(1, 2)
        # ... same for q2, k1, k2, and v (v stays full d_k)
        
        # TODO: compute λ scalar
        # λ = exp(λ_q1 · λ_k1) - exp(λ_q2 · λ_k2) + λ_init
        # Hint: torch.exp((self.lambda_q1 * self.lambda_k1).sum())
        lam = ???
        
        # TODO: two attention maps and subtract
        scale = self.d_k_half ** -0.5
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1
        )
        # map1 = softmax(q1 @ k1.T * scale + causal_mask)
        # map2 = softmax(q2 @ k2.T * scale + causal_mask)
        # attn_diff = map1 - lam * map2
        attn1 = ???
        attn2 = ???
        attn_diff = ???
        
        # TODO: apply to V, reshape, apply GroupNorm, output projection
        # Note: GroupNorm expects [batch, channels, ...] so transpose before/after
        attn_output = torch.matmul(attn_diff, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        # GroupNorm: self.norm(attn_output.transpose(1, 2)).transpose(1, 2)
        attn_output = ???
        return self.W_o(attn_output)
```

### Full Solution

```python
class DifferentialAttention(nn.Module):
    """Differential Attention: cancels attention noise via map subtraction."""
    
    def __init__(self, d_model, n_heads, layer_idx=0, dropout=0.0,
                 max_seq_len=2048, use_flash=True):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.d_k_half = self.d_k // 2
        self.dropout = dropout
        
        self.W_q = nn.Linear(d_model, n_heads * self.d_k,  bias=False)
        self.W_k = nn.Linear(d_model, n_heads * self.d_k,  bias=False)
        self.W_v = nn.Linear(d_model, n_heads * self.d_k,  bias=False)
        self.W_o = nn.Linear(d_model, d_model,              bias=False)
        
        lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_idx)
        self.lambda_init = lambda_init
        self.lambda_q1 = nn.Parameter(torch.randn(self.d_k_half) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(self.d_k_half) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(self.d_k_half) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(self.d_k_half) * 0.1)
        
        self.norm = nn.GroupNorm(n_heads, d_model)
        self.rope_cache = RoPECache(self.d_k_half, max_seq_len)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        k = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        v = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        
        # Split into two halves for differential computation
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)
        
        # Apply RoPE to each half
        freqs = self.rope_cache.get_freqs(seq_len)
        q1 = apply_rope(q1, freqs)
        q2 = apply_rope(q2, freqs)
        k1 = apply_rope(k1, freqs)
        k2 = apply_rope(k2, freqs)
        
        # Transpose: [batch, n_heads, seq, d_k/2]
        q1 = q1.transpose(1, 2)
        q2 = q2.transpose(1, 2)
        k1 = k1.transpose(1, 2)
        k2 = k2.transpose(1, 2)
        v  = v.transpose(1, 2)   # [batch, n_heads, seq, d_k]
        
        # Compute λ scalar from learnable vectors
        lam = (
            torch.exp((self.lambda_q1 * self.lambda_k1).sum())
            - torch.exp((self.lambda_q2 * self.lambda_k2).sum())
            + self.lambda_init
        )
        
        # Two attention maps, subtracted
        scale = self.d_k_half ** -0.5
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1
        )
        
        attn1 = F.softmax(torch.matmul(q1, k1.transpose(-2, -1)) * scale + causal_mask, dim=-1)
        attn2 = F.softmax(torch.matmul(q2, k2.transpose(-2, -1)) * scale + causal_mask, dim=-1)
        attn_diff = attn1 - lam * attn2
        
        # Apply to V
        attn_output = torch.matmul(attn_diff, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # GroupNorm: [batch, d_model, seq] → normalize → [batch, seq, d_model]
        attn_output = self.norm(attn_output.transpose(1, 2)).transpose(1, 2)
        
        return self.W_o(attn_output)
```

**Integration in TransformerBlock:**
```python
# TransformerBlock.__init__ needs layer_idx parameter:
def __init__(self, config, layer_idx=0):
    if config.use_diff_attn:
        self.attention = DifferentialAttention(
            d_model=config.d_model, n_heads=config.n_heads,
            layer_idx=layer_idx, dropout=config.dropout,
            max_seq_len=config.max_seq_len, use_flash=config.use_flash
        )
    else:
        self.attention = MultiHeadAttention(...)

# GPT.__init__ — pass layer_idx:
self.blocks = nn.ModuleList([
    TransformerBlock(config, layer_idx=i)
    for i in range(config.n_layers)
])
```

**Add to GPTConfig:**
```python
use_diff_attn: bool = True
```

### Interview Q&A

**"What's the compute cost vs standard attention?"**
Same parameter count, ~2x attention FLOPs (two maps vs one). But attention is not the bottleneck (FFN is ~4x larger). Wall-clock impact is small. At long context where attention IS the bottleneck, you can use fewer heads for equivalent quality — so net compute is similar.

**"What if λ=0?"**
You recover standard attention. λ=0 is always a valid optimum — the model can choose to ignore the differential mechanism per head. This makes initialization safe.

---

## Part 5: mHC (Manifold Hyper-Connections)

### Concept

Standard residual: `x = x + F(x)` — one stream.

mHC maintains N parallel streams mixed through a learned doubly-stochastic matrix A:
```
streams = [n_streams, batch, seq, d_model]
```
Each layer:
1. Aggregate streams → single input for F
2. Apply F (attention or MLP)
3. Mix streams through A (doubly stochastic via Sinkhorn)
4. Add scaled F output to all streams

**Doubly stochastic:** rows AND columns of A sum to 1. Mixing is always a convex combination — no explosion, no collapse. Gradient flow is more stable across 24 layers.

### Questions

**Q1:** A starts as identity. What does this mean for model behavior at initialization?

**Q2:** Why doubly stochastic rather than just row-stochastic?

**Q3:** At inference you collapse N streams via mean. Is this the right choice?

**Q4:** How much extra GPU memory does mHC cost at N=4, batch=16, seq=1024, d_model=1024?

### Answers

**A1:** Identity mixing: each stream maps only to itself at init. mHC behaves EXACTLY like standard residual at the start. The model begins in familiar territory and gradually learns to mix streams. Random A init would cause chaotic early training.

**A2:** Row-stochastic: each OUTPUT stream is a convex combination — prevents explosion. But one INPUT stream can still dominate ALL outputs (rich-get-richer). Column-stochastic: each input contributes equally to some output. Together: balanced flow in both directions.

**A3:** Mean averaging treats all N streams equally. The N streams specialize somewhat during training and averaging combines their specializations. The information isn't lost — all streams saw the full network depth. Alternative: learned weighted sum, or use only stream 0. Mean works well empirically.

**A4:** Extra activation memory = N × baseline per layer.
- Baseline: 16 × 1024 × 1024 × 2 bytes = 33.5MB per layer
- With mHC N=4: 4 × 33.5 = 134MB per layer
- 24 layers: ~3.2GB extra
- Well within 80GB VRAM — might reduce micro_batch from 16 to 12 to be safe.

### Try It

```python
# src/model/hyper_connection.py

import torch
import torch.nn as nn


def sinkhorn(log_A: torch.Tensor, iters: int = 20) -> torch.Tensor:
    """Project matrix to doubly stochastic via Sinkhorn-Knopp.
    
    Alternates normalizing rows and columns until both sum to 1.
    """
    # TODO:
    # 1. A = log_A.exp()  — work in probability space
    # 2. Loop iters times:
    #    a. A = A / A.sum(dim=-1, keepdim=True)  — normalize rows
    #    b. A = A / A.sum(dim=-2, keepdim=True)  — normalize cols
    # 3. return A
    pass


class HyperConnection(nn.Module):
    """mHC: N parallel residual streams mixed via doubly-stochastic matrix."""
    
    def __init__(self, d_model, n_streams=4, alpha_init=0.01, sinkhorn_iters=20):
        super().__init__()
        self.n_streams = n_streams
        self.sinkhorn_iters = sinkhorn_iters
        
        # TODO: three learnable parameters
        #
        # 1. self.log_A — mixing matrix logits [n_streams, n_streams]
        #    Init: torch.eye(n_streams).log()
        #    Why: after Sinkhorn → identity = standard residual at init
        #
        # 2. self.agg_weights — stream aggregation weights [n_streams]
        #    Init: torch.ones(n_streams) / n_streams  (uniform)
        #    Used to aggregate streams into single input for F
        #
        # 3. self.alpha — sublayer output scale (scalar)
        #    Init: torch.tensor(alpha_init)  (small, e.g. 0.01)
        #    Why small: sublayer output is noisy at init
        self.log_A = ???
        self.agg_weights = ???
        self.alpha = ???
    
    def forward(self, streams: torch.Tensor, sublayer) -> torch.Tensor:
        """
        Args:
            streams: [n_streams, batch, seq_len, d_model]
            sublayer: callable — the attention or MLP module
        Returns:
            Updated streams: [n_streams, batch, seq_len, d_model]
        """
        # TODO:
        # 1. Aggregate streams → single input
        #    a. weights = torch.softmax(self.agg_weights, dim=0)  [n_streams]
        #    b. aggregated = torch.einsum("n,nbsd->bsd", weights, streams)  [b, s, d]
        #
        # 2. Apply sublayer
        #    sublayer_out = sublayer(aggregated)  [b, s, d]
        #
        # 3. Get doubly-stochastic A
        #    A = sinkhorn(self.log_A, self.sinkhorn_iters)  [n, n]
        #
        # 4. Mix streams
        #    mixed = torch.einsum("mn,nbsd->mbsd", A, streams)  [n, b, s, d]
        #
        # 5. Add scaled sublayer output to all streams
        #    mixed = mixed + self.alpha * sublayer_out.unsqueeze(0)
        #
        # 6. return mixed
        pass
```

### Full Solution

```python
def sinkhorn(log_A: torch.Tensor, iters: int = 20) -> torch.Tensor:
    A = log_A.exp()
    for _ in range(iters):
        A = A / A.sum(dim=-1, keepdim=True)
        A = A / A.sum(dim=-2, keepdim=True)
    return A


class HyperConnection(nn.Module):
    def __init__(self, d_model, n_streams=4, alpha_init=0.01, sinkhorn_iters=20):
        super().__init__()
        self.n_streams = n_streams
        self.sinkhorn_iters = sinkhorn_iters
        
        # Identity init → standard residual behavior at start
        self.log_A = nn.Parameter(torch.eye(n_streams).log())
        # Uniform aggregation weights
        self.agg_weights = nn.Parameter(torch.ones(n_streams) / n_streams)
        # Small scale for stability
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
    
    def forward(self, streams: torch.Tensor, sublayer) -> torch.Tensor:
        # Aggregate streams into single input for sublayer
        weights = torch.softmax(self.agg_weights, dim=0)
        aggregated = torch.einsum("n,nbsd->bsd", weights, streams)
        
        # Apply sublayer
        sublayer_out = sublayer(aggregated)
        
        # Doubly-stochastic mixing
        A = sinkhorn(self.log_A, self.sinkhorn_iters)
        mixed = torch.einsum("mn,nbsd->mbsd", A, streams)
        
        # Add scaled sublayer output to all streams
        mixed = mixed + self.alpha * sublayer_out.unsqueeze(0)
        
        return mixed
```

**Integration in TransformerBlock:**
```python
def __init__(self, config, layer_idx=0):
    # ... norms, attention, mlp ...
    if config.use_mhc:
        from src.model.hyper_connection import HyperConnection
        self.attn_hc = HyperConnection(config.d_model, config.n_streams)
        self.mlp_hc  = HyperConnection(config.d_model, config.n_streams)

def forward(self, x):
    if self.config.use_mhc:
        # x is [n_streams, batch, seq, dim]
        x = self.attn_hc(x, lambda h: self.attention(self.norm1(h)))
        x = self.mlp_hc(x,  lambda h: self.mlp(self.norm2(h)))
    else:
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
    return x
```

**Integration in GPT.forward:**
```python
def forward(self, input_ids, targets=None):
    x = self.token_embeddings(input_ids)
    
    if self.config.use_mhc:
        # Expand to N streams
        x = x.unsqueeze(0).expand(self.config.n_streams, -1, -1, -1).clone()
    
    for block in self.blocks:
        x = block(x)
    
    if self.config.use_mhc:
        # Collapse streams via mean
        x = x.mean(dim=0)
    
    x = self.norm(x)
    logits = self.lm_head(x)
    # ... loss ...
    return logits, loss
```

**Add to GPTConfig:**
```python
use_mhc: bool = True
n_streams: int = 4
```

### Interview Q&A

**"Does mHC actually help?"**
Empirically from the DeepSeek paper: ~0.05-0.1 lower loss at equivalent parameter count. Your V1 vs V2 ablation will show the exact benefit on your data.

**"How does mHC relate to recurrent models?"**
Both maintain state across layers. mHC is like recurrence in layer space — each stream propagates through all layers. Key difference: fixed number of streams, fully differentiable, parallelizable across batch. Much more GPU-friendly than actual recurrence.

---

## Part 6: train.py Integration

### Updated TrainConfig

```python
@dataclass
class TrainConfig:
    # Data
    data_dir: str = "data/fineweb-edu/train"
    val_dir: str = "data/fineweb-edu/val"
    seq_len: int = 1024

    # Batch (same effective batch, larger micro_batch for efficiency)
    micro_batch_size: int = 32
    grad_accum_steps: int = 16   # 32 * 16 * 1024 = 524K tokens

    # Optimizer
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    muon_lr: float = 1.5e-4     # 0.5x max_lr
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Schedule
    warmup_steps: int = 1000
    max_steps: int = 15000       # Muon converges faster

    # Logging and Checkpointing
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000
    checkpoint_dir: str = "checkpoints"

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9995

    # Hardware
    device: str = "cuda"
```

### Updated GPTConfig

```python
@dataclass
class GPTConfig:
    vocab_size: int = 50304
    d_model: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 4         # GQA
    dropout: float = 0.0
    max_seq_len: int = 1024
    use_flash: bool = True
    tie_weights: bool = True
    use_mhc: bool = True
    n_streams: int = 4
    use_diff_attn: bool = True
    use_qk_norm: bool = True
```

### Updated Training Loop

```python
# Try it — fill in the ???:

def train(config: TrainConfig):
    # ... setup, model creation ...
    
    # TODO: Two optimizers instead of one
    # Hint: from src.optim.muon import configure_optimizers
    # muon_opt, adamw_opt = configure_optimizers(model, lr=..., muon_lr=..., weight_decay=...)
    muon_opt, adamw_opt = ???
    
    # TODO: EMA
    # Hint: ema = EMA(model, decay=config.ema_decay)
    ema = ???
    
    model.train()
    for step in range(start_step, config.max_steps):
        
        # TODO: LR — only update AdamW (Muon LR is fixed)
        lr = get_lr(step, ...)
        for pg in ???.param_groups:
            pg["lr"] = lr
        
        # TODO: Zero both optimizers
        ???.zero_grad()
        ???.zero_grad()
        
        # ... gradient accumulation loop unchanged ...
        
        # TODO: Clip and step BOTH
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        ???.step()
        ???.step()
        
        # TODO: Update EMA after optimizer steps
        if ema:
            ???.update(model)
        
        # TODO: Eval uses EMA model if available
        if step > 0 and step % config.eval_interval == 0:
            eval_model = ema.model if ema else model
            val_loss = evaluate(???, val_dataset, config, device, dtype)
```

### Full Solution

```python
def train(config: TrainConfig):
    # ... setup ...
    
    from src.optim.muon import configure_optimizers
    muon_opt, adamw_opt = configure_optimizers(
        model, lr=config.max_lr, muon_lr=config.muon_lr,
        weight_decay=config.weight_decay
    )
    
    ema = EMA(model, decay=config.ema_decay) if config.use_ema else None
    
    model.train()
    for step in range(start_step, config.max_steps):
        lr = get_lr(step, config.warmup_steps, config.max_steps, config.max_lr, config.min_lr)
        for pg in adamw_opt.param_groups:
            pg["lr"] = lr
        
        muon_opt.zero_grad()
        adamw_opt.zero_grad()
        
        # ... gradient accumulation (unchanged) ...
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        muon_opt.step()
        adamw_opt.step()
        
        if ema:
            ema.update(model)
        
        if step > 0 and step % config.eval_interval == 0:
            eval_model = ema.model if ema else model
            val_loss = evaluate(eval_model, val_dataset, config, device, dtype)
            print(f"  >>> val_loss (EMA): {val_loss:.4f}")
            for prompt in EVAL_PROMPTS:
                sample = generate_sample(eval_model, enc, device, prompt=prompt)
                print(f"  >>> [{prompt}] {sample[:150]}")
            model.train()

    # Save final checkpoint
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    path = os.path.join(config.checkpoint_dir, f"step_{config.max_steps:06d}.pt")
    checkpoint = {
        "model_weights": model.state_dict(),
        "muon_state": muon_opt.state_dict(),
        "adamw_state": adamw_opt.state_dict(),
        "step": config.max_steps,
        "rng_state": torch.random.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
    }
    if ema:
        checkpoint["ema_weights"] = ema.state_dict()
    torch.save(checkpoint, path)
    print(f"✓ Final checkpoint saved: {path}")
```

---

## Part 7: SFT

### Concept

Pretraining: model learns to predict next token in raw text. Doesn't know what a "question" is.

SFT: show it thousands of examples of the behavior you want:
```
<|endoftext|>User: What is photosynthesis?<|endoftext|>Assistant: Photosynthesis is...
```

The model learns: "when I see `Assistant:` after a question, I should produce a helpful answer." This teaches BEHAVIOR, not new knowledge — knowledge is already in pretrained weights.

### Key Differences from Pretraining

| | Pretraining | SFT |
|--|--|--|
| Data | Raw text | Instruction-response pairs |
| Loss | All tokens | Response tokens only (prompt masked) |
| LR | 3e-4 | 2e-5 |
| Epochs | 1 | 2-3 |
| Goal | Learn language | Learn behavior |

### Questions

**Q1:** Why mask prompt tokens? What would happen if you computed loss on them too?

**Q2:** Why 2-3 epochs, not 10?

**Q3:** Why LR 2e-5 instead of 3e-4?

**Q4:** What does catastrophic forgetting look like in SFT?

**Q5:** How do you evaluate whether SFT worked?

### Answers

**A1:** Prompt tokens are INPUTS — the model shouldn't be predicting the user's question, it should be answering it. Computing loss on prompts creates a degenerate gradient signal: the model tries to predict deterministic prompt tokens given the full context, wasting gradient updates on learning user behavior rather than assistant behavior.

**A2:** At 10 epochs: the model memorizes exact responses (overfitting), starts producing formulaic outputs, mode-collapses to always outputting similar-length responses, and may start to catastrophically forget pretraining knowledge. 2-3 epochs teaches the behavior without destroying the foundation.

**A3:** Too-high LR during SFT causes catastrophic forgetting — pretrained weights shift so far the model loses its language understanding. 2e-5 makes small, careful nudges. General language capabilities stay intact while behavior shifts toward instruction-following.

**A4:** Signs: loss on held-out pretraining data increases; generated text outside the SFT format becomes grammatically broken; model only "knows how to answer questions" and can no longer generate natural prose; responses become formulaic.

**A5:**
1. **Format compliance:** Does it stop at `<|endoftext|>`? Does it produce answers vs continuations?
2. **Qualitative:** Ask it questions, compare before/after. Does it ANSWER vs CONTINUE?
3. **Held-out task accuracy:** Test on questions not in SFT data — factual Q, code correctness, math.

### SFT Dataset — Try It

```python
# src/data/sft_dataset.py

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import tiktoken


class SFTDataset(Dataset):
    def __init__(self, data_sources, max_seq_len=1024, enc=None):
        if enc is None:
            enc = tiktoken.get_encoding("gpt2")
        self.enc = enc
        self.max_seq_len = max_seq_len
        self.eot = enc.eot_token
        self.examples = []
        
        for source in data_sources:
            self.examples.extend(self._load_source(source))
        print(f"SFT dataset: {len(self.examples):,} examples")
    
    def _load_source(self, source):
        # TODO:
        # 1. load_dataset(source["name"], split=source.get("split", "train"))
        # 2. Loop with max_examples limit
        # 3. Handle OpenHermes special case:
        #    if "conversations" in item:
        #        instruction = item["conversations"][0]["value"]
        #        response = item["conversations"][1]["value"]
        #    else:
        #        use source["instruction_key"] and source["response_key"]
        # 4. Skip if len(conversations) < 2
        # 5. Return list of {"instruction": ..., "response": ...}
        pass
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # TODO:
        # Format: [EOT] + "User: {instruction}" + [EOT] + "Assistant: {response}" + [EOT]
        #
        # 1. user_tokens = [self.eot] + enc.encode(user_text, disallowed_special=()) + [self.eot]
        #    asst_tokens = enc.encode(asst_text, disallowed_special=()) + [self.eot]
        #
        # 2. full_tokens = user_tokens + asst_tokens
        #
        # 3. Truncate if len(full_tokens) > max_seq_len + 1
        #
        # 4. input_ids = full_tokens[:-1]
        #    targets = full_tokens[1:]
        #
        # 5. Loss mask:
        #    prompt_len = len(user_tokens) - 1  # -1 for shift
        #    loss_mask = [-100] * min(prompt_len, len(targets)) + [1] * max(0, len(targets) - prompt_len)
        #
        # 6. Pad all to max_seq_len with eot / -100
        #
        # 7. Return (tensor, tensor, tensor)
        pass
```

### SFT Dataset — Full Solution

```python
class SFTDataset(Dataset):
    def __init__(self, data_sources, max_seq_len=1024, enc=None):
        if enc is None:
            enc = tiktoken.get_encoding("gpt2")
        self.enc = enc
        self.max_seq_len = max_seq_len
        self.eot = enc.eot_token
        self.examples = []
        for source in data_sources:
            self.examples.extend(self._load_source(source))
        print(f"SFT dataset: {len(self.examples):,} examples")
    
    def _load_source(self, source):
        dataset = load_dataset(source["name"], split=source.get("split", "train"))
        instruction_key = source.get("instruction_key", "instruction")
        response_key = source.get("response_key", "output")
        max_examples = source.get("max_examples", None)
        
        examples = []
        for i, item in enumerate(dataset):
            if max_examples and i >= max_examples:
                break
            if "conversations" in item:
                turns = item["conversations"]
                if len(turns) < 2:
                    continue
                instruction = turns[0]["value"]
                response = turns[1]["value"]
            else:
                instruction = item[instruction_key]
                response = item[response_key]
            examples.append({"instruction": instruction, "response": response})
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        user_text = f"User: {example['instruction']}"
        asst_text = f"Assistant: {example['response']}"
        
        user_tokens = [self.eot] + self.enc.encode(user_text, disallowed_special=()) + [self.eot]
        asst_tokens = self.enc.encode(asst_text, disallowed_special=()) + [self.eot]
        
        full_tokens = user_tokens + asst_tokens
        if len(full_tokens) > self.max_seq_len + 1:
            full_tokens = full_tokens[:self.max_seq_len + 1]
        
        input_ids = full_tokens[:-1]
        targets   = full_tokens[1:]
        
        prompt_len = len(user_tokens) - 1
        loss_mask = [-100] * min(prompt_len, len(targets)) + [1] * max(0, len(targets) - prompt_len)
        
        pad_len = self.max_seq_len - len(input_ids)
        input_ids = input_ids + [self.eot] * pad_len
        targets   = targets   + [-100] * pad_len
        loss_mask = loss_mask + [-100] * pad_len
        
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(targets,   dtype=torch.long),
            torch.tensor(loss_mask, dtype=torch.long),
        )
```

### sft.py — Try It

```python
# sft.py

def sft_train(config: SFTConfig):
    torch.manual_seed(42)
    device = config.device
    enc = tiktoken.get_encoding("gpt2")
    
    # TODO: init wandb if available
    
    # TODO: load base model from checkpoint
    # Hint: model_config = GPTConfig()
    #       model = GPT(model_config).to(device)
    #       checkpoint = torch.load(config.base_checkpoint, weights_only=False)
    #       model.load_state_dict(checkpoint["model_weights"])
    
    # TODO: torch.compile
    
    # TODO: create SFTDataset and DataLoader
    #   dataset = SFTDataset(config.data_sources, config.max_seq_len, enc)
    #   loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, ...)
    
    # TODO: compute total_steps = (len(loader) // grad_accum_steps) * num_epochs
    
    # TODO: create AdamW optimizer with config.lr, config.weight_decay
    
    # Training loop
    model.train()
    for step in range(total_steps):
        # TODO: LR schedule
        # TODO: zero_grad
        # TODO: grad accum loop:
        #   - next(train_iter) → input_ids, targets, loss_mask
        #   - autocast forward
        #   - sft_loss(logits, targets, loss_mask) / grad_accum_steps
        #   - backward
        # TODO: clip, step, log, save
    
    # TODO: final save
```

### sft.py — Full Solution

```python
# sft.py
"""Supervised Fine-Tuning for the 350M LLM."""

import os
import time
import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
import tiktoken

from src.model.gpt import GPT, GPTConfig
from src.data.sft_dataset import SFTDataset

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


@dataclass
class SFTConfig:
    base_checkpoint: str = "checkpoints/step_020000.pt"
    checkpoint_dir: str = "checkpoints_sft"
    data_sources: list = field(default=None)
    max_seq_len: int = 1024
    batch_size: int = 16
    grad_accum_steps: int = 4
    num_epochs: int = 3
    lr: float = 2e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 100
    log_interval: int = 10
    eval_interval: int = 200
    save_interval: int = 500
    device: str = "cuda"
    
    def __post_init__(self):
        if self.data_sources is None:
            self.data_sources = [
                {"name": "teknium/OpenHermes-2.5",    "split": "train", "max_examples": 600_000},
                {"name": "meta-math/MetaMathQA",       "split": "train", "instruction_key": "query", "response_key": "response", "max_examples": 300_000},
                {"name": "sahil2801/CodeAlpaca-20k",   "split": "train", "max_examples": 20_000},
            ]


def sft_loss(logits, targets, loss_mask):
    masked_targets = targets.clone()
    masked_targets[loss_mask == -100] = -100
    return F.cross_entropy(logits.view(-1, logits.size(-1)), masked_targets.view(-1), ignore_index=-100)


def get_sft_lr(step, warmup_steps, total_steps, lr):
    if step < warmup_steps:
        return lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return lr * (1 + math.cos(math.pi * progress)) / 2


def sft_train(config: SFTConfig):
    torch.manual_seed(42)
    device = config.device
    enc = tiktoken.get_encoding("gpt2")
    
    if HAS_WANDB:
        wandb.init(project="llm-350m-sft", config=vars(config))
    
    # Load base model
    model_config = GPTConfig()
    model = GPT(model_config).to(device)
    print(f"Loading from {config.base_checkpoint}...")
    checkpoint = torch.load(config.base_checkpoint, weights_only=False)
    model.load_state_dict(checkpoint["model_weights"])
    print("✓ Base model loaded")
    model = torch.compile(model)
    
    # Data
    train_dataset = SFTDataset(config.data_sources, config.max_seq_len, enc)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, pin_memory=True, drop_last=True, num_workers=4)
    steps_per_epoch = len(train_loader) // config.grad_accum_steps
    total_steps = steps_per_epoch * config.num_epochs
    print(f"✓ Training for {config.num_epochs} epochs, {total_steps} steps")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                   weight_decay=config.weight_decay, betas=(0.9, 0.95), fused=True)
    
    # Training loop
    model.train()
    train_iter = iter(train_loader)
    
    for step in range(total_steps):
        t_start = time.time()
        lr = get_sft_lr(step, config.warmup_steps, total_steps, config.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        
        optimizer.zero_grad()
        running_loss = 0.0
        
        for _ in range(config.grad_accum_steps):
            try:
                input_ids, targets, loss_mask = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                input_ids, targets, loss_mask = next(train_iter)
            
            input_ids = input_ids.to(device)
            targets   = targets.to(device)
            loss_mask = loss_mask.to(device)
            
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, _ = model(input_ids)
                loss = sft_loss(logits, targets, loss_mask)
            
            loss = loss / config.grad_accum_steps
            loss.backward()
            running_loss += loss.item()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        step_time = time.time() - t_start
        
        if step % config.log_interval == 0:
            print(f"Step {step:>6d} | Loss {running_loss:.4f} | LR {lr:.2e} | "
                  f"Grad Norm {grad_norm:.2f} | dt {step_time*1000:.0f}ms")
            if HAS_WANDB:
                wandb.log({"train/loss": running_loss, "train/lr": lr}, step=step)
        
        if step > 0 and step % config.save_interval == 0:
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            path = os.path.join(config.checkpoint_dir, f"sft_step_{step:06d}.pt")
            torch.save({"model_weights": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(), "step": step}, path)
            print(f"  >>> Saved: {path}")
    
    # Final save
    path = os.path.join(config.checkpoint_dir, "sft_final.pt")
    torch.save({"model_weights": model.state_dict(), "step": total_steps}, path)
    print(f"\nSFT complete! Final model: {path}")
    if HAS_WANDB:
        wandb.finish()


if __name__ == "__main__":
    config = SFTConfig()
    sft_train(config)
```

### Expected SFT Results

| | Value |
|--|--|
| Dataset | ~920K examples |
| Epochs | 3 |
| Time on A100 | ~3-4 hours |
| Cost | ~$5-8 |
| Starting loss | ~2.9 |
| Final SFT loss | ~1.5-2.0 |

### Before/After Eval Prompts

```
User: What is the capital of Australia?
Base:  "...is a country in the Southern Hemisphere. The capital is often..."
SFT:   "The capital of Australia is Canberra."

User: What is 15% of 80?
Base:  "...percent of the total is calculated by..."
SFT:   "15% of 80 is 12. To calculate: 80 × 0.15 = 12."

User: Write a Python function to reverse a string.
Base:  "...has many methods. The reverse() method is used..."
SFT:   "def reverse_string(s):\n    return s[::-1]"
```

### Interview Q&A

**"Why does SFT on 1M examples teach instruction-following, but 10B tokens of pretraining doesn't?"**
Pretraining teaches WHAT language looks like. The format is always "predict next token in text." SFT teaches the ROLE — "I am an assistant, my job is to answer questions." The model already has the capability; SFT teaches it WHEN to use which capability.

**"Could you do RLHF instead of SFT?"**
RLHF is applied AFTER SFT, not instead. RL training needs a well-behaved starting point. Starting RL from a raw pretrained model produces garbage that's too chaotic for a reward model to score usefully.

**"How do you prevent the model forgetting pretraining during SFT?"**
Low LR (2e-5) makes small, careful nudges. You can also mix in a small amount (5-10%) of pretraining data as "replay" to prevent forgetting. LoRA (only updating small adapters) is another option at larger scales.

---

## Implementation Order

```
Local (no GPU):
├── EMA class in train.py                    (20 min)
├── QK-Norm in attention.py                  (30 min)
├── GQA in attention.py                      (1 hour)
├── Differential Attention in attention.py   (2-3 hours)
├── mHC in hyper_connection.py               (2 hours)
├── Update GPTConfig flags                   (30 min)
├── Update train.py (Muon + EMA)             (1-2 hours)
└── Smoke test on tiny model                 (30 min)

GPU:
├── 500-step validation A100 (~$3)
├── Full V2 training H100 SXM (~$50, 15K steps)
└── SFT V1 on A100 (~$8, 3-4 hours)
```
