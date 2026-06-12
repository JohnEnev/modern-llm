# V2 Modules + SFT Guide
*Offline reference: questions first, then explanations, then full code, then interview Q&A.*

---

## Status

**Done:**
- ✅ Muon optimizer (`src/optim/muon.py`)

**V2 remaining (this doc):**
- ⬜ mHC Hyper-Connections (`src/model/hyper_connection.py`)
- ⬜ Differential Attention (modify `src/model/attention.py`)
- ⬜ QK-Norm + learnable scale (modify `src/model/attention.py`)
- ⬜ GQA (modify `src/model/attention.py`)
- ⬜ EMA (add to `train.py`)
- ⬜ train.py integration (two optimizers, EMA, new config flags)

**SFT (this doc):**
- ⬜ SFT dataset class (`src/data/sft_dataset.py`)
- ⬜ SFT training script (`sft.py`)

---

## Part 1: EMA (Exponential Moving Average)

### Concept

During training, model weights fluctuate step to step — the optimizer overshoots, then corrects, then overshoots again. EMA keeps a smoothed copy of the weights:

```
ema_weights = decay * ema_weights + (1 - decay) * current_weights
```

After training, the EMA model has seen a weighted average of all recent checkpoints. It's like taking the "center of mass" of where the model has been rather than where it ended up. Empirically gives 0.02-0.05 lower val loss for free.

**Decay = 0.9995** means the EMA weights update slowly — they weight the last 2000 steps roughly equally. Lower decay = faster update, more noise. Higher decay = slower update, smoother.

**Key rule:** Use EMA model for evaluation and inference, NOT for training. The training model continues updating normally; EMA just shadows it.

### Questions to Answer

**Q1:** If decay=0.9995 and you train for 20,000 steps, roughly how many recent steps does the EMA weight equally? (Hint: half-life = log(0.5) / log(decay))

**Q2:** Why use EMA weights for inference but NOT for training? What would happen if you continued training from the EMA checkpoint?

**Q3:** EMA is sometimes called "Polyak averaging." What's the theoretical justification — why should the average of past weights be better than the final weights?

### Answers

**A1:** Half-life = log(0.5) / log(0.9995) ≈ 1386 steps. So the EMA weights roughly average over the last ~2700 steps (two half-lives). Recent steps count more, older steps decay exponentially.

**A2:** EMA weights are smoother but they lag behind the training trajectory. If you trained FROM the EMA checkpoint, you'd lose the momentum state and be starting fresh with stale-ish weights. The gradients would be computed against a slightly different point in weight space. More importantly, the optimizer state (Adam's m and v buffers) corresponds to the TRAINING model, not the EMA model — resuming from EMA would discard all that learned optimization history.

**A3:** SGD and Adam find weights that minimize training loss, but the loss surface is non-convex and noisy. The optimizer may oscillate around a basin of attraction without settling perfectly. Averaging reduces the variance of this oscillation — you're averaging across many points near the true minimum, which by Jensen's inequality (for convex regions) lands closer to the minimum than any individual noisy point.

### Code

```python
# Add to train.py

class EMA:
    """Exponential Moving Average of model weights.
    
    Maintains a smoothed copy of model weights updated after each
    optimizer step. Use ema.model for evaluation and inference.
    
    Args:
        model: The training model to shadow
        decay: EMA decay rate (0.9995 = smooth, 0.999 = slightly faster)
    """
    
    def __init__(self, model: torch.nn.Module, decay: float = 0.9995):
        self.decay = decay
        
        # Create a deep copy of the model for EMA weights
        # This model is never trained directly — only updated via EMA
        import copy
        self.model = copy.deepcopy(model)
        self.model.eval()  # always in eval mode
        
        # Freeze EMA model — we update it manually, not via gradients
        for param in self.model.parameters():
            param.requires_grad_(False)
    
    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        """Update EMA weights after each optimizer step."""
        for ema_param, train_param in zip(self.model.parameters(), model.parameters()):
            # EMA update: smooth blend of old EMA and new training weights
            ema_param.data.mul_(self.decay).add_(train_param.data, alpha=1 - self.decay)
    
    def state_dict(self):
        """For checkpointing."""
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        """For resuming from checkpoint."""
        self.model.load_state_dict(state_dict)
```

**Integration in train.py:**

```python
# After creating model and before training loop:
ema = EMA(model, decay=config.ema_decay)  # config.ema_decay = 0.9995

# Inside the training loop, after optimizer.step():
ema.update(model)  # shadow the training model

# In the eval block:
if step > 0 and step % config.eval_interval == 0:
    # Evaluate EMA model, not training model
    val_loss = evaluate(ema.model, val_dataset, config, device, dtype)
    print(f"  >>> val_loss (EMA): {val_loss:.4f}")
    
    # Generate samples from EMA model
    for prompt in EVAL_PROMPTS:
        sample = generate_sample(ema.model, enc, device, prompt=prompt)
        print(f"  >>> [{prompt}] {sample[:150]}")
    
    model.train()  # EMA model stays in eval, only training model switches

# In checkpointing:
checkpoint = {
    "model_weights": model.state_dict(),
    "ema_weights": ema.state_dict(),   # save EMA separately
    "optimizer_state": ...,
    "step": step,
    ...
}
```

**Add to TrainConfig:**
```python
use_ema: bool = True
ema_decay: float = 0.9995
```

### Interview Q&A

**"Why not just use the last checkpoint?"**
The last checkpoint is wherever the optimizer happened to land at step N. The EMA is the average of where the optimizer has been for the last ~2000 steps — much less sensitive to the noise of any single step.

**"What's the relationship between EMA decay and batch size?"**
Larger batch sizes have less gradient noise per step, so the model is already more stable. You might use slightly lower decay (e.g., 0.999) with small batches and higher decay (0.9999) with large batches. But 0.9995 is robust across most training setups.

**"Does EMA help with fine-tuning too?"**
Yes, but less so. SFT training is shorter (1-3 epochs) and more aggressive — you WANT the model to change its behavior quickly. EMA decay can be lower (0.999 or even 0.99) or skipped entirely for short SFT runs.

---

## Part 2: QK-Norm + Learnable Scale

### Concept

In standard attention, Q and K vectors can have arbitrary magnitude. The dot product Q·K grows with magnitude, making softmax output increasingly sharp (attending almost entirely to one position) or flat (attention entropy collapse). This destabilizes training in deep networks.

QK-Norm normalizes Q and K to unit length before computing attention scores, then scales by a learnable parameter per head. This prevents both extremes:
- Prevents one position from dominating (attention spikes)
- Prevents all positions getting equal weight (attention collapse)

**Standard attention:**
```
scores = Q @ K.T / sqrt(d_k)
```

**With QK-Norm:**
```
Q_norm = F.normalize(Q, dim=-1) * scale
K_norm = F.normalize(K, dim=-1)
scores = Q_norm @ K_norm.T  # scale already applied
```

The learnable `scale` per head allows each head to independently control the sharpness of its attention distribution.

### Questions to Answer

**Q1:** After QK-norm, all Q and K vectors have magnitude 1. What is the range of dot products between any two unit vectors? What does this imply for the softmax temperature?

**Q2:** Why a learnable scale rather than a fixed value like `sqrt(d_k)`? What can a fixed scale not adapt to?

**Q3:** At initialization, the scale should be set so the attention distribution looks similar to pre-norm attention. What value should the scale start at? (Hint: think about what the pre-norm attention score magnitude is before normalization)

**Q4:** QK-Norm changes the gradient flow through attention. Before QK-norm, a very large Q vector has a large gradient through the score computation. What happens to the gradient magnitude after QK-norm?

### Answers

**A1:** Dot product of two unit vectors = cos(θ) ∈ [-1, 1]. This means attention logits are ALWAYS in [-1, 1] regardless of training duration or layer depth. The softmax temperature is now entirely controlled by the scale factor — QK-norm separates the direction (what to attend to) from the magnitude (how sharply to attend).

**A2:** A fixed scale can't adapt to different optimal sharpness per head. Some heads specialize in sharp, local attention (syntax); others maintain broad, diffuse attention (semantics). A learnable scale lets each head find its optimal sharpness independently during training.

**A3:** Pre-norm, Q·K ≈ sqrt(d_k) for randomly initialized weights (sum of d_k random products, each ≈ N(0,1/d_k)). After QK-norm, Q·K ∈ [-1, 1]. To match the original scale, initialize scale = sqrt(d_k) = sqrt(64) = 8 for your 16-head model with d_k=64. This way the transition to QK-norm doesn't shock the model if you're adding it mid-training. For training from scratch, you can initialize to 1.0 or any reasonable value since the model learns from scratch anyway.

**A4:** QK-norm introduces a normalization step, which projects gradients onto the tangent space of the unit sphere. This clips extremely large gradients from large Q/K vectors — effectively providing built-in gradient clipping for the attention computation. This is why QK-norm improves stability: it caps the gradient magnitude independently of the gradient clipping in the training loop.

### Code

Modify `src/model/attention.py`:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads=None, dropout=0.0, 
                 max_seq_len=2048, use_flash=True, use_qk_norm=True):
        super().__init__()
        # ... existing init ...
        
        # QK-Norm: learnable scale per head for Q (K is unit-normalized only)
        # Initialize to sqrt(d_k) so pre-norm attention distribution is preserved
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.qk_scale = nn.Parameter(
                torch.ones(n_heads) * (self.d_k ** 0.5)
            )  # [n_heads] — one scale per head
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        k = self.W_k(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k)
        v = self.W_v(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k)
        
        # Apply QK-Norm before RoPE
        if self.use_qk_norm:
            # Normalize to unit sphere, then scale by learnable per-head factor
            # Shape: [batch, seq, n_heads, d_k]
            q = F.normalize(q, dim=-1) * self.qk_scale.view(1, 1, -1, 1)
            k = F.normalize(k, dim=-1)  # K is normalized but not scaled
        
        # Apply RoPE (after QK-norm — normalization then position encoding)
        freqs = self.rope_cache.get_freqs(seq_len)
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)
        
        # ... rest of forward (transpose, attention, output) unchanged ...
```

**Add to GPTConfig:**
```python
use_qk_norm: bool = True
```

### Interview Q&A

**"Is QK-Norm equivalent to a temperature parameter?"**
Partially. A single global temperature scales all attention scores uniformly. QK-Norm with per-head scale is more flexible — it's N_heads separate temperature parameters, each learned independently. Also, QK-Norm bounds the logit range to [-scale, scale] which a temperature parameter doesn't.

**"Why normalize Q but scale it, while K is only normalized?"**
Convention — you only need one side scaled (Q or K, not both) since scaling Q by s is equivalent to scaling K by 1/s in the dot product. The paper puts the scale on Q. Some implementations scale both by sqrt(scale).

**"Does QK-Norm replace the sqrt(d_k) scaling in attention?"**
Yes — after QK-norm, the dot products are already bounded in [-scale, scale], so the additional 1/sqrt(d_k) scaling in standard attention would double-scale. Remove the standard scaling when using QK-Norm.

---

## Part 3: GQA (Grouped Query Attention)

### Concept

Standard multi-head attention has n_heads Q heads, n_heads K heads, n_heads V heads. During inference, you cache K and V for all previous tokens — the KV cache. For a 350M model with 16 heads, this is manageable. For a 7B model with long sequences and many concurrent users, it becomes the memory bottleneck.

**GQA groups Q heads together to share K and V heads:**

```
Standard MHA:     16 Q heads, 16 K heads, 16 V heads
GQA (4 groups):   16 Q heads,  4 K heads,  4 V heads
MQA (extreme):    16 Q heads,  1 K head,   1 V head
```

With 4 KV groups, each group of 4 Q heads shares one K and one V head. K/V cache is 4x smaller.

At training time: slightly faster (fewer K/V projections). Slight quality loss (~0.02-0.03 higher loss) because Q heads within a group see the same K/V representation. But at SERVING time: 4x smaller KV cache = serve 4x more concurrent users or handle 4x longer contexts.

### Questions to Answer

**Q1:** Your current model has d_model=1024, n_heads=16, so d_k=64. With GQA using 4 KV groups, what are the shapes of the W_q, W_k, W_v weight matrices? How does this change your parameter count?

**Q2:** During the forward pass with GQA, Q has shape [batch, seq, 16, 64] but K has shape [batch, seq, 4, 64]. How do you compute attention — you can't directly multiply 16 Q heads against 4 K heads?

**Q3:** GQA is a quality-efficiency tradeoff. At what model scale does this tradeoff start to favor GQA? Think about when KV cache size actually matters.

**Q4:** LLaMA-2 7B uses 32 Q heads and 8 KV heads (4 Q per KV group). LLaMA-2 70B uses 64 Q heads and 8 KV heads (8 Q per KV group). Why does the grouping ratio increase with model size?

### Answers

**A1:**
- W_q: [d_model, n_heads * d_k] = [1024, 16 * 64] = [1024, 1024] — unchanged
- W_k: [d_model, n_kv_heads * d_k] = [1024, 4 * 64] = [1024, 256] — 4x smaller
- W_v: [d_model, n_kv_heads * d_k] = [1024, 4 * 64] = [1024, 256] — 4x smaller

Parameter savings: 2 × (1024 × 1024 - 1024 × 256) = 2 × 786,432 ≈ 1.57M parameters. At 350M total, that's <0.5% — negligible. The win is at serving, not training.

**A2:** Expand (repeat) the KV heads to match Q heads. Each KV group has 4 Q heads, so each K/V head is repeated 4 times:
```python
n_rep = n_heads // n_kv_heads  # = 4
k = k.repeat_interleave(n_rep, dim=2)  # [batch, seq, 4, d_k] → [batch, seq, 16, d_k]
v = v.repeat_interleave(n_rep, dim=2)
# Now compute attention as usual
```

**A3:** GQA matters when KV cache is the memory bottleneck. For 350M at 1024 context: KV cache ≈ 2 × 24 × 16 × 1024 × 64 × 2 bytes ≈ 100MB — trivial. For 7B at 4096 context with 100 concurrent users: KV cache ≈ 100 × 2 × 32 × 32 × 4096 × 128 × 2 bytes ≈ 160GB — catastrophic. GQA becomes essential around 3-7B scale in production serving.

**A4:** Larger models have more capacity per head — the KV representation is richer, so more Q heads can share it without quality loss. At 70B, each KV head has 128 dimensions (d_k=128) containing much more information than 64 dimensions. You can afford to share more aggressively.

### Code

Modify `src/model/attention.py`:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads=None, dropout=0.0,
                 max_seq_len=2048, use_flash=True, use_qk_norm=True):
        super().__init__()
        assert d_model % n_heads == 0
        
        # GQA: n_kv_heads can be less than n_heads
        # If None, defaults to standard MHA (n_kv_heads = n_heads)
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.n_rep = n_heads // self.n_kv_heads  # how many Q heads share each KV head
        assert n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        
        self.d_k = d_model // n_heads
        self.d_model = d_model
        self.dropout = dropout
        self.use_flash = use_flash
        
        # Q projection: full n_heads
        self.W_q = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        # K/V projections: only n_kv_heads (4x smaller with 4 groups)
        self.W_k = nn.Linear(d_model, self.n_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, self.n_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.rope_cache = RoPECache(self.d_k, max_seq_len)
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # QK-Norm
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.qk_scale = nn.Parameter(torch.ones(n_heads) * (self.d_k ** 0.5))
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.W_q(x).view(batch_size, seq_len, self.n_heads,    self.d_k)
        k = self.W_k(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k)
        v = self.W_v(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k)
        
        # QK-Norm
        if self.use_qk_norm:
            q = F.normalize(q, dim=-1) * self.qk_scale.view(1, 1, -1, 1)
            k = F.normalize(k, dim=-1)
        
        # RoPE
        freqs = self.rope_cache.get_freqs(seq_len)
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)
        
        # GQA: expand K/V to match Q heads
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)  # [b, s, n_kv, d] → [b, s, n_heads, d]
            v = v.repeat_interleave(self.n_rep, dim=2)
        
        # Transpose for attention: [batch, n_heads, seq_len, d_k]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Attention
        if self.use_flash:
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
                scale=1.0 if self.use_qk_norm else None,  # QK-norm handles scaling
            )
        else:
            # Manual attention
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
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(attn_output)
```

**Add to GPTConfig:**
```python
n_kv_heads: int = 4   # set to n_heads for standard MHA, < n_heads for GQA
```

**Update TransformerBlock to pass n_kv_heads:**
```python
self.attention = MultiHeadAttention(
    d_model=config.d_model,
    n_heads=config.n_heads,
    n_kv_heads=config.n_kv_heads,  # add this
    ...
)
```

### Interview Q&A

**"Does GQA hurt quality much?"**
At 350M scale: ~0.02-0.03 higher loss. At 7B+ scale with enough data: essentially no degradation. LLaMA-2 70B uses GQA and is competitive with models of similar size that use MHA. The quality penalty diminishes as model size and training data increase.

**"What's the difference between GQA and MQA (Multi-Query Attention)?"**
MQA is the extreme case: one shared K/V head for all Q heads. GQA is the generalization: G shared groups. MQA is cheapest but has more quality degradation. GQA finds a sweet spot — typically G=n_heads/4 or G=8 regardless of model size. MQA was used in PaLM; GQA became standard with LLaMA-2+.

**"Does GQA change the KV cache size?"**
Yes — that's the whole point. KV cache ∝ n_kv_heads, so 4 KV groups instead of 16 = 4x smaller cache. At batch_size=1, 1024 context, 24 layers, d_k=64: KV cache = 2 × 24 × 4 × 1024 × 64 × 2 bytes = 25MB instead of 100MB.

---

## Part 4: Differential Attention

### Concept

Standard attention computes ONE softmax attention map per head. The problem: that single map assigns non-trivial attention weights to irrelevant tokens — this is "attention noise." The model wastes capacity learning to ignore this noise.

**Differential Attention** (Microsoft, 2024) computes TWO attention maps per head and subtracts them:

```
Attention_diff = softmax(Q1 @ K1.T / scale) - λ * softmax(Q2 @ K2.T / scale)
```

This is analogous to differential amplifiers in electronics — common-mode noise cancels out, signal is amplified. Irrelevant tokens that both maps attend to similarly get cancelled; relevant tokens that the maps disagree on survive.

**λ** is a learnable per-head scalar initialized small (so the subtraction doesn't dominate early training). It grows as the model learns which differences carry signal.

**Cost:** Each head now has TWO sets of Q and K projections (W_q1, W_q2, W_k1, W_k2) — roughly 2x the Q/K parameters. But the head dimension is halved (d_k/2 each), so the total parameter count stays the same.

### Questions to Answer

**Q1:** Standard attention: each head has Q=[d_model, d_k], K=[d_model, d_k], V=[d_model, d_k]. Differential attention: each head has Q1=[d_model, d_k/2], Q2=[d_model, d_k/2], K1=[d_model, d_k/2], K2=[d_model, d_k/2], V=[d_model, d_k]. Show that the total parameter count per head is the same.

**Q2:** λ is initialized to λ_init = 0.8 - 0.6 * exp(-0.3 * layer_idx). At layer 0, λ ≈ 0.2. At layer 23, λ ≈ 0.74. Why is λ smaller in early layers and larger in later layers?

**Q3:** After computing the differential attention map, the paper applies a GroupNorm across heads. Why? What happens to the output distribution without normalization?

**Q4:** The paper shows Differential Attention improves "attention pattern noise cancellation." Give an example of when this would visibly improve model output quality.

### Answers

**A1:**
Standard: Q + K + V = d_model×d_k + d_model×d_k + d_model×d_k = 3d_model×d_k

Differential: (Q1+Q2) + (K1+K2) + V 
= 2×d_model×(d_k/2) + 2×d_model×(d_k/2) + d_model×d_k
= d_model×d_k + d_model×d_k + d_model×d_k = 3d_model×d_k ✓

Same total. The split is purely in how the computation is organized, not in total capacity.

**A2:** Early layers learn low-level features (syntax, local patterns) where most tokens are somewhat relevant — you don't want to aggressively cancel attention. Later layers learn high-level semantics where most context IS noise relative to the few truly relevant tokens — stronger cancellation is beneficial. The layer-dependent initialization lets each layer find its natural level of noise cancellation.

**A3:** After differential subtraction, the attention map can have any mean — if both maps are similar, the difference is near zero (almost no attention anywhere). GroupNorm ensures the output has consistent scale and distribution regardless of how much cancellation occurs. Without it, layers with strong cancellation produce near-zero activations, causing vanishing gradients.

**A4:** Consider a long document QA task. "What is the capital of France?" The relevant sentence is "Paris is the capital of France." Standard attention must learn to suppress the other 500 irrelevant sentences. Differential attention: both maps might attend broadly, but only the specific relevant sentence causes a DIFFERENCE between the two maps — noise cancels and signal amplifies. The model would produce "Paris" more reliably instead of blending in irrelevant context about Lyon, Marseille, etc.

### Code

Modify `src/model/attention.py`, add a new class:

```python
class DifferentialAttention(nn.Module):
    """Differential Attention: cancels attention noise via map subtraction.
    
    Each head computes two attention maps and subtracts them, cancelling
    common-mode noise (irrelevant tokens both maps agree on) and amplifying
    signal (tokens the maps disagree about).
    
    Parameter count matches standard MHA: d_k/2 per differential pair.
    
    Reference: https://arxiv.org/abs/2410.05258
    """
    
    def __init__(self, d_model, n_heads, layer_idx=0, dropout=0.0, 
                 max_seq_len=2048, use_flash=True):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads       # full head dim
        self.d_k_half = self.d_k // 2      # half dim per differential pair
        self.dropout = dropout
        self.use_flash = use_flash
        
        # TWO Q and K projections per head, each d_k/2
        # Combined into single matrices for efficiency:
        # W_q outputs [n_heads * d_k] = [n_heads * 2 * (d_k/2)] — both Q1 and Q2
        self.W_q = nn.Linear(d_model, n_heads * self.d_k,      bias=False)
        self.W_k = nn.Linear(d_model, n_heads * self.d_k,      bias=False)
        self.W_v = nn.Linear(d_model, n_heads * self.d_k,      bias=False)
        self.W_o = nn.Linear(d_model, d_model,                  bias=False)
        
        # λ: learnable subtraction weight, initialized per layer
        # Starts small so early training behaves like standard attention
        lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_idx)
        self.lambda_q1 = nn.Parameter(torch.randn(self.d_k_half) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(self.d_k_half) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(self.d_k_half) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(self.d_k_half) * 0.1)
        self.lambda_init = lambda_init
        
        # GroupNorm for output stabilization (replaces per-head LayerNorm)
        self.norm = nn.GroupNorm(n_heads, d_model)
        
        self.rope_cache = RoPECache(self.d_k_half, max_seq_len)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V — Q and K will be split into two halves
        q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        k = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        v = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        
        # Split Q and K into two halves for differential computation
        q1, q2 = q.chunk(2, dim=-1)   # each [batch, seq, n_heads, d_k/2]
        k1, k2 = k.chunk(2, dim=-1)
        
        # Apply RoPE to each half
        freqs = self.rope_cache.get_freqs(seq_len)
        q1 = apply_rope(q1, freqs)
        q2 = apply_rope(q2, freqs)
        k1 = apply_rope(k1, freqs)
        k2 = apply_rope(k2, freqs)
        
        # Transpose for attention: [batch, n_heads, seq, d_k/2]
        q1 = q1.transpose(1, 2)
        q2 = q2.transpose(1, 2)
        k1 = k1.transpose(1, 2)
        k2 = k2.transpose(1, 2)
        v  = v.transpose(1, 2)
        
        # Compute λ as scalar from learnable vectors (ensures scalar per head)
        # λ = exp(λ_q1 · λ_k1) - exp(λ_q2 · λ_k2) + λ_init
        lam = (
            torch.exp((self.lambda_q1 * self.lambda_k1).sum())
            - torch.exp((self.lambda_q2 * self.lambda_k2).sum())
            + self.lambda_init
        )
        
        # Compute two attention maps and subtract
        scale = self.d_k_half ** -0.5
        
        if self.use_flash:
            # Flash attention doesn't directly support differential maps
            # so we compute manually here
            pass
        
        # Map 1
        scores1 = torch.matmul(q1, k1.transpose(-2, -1)) * scale
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1
        )
        attn1 = F.softmax(scores1 + causal_mask, dim=-1)
        
        # Map 2
        scores2 = torch.matmul(q2, k2.transpose(-2, -1)) * scale
        attn2 = F.softmax(scores2 + causal_mask, dim=-1)
        
        # Differential: subtract map2 from map1 (weighted by λ)
        attn_diff = attn1 - lam * attn2  # [batch, n_heads, seq, seq]
        
        # Apply to V
        attn_output = torch.matmul(attn_diff, v)  # [batch, n_heads, seq, d_k]
        
        # Reshape: [batch, seq, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # GroupNorm: stabilizes output after subtraction
        # GroupNorm expects [batch, channels, ...] so transpose seq and d_model
        attn_output = self.norm(attn_output.transpose(1, 2)).transpose(1, 2)
        
        return self.W_o(attn_output)
```

**To use in TransformerBlock, add a flag:**
```python
# In TransformerBlock.__init__:
if config.use_diff_attn:
    self.attention = DifferentialAttention(
        d_model=config.d_model,
        n_heads=config.n_heads,
        layer_idx=layer_idx,  # TransformerBlock needs to receive layer_idx
        ...
    )
else:
    self.attention = MultiHeadAttention(...)

# In GPT.__init__ — pass layer_idx when building blocks:
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

**"How does Differential Attention compare to multi-head attention in terms of compute?"**
Same parameter count, similar FLOPs. Computing two attention maps instead of one roughly doubles the attention computation. But attention is typically not the bottleneck (the FFN is ~4x larger) — so the wall-clock impact is small. At long context where attention IS the bottleneck, Differential Attention costs ~2x more per head but achieves comparable quality with fewer heads total (the paper shows you can use half the heads for same quality).

**"What happens when λ=0?"**
You recover standard attention (map2 cancelled out). λ=0 is a valid point in the optimization landscape — the model can choose to ignore the differential mechanism for any head. This makes initialization safe: start with small λ and let the model decide how much to use.

**"Is Differential Attention used in production models?"**
As of mid-2026, it's in research phase with strong results on long-context tasks. Microsoft's internal models reportedly use it. Not yet in mainstream open models (LLaMA, Mistral), but gaining traction. For your blog post, it's a strong signal of being current.

---

## Part 5: mHC (Manifold Hyper-Connections)

### Concept

Standard residual: `x = x + F(x)` — one stream of information.

With mHC, you maintain N parallel streams:
```
streams: [n_streams, batch, seq, d_model]
```

Each layer:
1. Aggregates streams (weighted sum) → single input for F
2. Applies F (attention or MLP)
3. Mixes streams through a learned doubly-stochastic matrix A
4. Adds scaled F output to all streams

**Doubly stochastic (via Sinkhorn):** Every row AND column of A sums to 1. This constrains mixing to be a convex combination — signals can't explode or vanish through many layers.

**Why it helps:** In deep networks (24 layers), gradients can vanish or explode through the many residual additions. mHC's constrained mixing matrix acts as a learned normalization of the information flow, making gradient magnitudes more stable and allowing each layer to contribute more meaningfully.

### Questions to Answer

**Q1:** The mixing matrix A starts as the identity (after Sinkhorn normalization of log(I)). What does this mean for the model's behavior at initialization? Why is this a good property?

**Q2:** Why doubly stochastic specifically? Row-stochastic (rows sum to 1) would also prevent explosion. What does the column constraint add?

**Q3:** At inference, you have N=4 streams going through the whole network. At the end, you collapse via mean: `x = streams.mean(dim=0)`. What does this averaging do to the final representation? Is this the right choice?

**Q4:** mHC increases memory usage by N× (you maintain N copies of activations). For N=4 with your 350M model at batch=16, seq=1024: how much extra GPU memory does this cost?

### Answers

**A1:** Identity mixing means each stream maps only to itself — stream 0 goes to output stream 0, stream 1 to stream 1, etc. At init, mHC behaves EXACTLY like standard residual (one independent residual per stream, all initialized the same). The model starts in familiar territory and gradually learns to mix streams as training progresses. If A started randomly, early training would be chaotic.

**A2:** Row-stochastic: each OUTPUT stream is a convex combination of inputs. Prevents output from being larger than the max input — no explosion. But nothing prevents one INPUT stream from dominating ALL output streams (rich-get-richer problem). Column-stochastic: each input stream contributes equally to some output. Together, doubly stochastic ensures balanced information flow in BOTH directions.

**A3:** Mean averaging is appropriate but not the only option. It treats all N streams equally. Alternative: use only stream 0 (the "main" stream that gets most updates) or take a learned weighted sum. Mean works well empirically — the N streams specialize somewhat during training, and averaging combines their specializations. The information is not lost since all streams saw the full network depth.

**A4:** Extra activation memory = N × baseline. For N=4:
- Baseline activations: batch × seq × d_model × bytes = 16 × 1024 × 1024 × 2 = 33.5MB per layer
- With mHC: 4 × 33.5 = 134MB per layer
- For 24 layers: 24 × 134 = ~3.2GB extra
- Total: ~3.2GB additional peak memory

With 80GB A100/H100, this is very manageable. You might need to reduce micro_batch_size from 16 to 12 to be safe.

### Code

`src/model/hyper_connection.py`:

```python
"""Manifold Hyper-Connections (mHC) from DeepSeek.

Replaces the standard single-stream residual x = x + F(x) with N parallel
streams mixed through a learned doubly-stochastic matrix. The Sinkhorn-Knopp
constraint ensures mixing is always a convex combination — signals can't explode
or vanish across many layers.

Reference: DeepSeek-V2 technical report.
"""

import torch
import torch.nn as nn


def sinkhorn(log_A: torch.Tensor, iters: int = 20) -> torch.Tensor:
    """Project matrix to doubly stochastic via Sinkhorn-Knopp normalization.
    
    Alternates between normalizing rows and columns. After enough iterations,
    both rows and columns sum to 1 (doubly stochastic).
    
    Args:
        log_A: Unconstrained log-space matrix [n, n]
        iters: Alternating normalization steps (20 is more than enough)
    
    Returns:
        Doubly stochastic matrix [n, n]
    """
    # Work in probability space — exponentiate to ensure positivity
    A = log_A.exp()
    
    for _ in range(iters):
        A = A / A.sum(dim=-1, keepdim=True)   # normalize rows: each row sums to 1
        A = A / A.sum(dim=-2, keepdim=True)   # normalize cols: each col sums to 1
    
    return A


class HyperConnection(nn.Module):
    """Wraps a sublayer (attention or MLP) with mHC residual streams.
    
    Instead of: x = x + F(x)
    Does:       streams = A @ streams + alpha * F(aggregate(streams))
    
    Where A is a learned doubly-stochastic mixing matrix (via Sinkhorn).
    
    Args:
        d_model: Model dimension
        n_streams: Number of parallel residual streams (4 is standard)
        alpha_init: Initial scale for sublayer output (small for stability)
        sinkhorn_iters: Iterations for doubly-stochastic projection
    """
    
    def __init__(
        self,
        d_model: int,
        n_streams: int = 4,
        alpha_init: float = 0.01,
        sinkhorn_iters: int = 20,
    ):
        super().__init__()
        self.n_streams = n_streams
        self.sinkhorn_iters = sinkhorn_iters
        
        # Mixing matrix — initialized as identity so early training = standard residual
        # log(I) = 0 on diagonal, -inf off diagonal, but we use a soft version
        self.log_A = nn.Parameter(torch.eye(n_streams).log())
        
        # Aggregation weights: how to combine streams into one input for F
        # Uniform init: all streams contribute equally at start
        self.agg_weights = nn.Parameter(torch.ones(n_streams) / n_streams)
        
        # Small initial scale: sublayer output is noisy at init
        # Starts near zero so streams stabilize before F has large effect
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
    
    def forward(self, streams: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """
        Args:
            streams: [n_streams, batch, seq_len, d_model]
            sublayer: Callable (the attention or MLP module)
        
        Returns:
            Updated streams: [n_streams, batch, seq_len, d_model]
        """
        # Step 1: Aggregate streams into a single input for the sublayer
        # Softmax ensures agg_weights are positive and sum to 1
        weights = torch.softmax(self.agg_weights, dim=0)                # [n_streams]
        aggregated = torch.einsum("n,nbsd->bsd", weights, streams)      # [batch, seq, dim]
        
        # Step 2: Apply the sublayer (attention or MLP)
        sublayer_out = sublayer(aggregated)                              # [batch, seq, dim]
        
        # Step 3: Get doubly-stochastic mixing matrix (Sinkhorn projection)
        A = sinkhorn(self.log_A, self.sinkhorn_iters)                   # [n, n]
        
        # Step 4: Mix existing streams (convex combination — bounded, no explosion)
        mixed = torch.einsum("mn,nbsd->mbsd", A, streams)               # [n, batch, seq, dim]
        
        # Step 5: Add scaled sublayer output broadcast across all streams
        mixed = mixed + self.alpha * sublayer_out.unsqueeze(0)          # [n, batch, seq, dim]
        
        return mixed
```

**Integration in TransformerBlock:**

```python
class TransformerBlock(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.config = config
        
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)
        
        if config.use_diff_attn:
            self.attention = DifferentialAttention(
                d_model=config.d_model, n_heads=config.n_heads,
                layer_idx=layer_idx, ...
            )
        else:
            self.attention = MultiHeadAttention(
                d_model=config.d_model, n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads, ...
            )
        
        self.mlp = SwiGLU(config.d_model)
        
        # mHC wraps each sublayer if enabled
        if config.use_mhc:
            self.attn_hc = HyperConnection(config.d_model, config.n_streams)
            self.mlp_hc  = HyperConnection(config.d_model, config.n_streams)
    
    def forward(self, x):
        if self.config.use_mhc:
            # x is [n_streams, batch, seq, dim]
            x = self.attn_hc(x, lambda h: self.attention(self.norm1(h)))
            x = self.mlp_hc(x,  lambda h: self.mlp(self.norm2(h)))
        else:
            # Standard residual
            x = x + self.attention(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x
```

**Integration in GPT.forward:**

```python
def forward(self, input_ids, targets=None):
    x = self.token_embeddings(input_ids)   # [batch, seq, dim]
    
    if self.config.use_mhc:
        # Expand to N streams: each stream starts as a copy of the input
        x = x.unsqueeze(0).expand(self.config.n_streams, -1, -1, -1).clone()
    
    for block in self.blocks:
        x = block(x)
    
    if self.config.use_mhc:
        # Collapse streams back to one by averaging
        x = x.mean(dim=0)   # [batch, seq, dim]
    
    x = self.norm(x)
    logits = self.lm_head(x)
    
    loss = None
    if targets is not None:
        loss = F.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            targets.view(-1),
            ignore_index=-1
        )
    
    return logits, loss
```

**Add to GPTConfig:**
```python
use_mhc: bool = True
n_streams: int = 4
```

### Interview Q&A

**"Does mHC actually help or is it just more computation?"**
Empirically from the DeepSeek paper: mHC improves loss by ~0.05-0.1 at equivalent parameter count. The intuition is sound — the constrained mixing provides better gradient flow than unconstrained residuals. Your ablation (v1 baseline vs v2 with mHC) will show the exact benefit on your data and model size.

**"How does mHC relate to recurrent models?"**
Both maintain state across layers. mHC is like a recurrence in LAYER space — each stream's state propagates through all layers. The key difference: mHC has a fixed number of streams (not token-level recurrence), and the mixing is fully differentiable and parallelizable across the batch. Much more GPU-friendly than actual recurrence.

**"Why not use more than 4 streams?"**
Linear memory cost with streams. 4 gives good quality-memory tradeoff empirically. 8 gives marginally better quality at 2x memory. The paper shows diminishing returns beyond 4 for most architectures.

---

## Part 6: Integrating Everything into train.py

### Updated TrainConfig

```python
@dataclass
class TrainConfig:
    # Data
    data_dir: str = "data/fineweb-edu/train"
    val_dir: str = "data/fineweb-edu/val"
    seq_len: int = 1024

    # Batch
    micro_batch_size: int = 32       # increased from 16 — fits on H100 with mHC
    grad_accum_steps: int = 16       # 32 * 16 * 1024 = 524K tokens, same effective batch

    # Optimizer
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    muon_lr: float = 1.5e-4         # Muon LR: typically 0.5x max_lr
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Schedule
    warmup_steps: int = 1000
    max_steps: int = 15000           # fewer steps needed — Muon is more efficient

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

    # Feature flags
    use_muon: bool = True
    use_mhc: bool = True
    use_diff_attn: bool = True
    use_qk_norm: bool = True
    use_gqa: bool = True
```

### Updated GPTConfig

```python
@dataclass
class GPTConfig:
    vocab_size: int = 50304
    d_model: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 4              # GQA: 4 KV groups
    dropout: float = 0.0
    max_seq_len: int = 1024
    use_flash: bool = True
    tie_weights: bool = True
    # V2 flags
    use_mhc: bool = True
    n_streams: int = 4
    use_diff_attn: bool = True
    use_qk_norm: bool = True
```

### Updated Training Loop

```python
def train(config: TrainConfig):
    # ... setup ...
    
    # Optimizers — two of them now
    from src.optim.muon import configure_optimizers
    muon_opt, adamw_opt = configure_optimizers(
        model, lr=config.max_lr, muon_lr=config.muon_lr,
        weight_decay=config.weight_decay
    )
    
    # EMA
    ema = EMA(model, decay=config.ema_decay) if config.use_ema else None
    
    # Training loop
    model.train()
    for step in range(start_step, config.max_steps):
        
        # LR — only update AdamW (Muon LR is fixed)
        lr = get_lr(step, ...)
        for pg in adamw_opt.param_groups:
            pg["lr"] = lr
        
        # Zero both optimizers
        muon_opt.zero_grad()
        adamw_opt.zero_grad()
        
        # ... gradient accumulation loop (unchanged) ...
        
        # Clip and step BOTH optimizers
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        muon_opt.step()
        adamw_opt.step()
        
        # Update EMA after optimizer step
        if ema:
            ema.update(model)
        
        # Eval uses EMA model
        if step > 0 and step % config.eval_interval == 0:
            eval_model = ema.model if ema else model
            val_loss = evaluate(eval_model, val_dataset, config, device, dtype)
            ...
        
    # Save final checkpoint
    save_checkpoint(model, optimizer=(muon_opt, adamw_opt), step=config.max_steps, config=config)
```

**Update save_checkpoint to handle two optimizers:**
```python
def save_checkpoint(model, optimizer, step, config, ema=None):
    muon_opt, adamw_opt = optimizer
    checkpoint = {
        "model_weights": model.state_dict(),
        "muon_state": muon_opt.state_dict(),
        "adamw_state": adamw_opt.state_dict(),
        "step": step,
        "rng_state": torch.random.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
    }
    if ema:
        checkpoint["ema_weights"] = ema.state_dict()
    torch.save(checkpoint, path)
```

---

## Part 7: SFT (Supervised Fine-Tuning)

### Concept

Pretraining: model learns to predict the next token in raw text. It doesn't know what a "question" is or that it should "answer" it. It just completes text.

SFT: you show the model thousands of examples of the behavior you want:

```
Input:  <|user|>What is photosynthesis?<|assistant|>
Output: Photosynthesis is the process by which plants...
```

The model learns: "when I see `<|assistant|>` after a question, I should produce a helpful answer." This teaches BEHAVIOR, not new knowledge — the knowledge is already in the pretrained weights.

**Key differences from pretraining:**

| | Pretraining | SFT |
|--|--|--|
| Data | Raw text, no structure | Instruction-response pairs |
| Loss | All tokens | Response tokens only |
| LR | 3e-4 | 1e-5 to 5e-5 |
| Epochs | 1 (see each token once) | 2-3 |
| Goal | Learn language | Learn behavior |

### Questions to Answer

**Q1:** During SFT, you mask the prompt tokens (set their loss targets to -100). `F.cross_entropy` with `ignore_index=-100` skips those positions. Why does this matter — what would happen if you computed loss on the prompt too?

**Q2:** Your SFT dataset has examples of varying length. Some are 50 tokens, some are 2000 tokens. How do you handle this in a batch? What are the tradeoffs between padding vs packing?

**Q3:** You train for 2-3 epochs on the SFT dataset (seeing each example 2-3 times). Why not 10 epochs? What does overfitting look like in SFT?

**Q4:** Your pretraining LR was 3e-4. SFT uses 2e-5. Why so much lower? What specific damage does too-high LR cause during SFT?

**Q5:** After SFT, you evaluate the model. List three concrete ways to evaluate whether SFT worked, beyond just measuring loss.

### Answers

**A1:** Computing loss on the prompt makes the model try to predict the user's question given only its own previous output. This is backwards — the user's message is an INPUT, not something the model should generate. It would waste gradient signal on learning to predict user behavior and would interfere with the response generation objective. More critically: the model would try to minimize loss on deterministic prompt tokens (given the full context, the prompt tokens are already "known") which creates a degenerate gradient signal.

**A2:**
- **Padding:** Pad all examples to max_len with a special token. Pros: simple. Cons: wastes compute on padded positions, memory inefficient for batches with mixed lengths.
- **Packing:** Concatenate multiple examples (with separator tokens) to fill max_seq_len exactly. Pros: no wasted compute. Cons: need careful masking so examples don't attend across boundaries.
- **Truncation:** For very long examples, truncate to max_seq_len. Keep the response if possible (truncate the prompt).

For your 350M model with max_seq_len=1024, most SFT examples fit without truncation. Simple padding is fine.

**A3:** Overfitting in SFT shows as:
- Val SFT loss still decreasing but val pretraining loss increasing (forgetting)
- Model starts completing prompt tokens rather than answering (format collapse)
- Responses become repetitive and formulaic (mode collapse)
- Length calibration breaks (always produces maximum length, or always very short)

At 2-3 epochs, you've given the model enough exposure to the format without overfitting to specific examples. 10 epochs would memorize exact responses.

**A4:** Too-high LR during SFT causes CATASTROPHIC FORGETTING — the pretrained weights shift so far that the language understanding from pretraining is damaged. You'd see:
- Loss spikes on held-out pretraining data
- Generated text becomes grammatically broken or incoherent outside the SFT format
- The model "forgets" how to write natural English and only knows SFT format

2e-5 makes small, careful nudges. The model's general capabilities stay intact while the behavior shifts.

**A5:**
1. **Format compliance:** Does the model stop generating at `<|endoftext|>`? Does it produce structured responses instead of open-ended completions?
2. **Human eval (qualitative):** Ask it questions, compare v1 base vs SFT. Does it ANSWER vs just CONTINUE the text?
3. **Held-out task performance:** Test on a set of questions not in the SFT data. Measure accuracy for factual questions, code correctness for code tasks.

### SFT Dataset Class

```python
# src/data/sft_dataset.py
"""SFT dataset: loads instruction-response pairs with loss masking."""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import tiktoken


class SFTDataset(Dataset):
    """Dataset for supervised fine-tuning.
    
    Returns (input_ids, targets, loss_mask) where loss_mask=-100 on
    prompt tokens so the model only learns from response tokens.
    
    Format:
        <|endoftext|>User: {instruction}<|endoftext|>Assistant: {response}<|endoftext|>
    
    Args:
        data_sources: List of (dataset_name, split, weight, instruction_key, response_key)
        max_seq_len: Maximum sequence length (longer examples are truncated)
        enc: tiktoken encoder
    """
    
    def __init__(
        self,
        data_sources: list[dict],
        max_seq_len: int = 1024,
        enc=None,
    ):
        if enc is None:
            enc = tiktoken.get_encoding("gpt2")
        self.enc = enc
        self.max_seq_len = max_seq_len
        self.eot = enc.eot_token  # use as separator
        
        # Load and mix all data sources
        self.examples = []
        for source in data_sources:
            examples = self._load_source(source)
            self.examples.extend(examples)
        
        print(f"SFT dataset: {len(self.examples):,} examples")
    
    def _load_source(self, source: dict) -> list[dict]:
        """Load examples from a HuggingFace dataset."""
        dataset = load_dataset(
            source["name"],
            split=source.get("split", "train"),
            streaming=False,
        )
        
        instruction_key = source.get("instruction_key", "instruction")
        response_key = source.get("response_key", "output")
        max_examples = source.get("max_examples", None)
        
        examples = []
        for i, item in enumerate(dataset):
            if max_examples and i >= max_examples:
                break
            examples.append({
                "instruction": item[instruction_key],
                "response": item[response_key],
            })
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int):
        example = self.examples[idx]
        
        # Format: EOT + "User: " + instruction + EOT + "Assistant: " + response + EOT
        # Using EOT as separator since it's already in the tiktoken GPT-2 vocab
        user_text = f"User: {example['instruction']}"
        asst_text = f"Assistant: {example['response']}"
        
        # Tokenize separately so we know the boundary
        user_tokens  = [self.eot] + self.enc.encode(user_text)  + [self.eot]
        asst_tokens  = self.enc.encode(asst_text) + [self.eot]
        
        # Full sequence: user + assistant
        full_tokens = user_tokens + asst_tokens
        
        # Truncate if too long (keep as much of response as possible)
        if len(full_tokens) > self.max_seq_len + 1:
            full_tokens = full_tokens[:self.max_seq_len + 1]
        
        # input_ids: all but last token
        # targets: all but first token (shifted by 1)
        input_ids = full_tokens[:-1]
        targets   = full_tokens[1:]
        
        # Loss mask: -100 on prompt (user) tokens, keep response tokens
        prompt_len = len(user_tokens) - 1  # -1 because of the shift
        loss_mask  = [-100] * min(prompt_len, len(targets)) + [1] * max(0, len(targets) - prompt_len)
        
        # Pad to max_seq_len
        pad_len = self.max_seq_len - len(input_ids)
        input_ids  = input_ids  + [self.eot] * pad_len
        targets    = targets    + [-100]      * pad_len  # -100 on padding
        loss_mask  = loss_mask  + [-100]      * pad_len
        
        return (
            torch.tensor(input_ids,  dtype=torch.long),
            torch.tensor(targets,    dtype=torch.long),
            torch.tensor(loss_mask,  dtype=torch.long),
        )
```

### SFT Training Script

```python
# sft.py
"""Supervised Fine-Tuning for the 350M LLM."""

import os
import time
import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass
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
    # Base model to start from
    base_checkpoint: str = "checkpoints/step_020000.pt"
    
    # Output
    checkpoint_dir: str = "checkpoints_sft"
    
    # Data — mix of general, math, code
    # Ratios: 60% OpenHermes, 30% MetaMath, 10% CodeAlpaca
    data_sources: list = None  # defined in __post_init__
    
    # Training
    max_seq_len: int = 1024
    batch_size: int = 16
    grad_accum_steps: int = 4      # smaller than pretraining — SFT data is smaller
    num_epochs: int = 3
    
    # Optimizer — much lower LR than pretraining
    lr: float = 2e-5              # 15x lower than pretraining max_lr
    weight_decay: float = 0.01   # lower than pretraining — don't regularize as aggressively
    grad_clip: float = 1.0
    warmup_steps: int = 100       # short warmup — model is already trained
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 200
    save_interval: int = 500
    
    # Hardware
    device: str = "cuda"
    
    def __post_init__(self):
        if self.data_sources is None:
            self.data_sources = [
                {
                    "name": "teknium/OpenHermes-2.5",
                    "split": "train",
                    "instruction_key": "conversations",  # needs special handling
                    "max_examples": 600_000,  # 60%
                },
                {
                    "name": "meta-math/MetaMathQA",
                    "split": "train",
                    "instruction_key": "query",
                    "response_key": "response",
                    "max_examples": 300_000,  # 30%
                },
                {
                    "name": "sahil2801/CodeAlpaca-20k",
                    "split": "train",
                    "instruction_key": "instruction",
                    "response_key": "output",
                    "max_examples": 100_000,  # 10%
                },
            ]


def sft_loss(logits, targets, loss_mask):
    """Cross-entropy loss applied only to non-masked positions."""
    # Apply loss_mask: where mask is -100, set target to -100 (ignored by cross_entropy)
    masked_targets = targets.clone()
    masked_targets[loss_mask == -100] = -100
    
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        masked_targets.view(-1),
        ignore_index=-100
    )


def get_sft_lr(step, warmup_steps, total_steps, lr):
    """Linear warmup + cosine decay for SFT."""
    if step < warmup_steps:
        return lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return lr * (1 + math.cos(math.pi * progress)) / 2


def sft_train(config: SFTConfig):
    """Main SFT training loop."""
    torch.manual_seed(42)
    device = config.device
    enc = tiktoken.get_encoding("gpt2")
    
    # Initialize wandb
    if HAS_WANDB:
        wandb.init(project="llm-350m-sft", config=vars(config))
    
    # Load base model
    model_config = GPTConfig()
    model = GPT(model_config).to(device)
    
    print(f"Loading base model from {config.base_checkpoint}...")
    checkpoint = torch.load(config.base_checkpoint, weights_only=False)
    model.load_state_dict(checkpoint["model_weights"])
    print("✓ Base model loaded")
    
    # Compile for speed
    model = torch.compile(model)
    
    # SFT dataset
    train_dataset = SFTDataset(config.data_sources, config.max_seq_len, enc)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=4,
    )
    
    # Total steps across epochs
    steps_per_epoch = len(train_loader) // config.grad_accum_steps
    total_steps = steps_per_epoch * config.num_epochs
    print(f"✓ Training for {config.num_epochs} epochs, {total_steps} steps")
    
    # Optimizer — single AdamW, no Muon (optional for SFT)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
        fused=True,
    )
    
    # Training loop
    model.train()
    step = 0
    train_iter = iter(train_loader)
    
    for step in range(total_steps):
        t_start = time.time()
        
        # LR schedule
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
            print(
                f"Step {step:>6d} | Loss {running_loss:.4f} | "
                f"LR {lr:.2e} | Grad Norm {grad_norm:.2f} | dt {step_time*1000:.0f}ms"
            )
            if HAS_WANDB:
                wandb.log({"train/loss": running_loss, "train/lr": lr}, step=step)
        
        if step > 0 and step % config.save_interval == 0:
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            path = os.path.join(config.checkpoint_dir, f"sft_step_{step:06d}.pt")
            torch.save({
                "model_weights": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "step": step,
            }, path)
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

### Expected SFT Timeline

| | Value |
|--|--|
| Dataset size | ~1M examples |
| Epochs | 3 |
| Steps | ~3000 (with grad_accum) |
| Time on A100 | ~3-4 hours |
| Cost | ~$5-8 |
| Starting loss | ~2.9 (same as pretraining end) |
| Final SFT loss | ~1.5-2.0 |

### Evaluation After SFT

Test these prompts before and after:

**General:**
```
User: What is the capital of Australia?
Base model:  "...is a country in the Southern Hemisphere. The capital is..."
SFT model:   "The capital of Australia is Canberra."
```

**Math:**
```
User: What is 15% of 80?
Base model:  "...percent of the total..." (continues the text)
SFT model:   "15% of 80 is 12. To calculate: 80 × 0.15 = 12."
```

**Code:**
```
User: Write a Python function to reverse a string.
Base model:  "...has many methods. The reverse() method..."
SFT model:   "def reverse_string(s):\n    return s[::-1]"
```

### Interview Q&A on SFT

**"Why does SFT on 1M examples teach the model to follow instructions, but pretraining on 10B tokens doesn't?"**
Pretraining teaches WHAT language looks like — grammar, facts, reasoning patterns. But the format is always "predict the next token in text." SFT teaches the ROLE — "I am an assistant, my job is to answer questions." The model already has the capability; SFT teaches it WHEN to use which capability.

**"Could you do RLHF instead of SFT?"**
RLHF (via PPO or GRPO) is typically applied AFTER SFT, not instead of it. The RL training needs a well-behaved starting point — the SFT model — to generate responses that are at least in the right ballpark for the reward model to score. Starting RL from a raw pretrained model produces garbage that's hard to reward-signal into coherent behavior.

**"How do you prevent the model from only knowing how to answer questions in the SFT format?"**
Mixing the SFT dataset with a small amount of pretraining data (5-10%) maintains general language capabilities. This is sometimes called "replay" — you're replaying pretraining samples to prevent forgetting. You can also use a low enough LR (2e-5) that pretraining knowledge isn't overwritten, just supplemented.

---

## Summary: V2 Implementation Order

```
Week 1 (local, no GPU):
├── EMA class (20 min)
├── QK-Norm in attention.py (30 min)
├── GQA in attention.py (1 hour)
├── Differential Attention (2-3 hours)
├── mHC in hyper_connection.py (2 hours)
├── Update GPTConfig with all flags (30 min)
├── Update train.py (Muon, EMA, flag-gated blocks) (2 hours)
└── Smoke test locally on tiny model

Week 1-2 (GPU):
├── 500-step validation run on A100 (~$3)
│   └── Verify loss drops, no crashes, wandb shows correct metrics
├── Full V2 training run on H100 SXM (~$50)
│   └── 15K steps, compare to V1 on wandb
└── SFT V1 on A100 (~$8)

Week 2:
├── Analyze V1 vs V2 loss curves
├── Write blog post sections
└── Set up serving VPC
```

### Interview Q: "What's the biggest risk in V2?"

The interaction between Differential Attention and mHC. Both change how information flows through the network — DiffAttn changes what attention sees, mHC changes how residuals combine. They've never been tested together. The risk is: they might interfere, causing slower convergence or instability. Mitigation: run a 500-step validation first, watch grad norms carefully. If grad norms spike beyond 2.0 in the first 100 steps, disable one of them and debug.

### Portfolio Story

After V2 completes, your story is:

> "I built a 350M LLM from scratch, implementing every component: byte-pair tokenization, rotary position embeddings, SwiGLU activation, flash attention. Then I trained two versions: a baseline following standard modern architecture, and an improved version incorporating Muon optimization (better gradient geometry), mHC residuals (DeepSeek's manifold-constrained multi-stream residuals), Differential Attention (noise-cancelling attention maps), GQA (serving-efficient KV cache reduction), and EMA (smoother inference). The V2 model achieved lower validation loss in fewer steps. Both were SFT'd on instruction data. I served them on a VPC with FastAPI."

That's a complete, credible, technically deep portfolio piece that almost no candidates can match.
