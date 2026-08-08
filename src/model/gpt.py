import torch
import torch.nn as nn
import torch.nn.functional as F
from .block import TransformerBlock
from .rmsnorm import RMSNorm
from .kv_cache import KVCache

class GPTConfig:
    """Configuration for GPT model"""
    def __init__(
            self,
            vocab_size: int = 50304,
            d_model: int = 1024,
            n_layers: int = 24,
            n_heads: int = 16,
            n_kv_heads: int = 4,
            dropout: float = 0.0,
            max_seq_len: int = 1024,
            use_flash: bool = True,
            tie_weights: bool = True,
            use_qk_norm: bool = True,
            use_diff_attn: bool = True,
            use_mhc: bool = True,
            n_streams: int = 2,
            mhc_every_n_layers: int = 1,
            use_xsa: bool = False,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.use_flash = use_flash
        self.tie_weights = tie_weights
        self.use_qk_norm = use_qk_norm
        self.use_diff_attn = use_diff_attn
        self.use_mhc = use_mhc
        self.n_streams = n_streams
        self.mhc_every_n_layers = mhc_every_n_layers
        self.use_xsa = use_xsa
        
class GPT(nn.Module):
    """
    GPT decoder-only transformer with modern architecture.
    
    Components:
    - Token embeddings (no positional embeddings - RoPE handles this)
    - Stack of transformer blocks
    - Final RMSNorm
    - Language modeling head
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model) # [vocab_size, d_model]

        # Dropout after embeddings, if specified
        self.emb_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None # [batch, seq_len, d_model]

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                layer_idx=i,
                dropout=config.dropout,
                max_seq_len=config.max_seq_len,
                use_flash=config.use_flash,
                use_qk_norm=config.use_qk_norm,
                use_diff_attn=config.use_diff_attn, 
                use_xsa=config.use_xsa,
                use_mhc=config.use_mhc and (i % config.mhc_every_n_layers == 0),
                n_streams = config.n_streams,
            )
            for i in range(config.n_layers)
        ]) # List of [TransformerBlock] of length n_layers, each block processes [batch, seq_len, d_model]

        # Final RMSNorm
        self.norm = RMSNorm(config.d_model) # [d_model]

        # Language modeling head (projects to vocabulary)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False) # [d_model, vocab_size]

        # Optionally tie weights of token embeddings and language modeling head
        if config.tie_weights:
            self.lm_head.weight = self.token_embeddings.weight

        if config.use_mhc:
            read_logits = torch.full((config.n_streams,), -2.0)
            read_logits[0] = 2.0
            self.final_read_logits = nn.Parameter(read_logits)
        else:
            self.final_read_logits = None


        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize weights for better training.
        
        Standard practice for transformers:
        - Linear layers: normal distribution with std=0.02
        - Embeddings: normal distribution with std=0.02
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
            self,
            input_ids: torch.Tensor,
            targets: torch.Tensor = None,
            cache: torch.Tensor = None,
            use_cache: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            targets: Target token IDs [batch, seq_len] (optional, for training)
        
        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
            loss: Cross-entropy loss (if targets provided), else None
        """
        # Step 1 - Get token embeddings
        x = self.token_embeddings(input_ids) # [batch, seq_len, d_model]

        # Step 2 - Apply dropout to embeddings, if specified
        if self.emb_dropout is not None:
            x = self.emb_dropout(x) # [batch, seq_len, d_model]

        # Step 3 - Pass through transformer blocks - or mHC streams
        if self.config.use_mhc:
            # Initialize S streams as copies of x: [S, B, T, D]
            streams = x.unsqueeze(0).repeat(self.config.n_streams, 1, 1, 1)
            
            for block in self.blocks:
                streams, _ = block(streams)
            
            # Learned final readout over streams
            read_weights = F.softmax(self.final_read_logits, dim=0)
            x = torch.einsum('s,sbtd->btd', read_weights, streams)

        else:
            for layer_idx, block in enumerate(self.blocks):
                # Pull this layer's cached KV out of the cache, if any
                if cache is not None and cache.k_cache[layer_idx] is not None:
                    past_kv = (cache.k_cache[layer_idx], cache.v_cache[layer_idx])
                else:
                    past_kv = None
                x, new_kv = block(x, past_kv=past_kv, use_cache=use_cache) # [batch, seq_len, d_model]

                # attention already concatenated past+new, so new_kv is the full
                # cache for this layer. Write it straight back.
                if use_cache and cache is not None:
                    cache.k_cache[layer_idx] = new_kv[0]
                    cache.v_cache[layer_idx] = new_kv[1]

            # Bump length once per step, after all the layers, by the new token count
            if use_cache and cache is not None:
                cache.length += input_ids.shape[1]

        # Step 4 - Final RMSNorm
        x = self.norm(x) # [batch, seq_len, d_model]

        # Step 5 - Language modeling head to get logits
        logits = self.lm_head(x) # [batch, seq_len, vocab_size]

        # Step 6 - If targets provided, compute cross-entropy loss
        loss = None
        if targets is not None:
            # Flatten for cross-entropy
            # logits: [batch*seq_len, vocab_size], targets: [batch*seq_len]
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
                ignore_index=-100 # Ignore padding tokens if any
            )

        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None
    ) -> torch.Tensor:
        """
        Generate new tokens autoregressively.
        
        Args:
            input_ids: Starting tokens [batch, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens
        
        Returns:
            Generated sequence [batch, seq_len + max_new_tokens]
        """
        self.eval() # Set to eval mode for generation (disables dropout)
        for _ in range(max_new_tokens):
            # Crop input_ids if longer than max_seq_len
            input_ids_crop = input_ids[:, -self.config.max_seq_len:]
            # Get logits for current input
            logits, _ = self(input_ids_crop) # [batch, seq_len_crop, vocab_size]
            # We only care about the last token's logits for sampling
            logits = logits[:, -1, :]  # [batch, vocab_size]
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            # Apply top-k filtering if specified
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            # Sample from the distribution
            probs = F.softmax(logits, dim=-1) # [batch, vocab_size]
            next_token = torch.multinomial(probs, num_samples=1) # [batch, 1]
            # Append to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=-1) # [batch, seq_len + 1]
        return input_ids


    @torch.no_grad()
    def generate_cached(
        self,
        input_ids: torch.Tensor,       # [batch, prompt_len]
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation using the KV cache.

        Two phases:
        - prefill: run the whole prompt once, filling the cache.
        - decode:  feed one token at a time; the cache supplies the past.

        Returns the full sequence [batch, prompt_len + max_new_tokens].
        """
        self.eval()

        cache = KVCache(self.config.n_layers)

        # --- helper: sample one next token from a [batch, vocab] logits row ---
        def sample_next(logits):
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            # Sample from the distribution
            probs = F.softmax(logits, dim=-1) # [batch, vocab_size]
            next_token = torch.multinomial(probs, num_samples=1) # [batch, 1]

            return next_token

        logits, _ = self(input_ids=input_ids, cache=cache, use_cache=True)
        next_token = sample_next(logits[:, -1, :])   # [batch, 1]

        generated = [next_token]

        for _ in range(max_new_tokens - 1):
            logits, _ = self(input_ids=next_token, cache=cache, use_cache=True) # feed next_token, not input_ids
            next_token = sample_next(logits[:, -1, :])
            generated.append(next_token)

        generated = torch.cat(generated, dim=1)          # list of [B,1] -> [B, max_new_tokens]
        result = torch.cat([input_ids, generated], dim=1) # [B, prompt_len + max_new_tokens]
        return result

    @torch.no_grad()
    def generate_stream(
        self,
        input_ids: torch.Tensor, # [1, prompt_len] (streaming assumes batch=1)
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token: int | None = None,
    ):
        """
        Streaming generation: yield one token id at a time as it's 
        produced (versus all response in one go).

        Same prefill/decode as generate_cached(), but instead of collecting 
        tokens and returning them at the end, it yields each token the moment
        it's sampled, so a caller (eg. API) can send it to user immediately.

        Yields:
            int: the next token_id, one per decode step.
        """

        self.eval()
        # One cache for this whole generation. Persists across every step below.
        # Holds K/V for all n_layers. Created here, lives until the generator ends.
    
        cache = KVCache(self.config.n_layers)

        def sample_next(logits):
            # identical to generate_cached()
            # logits in: [1, vocab_size]  (already the last position's row) - would be [batch, vocab_size] if not streaming
            logits = logits / temperature # [1, vocab] — sharpen/flatten the distribution
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1))) # v: [1, top_k], the top_k largest logits
                logits[logits < v[:, [-1]]] = -float('inf') # mask everything below the k-th largest to -inf

            # Sample from the distribution 
            probs = F.softmax(logits, dim=-1) # [1, vocab_size] — now a probability distribution
            return torch.multinomial(probs, num_samples=1) # [1, 1] — one sampled token id

        # PREFILL
        # input_ids: [1, prompt_len]. One parallel pass over the whole prompt.
        # Fills the cache with the prompt's K/V. logits: [1, prompt_len, vocab].    
        logits, _ = self(input_ids=input_ids, cache=cache, use_cache=True)
        # logits[:, -1, :] is [1, vocab]: the prediction AFTER the last prompt token,
        # i.e. the first token of the answer.
        next_token = sample_next(logits[:, -1, :]) # [1, 1]
        # .item() pulls the scalar out of the [1,1] tensor -> plain Python int.
        # The caller detokenizes and sends this to the user immediately.
        yield next_token.item() # int

        if eos_token is not None and next_token.item() == eos_token:
            return # ends the generator cleanly

        # DECODE
        # max_new_tokens - 1 because prefill already produced one token above.
        for _ in range(max_new_tokens - 1):
            # Feed ONLY the new token: [1, 1], not the prompt.
            # The cache supplies all the past; the model does one token's work.
            # logits: [1, 1, vocab] (seq_len is 1 now).
            logits, _ = self(input_ids=next_token, cache=cache, use_cache=True)
            # [:, -1, :] is [1, vocab] again (the only position). Sample the next token.
            next_token = sample_next(logits[:, -1, :])

            yield next_token.item()

            if eos_token is not None and next_token.item() == eos_token:
                return # ends the generator cleanly

    
    def count_parameters(self) -> dict:
        """Count parameters in different components."""
        def count_params(module):
            return sum(p.numel() for p in module.parameters())
        
        counts = {
            'embeddings': count_params(self.token_embeddings),
            'blocks': sum(count_params(block) for block in self.blocks),
            'final_norm': count_params(self.norm),
            'lm_head': 0 if self.config.tie_weights else count_params(self.lm_head),
            'total': sum(p.numel() for p in self.parameters())
        }
        
        return counts


# ============================================================================
# TESTS
# ============================================================================

def test_gpt_model():
    """Test GPT model implementation."""
    print("="*60)
    print("Testing GPT Model")
    print("="*60)
    
    # Small config for testing
    config = GPTConfig(
        vocab_size=1000,
        d_model=256,
        n_layers=4,
        n_heads=8,
        dropout=0.1,
        max_seq_len=128,
        use_flash=False,
        tie_weights=True
    )
    
    print("\n1. Testing model initialization...")
    model = GPT(config)
    print("   ✓ Model created successfully")
    
    print("\n2. Testing forward pass...")
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    logits, loss = model(input_ids)
    
    assert logits.shape == (batch_size, seq_len, config.vocab_size), \
        f"Expected shape {(batch_size, seq_len, config.vocab_size)}, got {logits.shape}"
    assert loss is None, "Loss should be None when targets not provided"
    print("   ✓ Forward pass works")
    print(f"   ✓ Output shape: {logits.shape}")
    
    print("\n3. Testing training mode (with loss)...")
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    logits, loss = model(input_ids, targets)
    
    assert loss is not None, "Loss should be computed when targets provided"
    assert loss.item() > 0, "Loss should be positive"
    print(f"   ✓ Loss computed: {loss.item():.4f}")
    
    print("\n4. Testing backward pass...")
    loss.backward()
    
    # Check that gradients exist for key parameters
    assert model.token_embeddings.weight.grad is not None
    assert not torch.isnan(model.token_embeddings.weight.grad).any()
    print("   ✓ Gradients flow correctly")
    
    print("\n5. Testing weight tying...")
    if config.tie_weights:
        assert model.token_embeddings.weight is model.lm_head.weight, \
            "Weights should be tied (same tensor)"
        print("   ✓ Weight tying verified")
    
    print("\n6. Testing generation...")
    model.eval()
    start_tokens = torch.randint(0, config.vocab_size, (1, 5))
    generated = model.generate(start_tokens, max_new_tokens=10, temperature=1.0)
    
    assert generated.shape == (1, 15), f"Expected (1, 15), got {generated.shape}"
    print(f"   ✓ Generation works: {generated.shape}")
    
    print("\n7. Testing parameter count...")
    param_counts = model.count_parameters()
    print(f"   Total parameters: {param_counts['total']:,}")
    print(f"   - Embeddings: {param_counts['embeddings']:,}")
    print(f"   - Transformer blocks: {param_counts['blocks']:,}")
    print(f"   - Final norm: {param_counts['final_norm']:,}")
    print(f"   - LM head: {param_counts['lm_head']:,} (tied: {config.tie_weights})")
    
    print("\n8. Testing different sequence lengths...")
    for test_seq_len in [5, 20, 50]:
        input_test = torch.randint(0, config.vocab_size, (1, test_seq_len))
        logits_test, _ = model(input_test)
        assert logits_test.shape == (1, test_seq_len, config.vocab_size)
    print("   ✓ Works with variable sequence lengths")
    
    print("\n" + "="*60)
    print("GPT Model Tests Passed! ✓")
    print("="*60)


def test_full_350m_model():
    """Test creating a full 350M parameter model."""
    print("\n" + "="*60)
    print("Testing Full 350M Model Configuration")
    print("="*60)
    
    # Full 350M config
    config = GPTConfig(
    vocab_size=50304,
    d_model=1024, n_layers=24, n_heads=16,
    dropout=0.0, max_seq_len=1024, use_flash=True, tie_weights=True,
    )   
    
    print("\n1. Creating 350M model...")
    model = GPT(config)
    print("   ✓ Model created")
    
    print("\n2. Counting parameters...")
    param_counts = model.count_parameters()
    total_params = param_counts['total']
    print(f"   Total parameters: {total_params:,}")
    print(f"   Target: ~315,000,000")
    
    # Check if we're in the right ballpark (340M - 360M)
    assert 310_000_000 < total_params < 325_000_000, \
        f"Expected ~350M params, got {total_params:,}"
    print("   ✓ Parameter count in target range")
    
    print("\n3. Testing forward pass on small batch...")
    # Small batch to test (don't want to OOM)
    input_ids = torch.randint(0, config.vocab_size, (1, 128))
    logits, _ = model(input_ids)
    
    assert logits.shape == (1, 128, config.vocab_size)
    print("   ✓ Forward pass works")
    
    print("\n4. Model breakdown:")
    print(f"   - Embeddings: {param_counts['embeddings']:,} " +
          f"({param_counts['embeddings']/total_params*100:.1f}%)")
    print(f"   - Blocks: {param_counts['blocks']:,} " +
          f"({param_counts['blocks']/total_params*100:.1f}%)")
    print(f"   - Other: {param_counts['final_norm']:,}")
    
    print("\n" + "="*60)
    print("350M Model Tests Passed! ✓")
    print("="*60)

def test_gpt_mhc_model():
    config = GPTConfig(
        vocab_size=1000,
        d_model=256,
        n_layers=4,
        n_heads=8,
        n_kv_heads=2,
        dropout=0.0,
        max_seq_len=128,
        use_flash=True,
        tie_weights=True,
        use_qk_norm=True,
        use_diff_attn=True,
        use_mhc=True,
        n_streams=2,
    )

    model = GPT(config)

    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    targets = torch.randint(0, config.vocab_size, (2, 16))

    logits, loss = model(input_ids, targets)

    assert logits.shape == (2, 16, config.vocab_size)
    assert loss is not None
    assert torch.isfinite(loss)

    loss.backward()

    print("✓ GPT with mHC works")

def test_xsa_propagates():
    """Ensure use_xsa actually reaches the attention modules (GPT->Block->Attn)."""
    from .attention import MultiHeadAttention
    config = GPTConfig(
        vocab_size=1000, d_model=256, n_layers=4, n_heads=8, n_kv_heads=2,
        dropout=0.0, max_seq_len=128, use_flash=True, tie_weights=True,
        use_qk_norm=True, use_diff_attn=False, use_xsa=True, use_mhc=False,
    )
    model = GPT(config)
    xsa_count = 0
    for block in model.blocks:
        attn = block.attention
        assert isinstance(attn, MultiHeadAttention), f"expected MHA, got {type(attn)}"
        assert attn.use_xsa is True, "use_xsa did NOT propagate to attention module!"
        xsa_count += 1
    assert xsa_count == config.n_layers
    print(f"✓ use_xsa propagates to all {xsa_count} attention modules")

@torch.no_grad()
def test_cache_equivalence(seq_len=16, tol=1e-4):
    """Full-sequence forward vs token-by-token cached forward: logits must match."""
    torch.manual_seed(0)
    config = GPTConfig(
        vocab_size=1000, d_model=256, n_layers=4, n_heads=8, n_kv_heads=2,
        dropout=0.0, max_seq_len=128, use_flash=True, tie_weights=True,
        use_qk_norm=True, use_diff_attn=False, use_xsa=True, use_mhc=False,
    )
    model = GPT(config)
    model.eval()

    ids = torch.randint(0, config.vocab_size, (1, seq_len))

    # Path A: whole sequence in one pass, no cache
    full_logits, _ = model(ids)                      # [1, seq_len, vocab]

    # Path B: prefill first token, then decode the rest one at a time
    cache = KVCache(config.n_layers)
    step, _ = model(ids[:, :1], cache=cache, use_cache=True)   # prefill 1 token
    collected = [step[:, -1, :]]
    for t in range(1, seq_len):
        step, _ = model(ids[:, t:t+1], cache=cache, use_cache=True)  # decode 1
        collected.append(step[:, -1, :])
    cached_logits = torch.stack(collected, dim=1)    # [1, seq_len, vocab]

    max_diff = (full_logits - cached_logits).abs().max().item()
    print(f"max diff: {max_diff:.2e}")
    assert max_diff < tol, f"CACHE MISMATCH: {max_diff:.2e}"
    print("✓ cache equivalence passed")

@torch.no_grad()
def test_cache_equivalence_diffattn(seq_len=16, tol=1e-4):
    """Same gate as the XSA test, but for the DiffAttn path."""
    torch.manual_seed(0)
    config = GPTConfig(
        vocab_size=1000, d_model=256, n_layers=4, n_heads=8, n_kv_heads=2,
        dropout=0.0, max_seq_len=128, use_flash=True, tie_weights=True,
        use_qk_norm=True, use_diff_attn=True, use_xsa=False, use_mhc=False,
    )
    model = GPT(config)
    model.eval()

    ids = torch.randint(0, config.vocab_size, (1, seq_len))

    full_logits, _ = model(ids)                       # path A: full forward

    cache = KVCache(config.n_layers)                  # path B: prefill + decode
    step, _ = model(ids[:, :1], cache=cache, use_cache=True)
    collected = [step[:, -1, :]]
    for t in range(1, seq_len):
        step, _ = model(ids[:, t:t+1], cache=cache, use_cache=True)
        collected.append(step[:, -1, :])
    cached_logits = torch.stack(collected, dim=1)

    max_diff = (full_logits - cached_logits).abs().max().item()
    print(f"diffattn max diff: {max_diff:.2e}")
    assert max_diff < tol, f"DIFFATTN CACHE MISMATCH: {max_diff:.2e}"
    print("✓ diffattn cache equivalence passed")

def repl_test():
    torch.manual_seed(0)
    config = GPTConfig(
        vocab_size=1000, d_model=256, n_layers=4, n_heads=8, n_kv_heads=2,
        dropout=0.0, max_seq_len=128, use_flash=True, tie_weights=True,
        use_qk_norm=True, use_diff_attn=False, use_xsa=True, use_mhc=False,
    )
    model = GPT(config)
    model.eval()
    ids = torch.randint(0, config.vocab_size, (1, 5))

    # 1. count: max_new_tokens=10 should yield exactly 10 (no eos set)
    toks = list(model.generate_stream(ids, max_new_tokens=10))
    assert len(toks) == 10, f"expected 10 tokens, got {len(toks)}"
    assert all(isinstance(t, int) for t in toks), "tokens must be plain ints"
    assert all(0 <= t < config.vocab_size for t in toks), "token id out of range"
    print(f"✓ streamed {len(toks)} tokens, all valid ids: {toks}")

    # 2. eos stops early: force eos to be the very next token the model will pick
    #    by seeding identically and reading what token 1 is, then asserting the
    #    stream stops at it.
    torch.manual_seed(0)
    first = next(model.generate_stream(ids, max_new_tokens=10))
    print(first)
    torch.manual_seed(0)
    stopped = list(model.generate_stream(ids, max_new_tokens=10, eos_token=first))
    print(stopped)
    assert len(stopped) == 1, f"eos should stop after 1 token, got {len(stopped)}"
    print(f"✓ eos stops early: yielded {stopped}")

    # 3. streaming matches generate_cached exactly (same seed => same tokens)
    torch.manual_seed(0)
    stream_toks = list(model.generate_stream(ids, max_new_tokens=8))
    torch.manual_seed(0)
    cached = model.generate_cached(ids, max_new_tokens=8)
    cached_toks = cached[0, ids.shape[1]:].tolist()   # drop the prompt
    assert stream_toks == cached_toks, f"stream {stream_toks} != cached {cached_toks}"
    print("✓ streaming matches generate_cached token-for-token")


def run_all_tests():
    """Run all GPT tests."""
    print("\n" + "="*70)
    print(" "*20 + "GPT MODEL TESTS")
    print("="*70 + "\n")
    
    test_gpt_model()
    test_full_350m_model()
    test_gpt_mhc_model()
    test_xsa_propagates()
    
    print("\n" + "="*70)
    print(" "*15 + "ALL GPT MODEL TESTS PASSED! 🎉")
    print("="*70)
    print("\nYou now have a complete modern transformer implementation!")
    print("Ready to move on to training infrastructure!")


if __name__ == "__main__":
    #run_all_tests()
    test_cache_equivalence()
    test_cache_equivalence_diffattn()
    #repl_test()