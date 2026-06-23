"""lm-evaluation-harness adapter for the custom GPT model (Path A: thin adapter).

Wraps the raw nn.Module GPT so EleutherAI's lm-evaluation-harness can drive it.
Subclass lm_eval.api.model.LM and implement the three request methods.

Usage:
    import lm_eval
    from src.eval.harness_adapter import CustomGPTLM, build_eval_model

    model = build_eval_model(
        "/workspace/checkpoints_grpo_v1/percentage/grpo_final_percentage.pt",
        device="cuda",
    )
    lm = CustomGPTLM(model=model, device="cuda", batch_size=8)
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=["hellaswag", "lambada_openai", "arc_easy", "winogrande", "piqa"],
        num_fewshot=0,
    )
    print(results["results"])

Install: pip install lm-eval
"""

import torch
import torch.nn.functional as F
import tiktoken

from lm_eval.api.model import LM
from lm_eval.api.instance import Instance

from src.model.gpt import GPT, GPTConfig


# ---------------------------------------------------------------------------
# TODO 1 — model loader
# ---------------------------------------------------------------------------

def build_eval_model(
    checkpoint_path: str,
    device: str = "cuda",
    # architecture: defaults are V1 (no GQA, flags off). Override for V2/V3.
    vocab_size: int = 50304,
    d_model: int = 1024,
    n_layers: int = 24,
    n_heads: int = 16,
    n_kv_heads: int = 16,
    max_seq_len: int = 1024,
    use_qk_norm: bool = False,
    use_diff_attn: bool = False,
    use_ema: bool = False,
    use_mhc: bool = False,
    n_streams: int = 1,
    mhc_every_n_layers: int =1,
) -> GPT:
    """Load a checkpoint into a GPT for evaluation (eval mode, no grad).

    TODOs:
        - build GPTConfig from the arch args (dropout=0.0, use_flash=True,
          tie_weights=True). The defaults above are V1; pass V2/V3 values
          when evaluating those models (e.g. n_kv_heads=4, use_qk_norm=True,
          use_diff_attn=True for V2).
        - model = GPT(config)
        - load checkpoint: torch.load(weights_only=False, map_location=device);
          pick state key "model_weights" if present else "ema";
          strip "_orig_mod." prefix from keys (torch.compile artifact)
        - model.load_state_dict(...), model.eval(), model.to(device)
        - return model
    """
    config = GPTConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        dropout=0.0,
        max_seq_len=max_seq_len,
        use_flash=True,
        tie_weights=True,
        use_qk_norm=use_qk_norm,
        use_diff_attn=use_diff_attn,
        use_mhc=use_mhc,
        n_streams=n_streams,
        mhc_every_n_layers=mhc_every_n_layers,
    )

    model = GPT(config)
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    if use_ema and "ema" in ckpt:
        state_key = "ema"
    elif "model_weights" in ckpt:
        state_key = "model_weights"
    elif "ema" in ckpt:
        state_key = "ema"
    else:
        raise KeyError("Checkpoint has neither 'model_weights' nor 'ema'")
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt[state_key].items()}
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model

# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------

class CustomGPTLM(LM):
    """Exposes the custom GPT to lm-evaluation-harness."""

    def __init__(self, model: GPT, device: str = "cuda", batch_size: int = 8,
                 max_length: int = 1024):
        super().__init__()
        self.model = model
        self.device = device
        self._batch_size = batch_size
        self.max_length = max_length
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot_token = self.enc.eot_token

    @property
    def batch_size(self):
        return self._batch_size

    # -----------------------------------------------------------------------
    # Joint tokenization (the H4 boundary fix)
    # -----------------------------------------------------------------------

    def _encode_pair(self, context: str, continuation: str) -> tuple[list[int], list[int]]:
        """Tokenize context+continuation JOINTLY, then split.

        TODOs (the H4 fix — respects BPE merges at the boundary):
            - context_enc = self.enc.encode(context)
            - whole_enc   = self.enc.encode(context + continuation)
            - continuation_enc = whole_enc[len(context_enc):]
            - return (context_enc, continuation_enc)
        Edge case: empty context -> context_enc == [] -> whole thing is the
        continuation (used by loglikelihood_rolling).
        """
        context_enc = self.enc.encode(context)
        whole_enc = self.enc.encode(context + continuation)
        continuation_enc = whole_enc[len(context_enc):]

        return (context_enc, continuation_enc)

    # -----------------------------------------------------------------------
    # Loglikelihood
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """Per request, return (sum_continuation_logprob, is_greedy).

        Each request.args == (context_str, continuation_str).
        """

        logp_requests = []

        for request in requests:
            context_str, continuation_str = request.args
            context_enc, continuation_enc = self._encode_pair(context_str, continuation_str)

            if len(continuation_enc) == 0:
                logp_requests.append((0.0, True))
                continue

            # For ordinary loglikelihood, give empty-context continuations an EOT prefix.
            if len(context_enc) == 0:
                context_enc = [self.eot_token]

            # Truncate context only, preserving continuation if possible.
            if len(context_enc) + len(continuation_enc) > self.max_length:
                max_context_len = self.max_length - len(continuation_enc)

                if max_context_len <= 0:
                    continuation_enc = continuation_enc[-(self.max_length - 1):]
                    context_enc = [self.eot_token]
                else:
                    context_enc = context_enc[-max_context_len:]

            input_tokens = context_enc + continuation_enc

            input_ids = torch.tensor(
                input_tokens,
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0)

            logits, _ = self.model(input_ids)
            logits = logits.squeeze(0)

            log_probs_all = F.log_softmax(logits[:-1, :], dim=-1)

            cont_log_probs = log_probs_all[-len(continuation_enc):]

            targets = torch.tensor(
                continuation_enc,
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(-1)

            token_logprobs = cont_log_probs.gather(dim=-1, index=targets).squeeze(-1)
            total_logprob = float(token_logprobs.sum().item())

            greedy_tokens = cont_log_probs.argmax(dim=-1)
            actual_tokens = torch.tensor(continuation_enc, dtype=torch.long, device=self.device)

            is_greedy = bool((greedy_tokens == actual_tokens).all())

            logp_requests.append((total_logprob, is_greedy))

        return logp_requests

    # -----------------------------------------------------------------------
    # Loglikelihood_rolling
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float]]:
        """Total log-prob of a whole string for perplexity.

        We score tokens[1:] from tokens[:-1]. Token 0 is not scored because
        there is no preceding context token.

        For long strings, we split into windows of length <= max_length with
        one-token overlap, so each chunk can score its next tokens.
        """
        results = []

        for request in requests:
            (text,) = request.args

            tokens = self.enc.encode(text)

            # Nothing to score if fewer than 2 tokens.
            if len(tokens) < 2:
                results.append((0.0,))
                continue

            total_logprob = 0.0

            # Chunk with one-token overlap.
            # Example:
            #   chunk 1: tokens[0:1024] scores tokens[1:1024]
            #   chunk 2: tokens[1023:2047] scores tokens[1024:2047]
            start = 0

            while start < len(tokens) - 1:
                end = min(start + self.max_length, len(tokens))
                chunk = tokens[start:end]

                if len(chunk) < 2:
                    break

                input_ids = torch.tensor(
                    chunk,
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0)  # [1, T]

                logits, _ = self.model(input_ids)  # [1, T, vocab]
                logits = logits.squeeze(0)         # [T, vocab]

                # logits[t] predicts chunk[t + 1]
                log_probs = F.log_softmax(logits[:-1, :], dim=-1)  # [T-1, vocab]

                targets = torch.tensor(
                    chunk[1:],
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(-1)  # [T-1, 1]

                token_logprobs = log_probs.gather(dim=-1, index=targets).squeeze(-1)
                total_logprob += float(token_logprobs.sum().item())

                if end == len(tokens):
                    break

                # One-token overlap: next chunk starts with the previous final token,
                # so its first scored token has context.
                start = end - 1

            results.append((total_logprob,))

        return results
        
    # -----------------------------------------------------------------------
    # Generate_until
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def generate_until(self, requests: list[Instance]) -> list[str]:
        """Generate text until a stop string (for GSM8K etc.).

        Each request.args == (context_str, gen_kwargs_dict).
        gen_kwargs may have: "until" (list[str] stop seqs), "max_gen_toks" (int),
        "temperature", etc.

        TODOs (per request; reuse your GRPO generation loop):
            1. ctx_enc = self.enc.encode(context), truncate to max_length.
            2. stops = gen_kwargs.get("until", []); max_gen = gen_kwargs.get(
               "max_gen_toks", 256); temp = gen_kwargs.get("temperature", 0.0).
            3. autoregressive loop:
               - forward, take last-position logits, mask padding ids
                 (>= enc.n_vocab) to -inf
               - greedy (argmax) if temp == 0 else sample at temp
               - append token; DECODE the generated-so-far ids to a string and
                 check if any stop string appears (H3); if so, cut at the
                 earliest stop and break. Also stop at max_gen or EOT.
            4. append the generated string (continuation only, cut at first stop).
        """
        completions = []
        
        for request in requests:
            context_string, gen_kwargs_dict = request.args

            if gen_kwargs_dict is None:
                gen_kwargs_dict = {}

            # Encode context and keep the rightmost context tokens.
            context_enc = self.enc.encode(context_string)
            context_enc = context_enc[-self.max_length:]

            # Empty prompt edge case: give model an EOT prefix.
            if len(context_enc) == 0:
                context_enc = [self.eot_token]

            stops = gen_kwargs_dict.get("until", [])
            if isinstance(stops, str):
                stops = [stops]
            if stops is None:
                stops = []
            max_gen = int(gen_kwargs_dict.get("max_gen_toks", 256))
            temperature = float(gen_kwargs_dict.get("temperature", 0.0))

            input_ids = torch.tensor(context_enc, dtype=torch.long, device=self.device).unsqueeze(0) # [1, T]

            generated_ids: list[int] = []
            generated_text = ""

            for _ in range(max_gen):
                # Truncate to model context window
                input_ids = input_ids[:, -self.max_length:] # [1, <= max_length]

                logits, _ = self.model(input_ids) # [1, T, vocab]
                next_logits = logits[:, -1, :] # [1, vocab]

                # Mask padding token ids (50257-50303) to -inf so they can never be sampled
                # Without this, enc.decode() crashes on out-of-range token ids
                next_logits[:, self.enc.n_vocab:] = -float("inf")

                if temperature <= 0.0: # greedy
                    next_token = torch.argmax(next_logits, dim=-1, keepdim=True) # [1, 1]
                else:
                    next_logits = next_logits / temperature
                    # Sample one token from the distribution
                    probs = F.softmax(next_logits, dim=-1) # [1, vocab_size]
                    next_token = torch.multinomial(probs, num_samples=1) # [1, 1]

                token_id = int(next_token.item())

                if token_id == self.eot_token:
                    break

                # Append sampled token — becomes part of context for next forward pass
                generated_ids.append(token_id)
                input_ids = torch.cat([input_ids, next_token], dim=1)  # [1, T+1]

                # Decode only generated continuation, not prompt + continuation.
                generated_text = self.enc.decode(generated_ids)

                # Stop on earliest stop string
                earliest_stop_index = None
                for stop in stops:
                    if stop == "":
                        continue
                    index = generated_text.find(stop)
                    if index != -1:
                        if earliest_stop_index is None or index < earliest_stop_index:
                            earliest_stop_index = index

                if earliest_stop_index is not None:
                    generated_text = generated_text[:earliest_stop_index]
                    break
            
            # Always append one output per request.
            completions.append(generated_text)

        return completions


