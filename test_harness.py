import lm_eval
from src.eval.harness_adapter import CustomGPTLM, build_eval_model

# V1 GRPO checkpoint (350M, V1 arch = defaults). EMA off (GRPO saves model_weights).
model = build_eval_model(
    "/workspace/checkpoints_grpo_v1/percentage/grpo_final_percentage.pt",
    device="cuda",
    # V1 arch defaults are correct (d_model 1024, n_layers 24, n_heads 16,
    # n_kv_heads 16, flags off). Confirm these match your V1 config!
)

lm = CustomGPTLM(model=model, device="cuda", batch_size=8, max_length=1024)

# ONE fast task first — lambada_openai is small and quick, exercises loglikelihood.
results = lm_eval.simple_evaluate(
    model=lm,
    tasks=["lambada_openai"],
    num_fewshot=0,
    limit=50,   # only 50 examples for the smoke test — fast. Remove for full run.
)

print(results["results"])