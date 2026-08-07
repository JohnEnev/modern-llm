import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from src.model.gpt import GPT, GPTConfig

# the 3 architecures
V1_ARCH = dict(d_model=1024, n_layers=24, n_heads=16, n_kv_heads=16,
                use_qk_norm=False, use_diff_attn=False, use_xsa=False)
V2_ARCH = dict(d_model=1024, n_layers=24, n_heads=16, n_kv_heads=4,
               use_qk_norm=True, use_diff_attn=True, use_xsa=False)
V3_ARCH = dict(d_model=1536, n_layers=24, n_heads=12, n_kv_heads=3,
               use_qk_norm=True, use_diff_attn=False, use_xsa=True)

# name -> (hf_repo, architecture, is_default_for_its_version)
MODELS = {
    "v1-base": dict(repo="JohnEnev/modern-llm-v1-base", arch=V1_ARCH),
    "v1-sft":  dict(repo="JohnEnev/modern-llm-v1-sft",  arch=V1_ARCH, default=True),
    "v1-grpo": dict(repo="JohnEnev/modern-llm-v1-grpo", arch=V1_ARCH),
    "v2-base": dict(repo="JohnEnev/modern-llm-v2-base", arch=V2_ARCH),
    "v2-sft":  dict(repo="JohnEnev/modern-llm-v2-sft",  arch=V2_ARCH, default=True),
    "v2-grpo": dict(repo="JohnEnev/modern-llm-v2-grpo", arch=V2_ARCH),
    "v3-base": dict(repo="JohnEnev/modern-llm-v3-base", arch=V3_ARCH),
    "v3-sft":  dict(repo="JohnEnev/modern-llm-v3-sft",  arch=V3_ARCH, default=True),
    "v3-grpo": dict(repo="JohnEnev/modern-llm-v3-grpo", arch=V3_ARCH),
}

class ModelRegistry:
    """Loads and holds models by name. Lazy: a model is loaded on first request
    and kept in memory after."""

    def __init__(self, device="cpu"):
        self.device = device
        self._loaded = {} # name -> GPT, populated lazy

    def available(self):
        return list(MODELS.keys())

    def get(self, name):
        """Return the model for `name`, loading it if not already in memory."""

        if name in self._loaded:
            return self._loaded[name] # return the model if already loaded

        if name not in MODELS:
            raise KeyError(f"unknown model '{name}'. Available: {list(MODELS)}") # error if the model doesn't exist

        # Creating and loading the model if not memory
        spec = MODELS[name]

        config = GPTConfig(
                vocab_size=50304, 
                d_model=spec["arch"]["d_model"], 
                n_layers=spec["arch"]["n_layers"],
                n_heads=spec["arch"]["n_heads"],
                n_kv_heads=spec["arch"]["n_kv_heads"],
                dropout=0,
                max_seq_len=1024,
                use_flash=True,
                tie_weights=True,
                use_qk_norm=spec["arch"]["use_qk_norm"],
                use_diff_attn=spec["arch"]["use_diff_attn"],
                use_mhc=False,
                n_streams=0,
                mhc_every_n_layers=0,
                use_xsa=spec["arch"]["use_xsa"],
                           )
        
        # Create the model and move to device
        model = GPT(config).to(self.device)

        # Download the weights from HF
        weights_path = hf_hub_download(repo_id=spec["repo"], filename="model.safetensors")
        state = load_file(weights_path)

        # tied lm_head was dropped on upload; strict=False re-ties on load.
        model.load_state_dict(state, strict=False)
        model.eval()

        # sanity: XSA only active when it should be.
        n_params = sum(p.numel() for p in model.parameters())
        expected = {"v1": 353_000_000, "v2": 316_000_000, "v3": 672_000_000}
        version = name.split("-")[0]           # "v1" / "v2" / "v3"
        lo, hi = expected[version] * 0.98, expected[version] * 1.02
        assert lo < n_params < hi, (
            f"{name}: {n_params:,} params, expected ~{expected[version]:,}. "
            f"Wrong arch dict?"
        )

        self._loaded[name] = model
        return model

# ------ TESTS -------
def test_registry():
    reg = ModelRegistry(device="cpu")
    print(reg.available())
    m = reg.get("v3-sft")                     # downloads on first call
    ids = torch.randint(0, 50304, (1, 4))
    print(list(m.generate_stream(ids, max_new_tokens=5)))   # streams from a real model
    m2 = reg.get("v3-sft")                     # second call, no re-download
    assert m is m2, "should return the cached instance"
    print("✓ registry loads and caches")

if __name__ == "__main__":
    test_registry()

