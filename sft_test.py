import torch
import tiktoken
from src.model.gpt import GPT, GPTConfig

enc = tiktoken.get_encoding("gpt2")
model_config = GPTConfig()

prompts = [
    "User: What is the capital of France?\nAssistant:",
    "User: Write a Python function to reverse a string.\nAssistant:",
    "User: What is 15% of 80?\nAssistant:",
    "User: Explain what a transformer model is.\nAssistant:",
    "User: What causes rainbows?\nAssistant:",
]

for ckpt_path, label in [
    ("checkpoints/step_020000.pt", "BASE MODEL"),
    ("/workspace/checkpoints_sft/sft_final.pt", "SFT MODEL"),
]:
    print(f"\n{'='*60}")
    print(f"{label}")
    print('='*60)

    model = GPT(model_config)
    ckpt = torch.load(ckpt_path, weights_only=False)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_weights"].items()}
    model.load_state_dict(state_dict)
    model.eval()

    for prompt in prompts:
        tokens = enc.encode(prompt)
        input_ids = torch.tensor([tokens])
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=80, temperature=0.4)
        output_tokens = output[0].tolist()[len(tokens):]
        eot = enc.eot_token
        if eot in output_tokens:
            output_tokens = output_tokens[:output_tokens.index(eot)]
        print(f"\n>>> {prompt}")
        print(enc.decode(output_tokens))
        print("---")