# make_prompts.py
## This function makes the prompt file used for vllm logger evaluation

from datasets import load_dataset
ds = load_dataset("openai/gsm8k", "main", split="test")  # MIT-licensed
prompts = [ex["question"] for ex in ds.select(range(25))]
open("/home/ubuntu/research/MoE-logger/prompts.txt","w").write("\n\n---\n\n".join(prompts))

