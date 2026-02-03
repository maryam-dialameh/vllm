# run_generate_twice.py
import sys, os, json, time, random, gc
# sys.path.append("/home/ubuntu/research/MoE-logger/vllm/")

from vllm import LLM, SamplingParams

# ------------------ config ------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # GPU id (within this process)
random.seed(1234)

MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
PROMPTS_PATH = "./prompts.txt"
LOG_DIR = "./logs"


TIMING_LOG = os.path.join(LOG_DIR, "timing_runs.jsonl")
# remove old log file if it exists
# if os.path.exists(TIMING_LOG):
#     os.remove(TIMING_LOG)
os.makedirs(LOG_DIR, exist_ok=True)

prompts = open(PROMPTS_PATH, "r", encoding="utf-8").read().split("\n\n---\n\n")

# Sampling
sp = SamplingParams(temperature=0.0, max_tokens=128, seed=1234)  # seed supported in newer vLLM
# --------------------------------------------


def shutdown_llm(llm):
    # vLLM v1: EngineCore client lives here
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:
        pass


def run_once(*, target_layer_idx: int | None, run_name: str):
    # measure init separately
    t_init0 = time.time()
    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        max_model_len=2048,
        target_layer_idx=target_layer_idx,
    )
    t_init1 = time.time()

    t_gen0 = time.time()
    try:
        outs = llm.generate(prompts, sp)
    finally:
        shutdown_llm(llm)
    t_gen1 = time.time()

    # tokens generated
    tokens_generated = sum(len(o.outputs[0].token_ids) for o in outs)

    rec = {
        "type": "timing",
        "run_name": run_name,
        "record_topk": os.environ["VLLM_LOG_MOE"] != "",
        "target_layer_idx": target_layer_idx,
        "model_id": MODEL_ID,
        "num_prompts": len(prompts),
        "max_new_tokens": sp.max_tokens,
        "temperature": sp.temperature,
        "seed": 1234,
        "init_wall_time_sec": t_init1 - t_init0,
        "generate_wall_time_sec": t_gen1 - t_gen0,
        "total_wall_time_sec": (t_init1 - t_init0) + (t_gen1 - t_gen0),
        "tokens_generated": int(tokens_generated),
        "timestamp_unix": time.time(),
    }

    # append to ONE timing log
    with open(TIMING_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

    # cleanup between runs (important)
    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return rec


if __name__ == "__main__":
    # Run A: with logging
    os.environ["VLLM_LOG_MOE"] = LOG_DIR
    rec_a = run_once(target_layer_idx=5, run_name="with_topk_logging")

    # Run B: without logging
    os.environ["VLLM_LOG_MOE"] = ""
    rec_b = run_once(target_layer_idx=None, run_name="no_topk_logging")

    print("Wrote timings to:", TIMING_LOG)
    print("Option A, with_topk_logging:", rec_a["generate_wall_time_sec"], "sec")
    print("Option B, no_topk_logging", rec_b["generate_wall_time_sec"], "sec")
