<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
MoE Top-K Routing Logger for vLLM (Minimal-Overhead)
</h3>

## What this repo contains

A small patch/fork of vLLM that logs MoE router **top-k expert IDs + weights** for a **target MoE layer** with a runtime flag.

Artifacts from one run:

- `moe_routes.jsonl` (routing log)
- `expert_hist.png` (raw expert selection histogram)
- `expert_norm.png` (normalized routing distribution)
- `timing.json` (wall times with/without logging)
- plot script (builds hist + normalized plot + entropy)

---

## 1) Where we hooked in vLLM

**Hook point:** `vllm/model_executor/layers/fused_moe/layer.py`, inside class `FusedMoE(CustomOp)`.

**Data source:** in `FusedMoE.forward_impl(...)`, when router produces:

```python
topk_weights, topk_ids = self.router.select_experts(...)
```
Gating logic: we read vllm_config.additional_config and only log when:
```
record_topk=True
layer_id == target_layer_idx
```
This lets you turn logging on/off without changing model code paths, and avoids logging all layers.

## 2) The “trick”: asynchronous .bin logging with minimal latency overhead
We implemented an AsyncTopKLogger with Non-blocking D2H copies to pinned CPU memory:
```python
ids_cpu.copy_(topk_ids, non_blocking=True)
wts_cpu.copy_(topk_weights, non_blocking=True)
```

A CUDA event recorded on the current stream so we can wait later:
```python
ev.record(torch.cuda.current_stream())
```

A background thread that:
```python
waits on event.synchronize() (off the critical path),
```
appends a compact record to a .bin file.

Key idea: the forward pass does no blocking sync and no JSON work. It only enqueues a copy + pushes a small object into a bounded queue.
If the queue is full, we drop records instead of slowing inference.

File format choice: We write a simple binary append format. This avoids expensive JSON serialization during inference.

CUDA graph safety: We also skip logging when capture is active:
```python
if torch.cuda.is_current_stream_capturing():
    return
```
## 3) Post-processing: .bin → moe_routes.jsonl after inference completes

## 4) How to run
install vllm: cd into vllm dir and then: pip install -e .
run: make the data by python make_prompt.py and then adjust your environment variables e.g. saving path, and then python main.py, then check the ./logs
Then, run python plot_hist.py for the figures and some statististics.

## 5) Results note:
- Top-3 experts (by selection probability): expert 11 (2.598%), expert 54 (2.532%), expert 22 (2.393%).
- Normalized distribution: routing is nearly uniform across 60 active experts (support=60).
- Entropy metric: entropy = 5.8929 bits, normalized entropy = 0.9976 (≈ 1.0 is uniform over support).
- Interpretation: the router spreads traffic almost evenly across experts, indicating strong load balancing (little expert collapse).
- Timing: logging run A=3.97s vs no-log B=2.06s → overhead ≈ 1.91s in this setting.

6) AI usage log (how outputs were verified)

Used ChatGPT to:

- design the minimal-overhead logging approach (CUDA event + pinned memory + background thread),
- propose robust file/path handling (create parent dir, avoid treating file path as directory),
- provide analysis scripts (histogram, normalized distribution, entropy).

Verification steps:
- sanity-checked that Total selections ≈ route_records × top_k (here: 505,811 × 4 = 2,023,244 matches),
- verified JSONL schema by spot-checking first lines and ensuring topk_ids/topk_weights lengths match top_k,
- confirmed plots reflect counts and normalized probabilities.
- GPU device: 1 card H100-81Gig-HBM3