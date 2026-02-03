import ast
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def read_records(path: str, ids_dtype=np.int32, wts_dtype=np.float16):
    """Reads the custom bin format:
    meta_line (utf-8, dict string) + ids_raw + wts_raw + newline per record.
    Returns list[(meta, ids, wts)].
    """
    records = []
    with open(path, "rb") as f:
        while True:
            meta_line = f.readline()
            if not meta_line:
                break  # EOF
            meta_line = meta_line.strip()
            if not meta_line:
                continue

            meta = ast.literal_eval(meta_line.decode("utf-8"))
            num_tokens = int(meta["num_tokens"])
            top_k = int(meta["top_k"])
            shape = (num_tokens, top_k)

            n_ids = int(np.prod(shape)) * np.dtype(ids_dtype).itemsize
            n_wts = int(np.prod(shape)) * np.dtype(wts_dtype).itemsize

            ids_buf = f.read(n_ids)
            wts_buf = f.read(n_wts)

            # record separator newline
            _ = f.readline()

            if len(ids_buf) != n_ids or len(wts_buf) != n_wts:
                raise RuntimeError(
                    f"Truncated record in {path}: expected ids {n_ids}B, wts {n_wts}B "
                    f"but got ids {len(ids_buf)}B, wts {len(wts_buf)}B"
                )

            ids = np.frombuffer(ids_buf, dtype=ids_dtype).reshape(shape)
            wts = np.frombuffer(wts_buf, dtype=wts_dtype).reshape(shape)
            records.append((meta, ids, wts))
    return records


def get_vllm_version() -> str:
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version("vllm")
        except PackageNotFoundError:
            # Sometimes the dist name differs (rare), try common alternates
            for name in ("vllm-nightly", "vllm-rocm", "vllm-cpu"):
                try:
                    return version(name)
                except PackageNotFoundError:
                    pass
            return "unknown"
    except Exception:
        return "unknown"
    
def _get_runtime_meta(model_id: str, layers_logged: List[int], top_k: int, seed: int):
    # Best-effort runtime fields
    try:
        import vllm  # type: ignore
        vllm_version = getattr(vllm, "__version__", "unknown")
        if vllm_version == "unknown":
            vllm_version = get_vllm_version()
    except Exception:
        # vllm_version = get_vllm_version()
        vllm_version = "unknown"

    try:
        import torch
        torch_version = getattr(torch, "__version__", "unknown")
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
        else:
            device = "cpu"
    except Exception:
        torch_version = "unknown"
        device = "unknown"

    return {
        "type": "meta",
        "model_id": model_id,
        "vllm_version": vllm_version,
        "torch_version": torch_version,
        "device": device,
        "seed": seed,
        "layers_logged": layers_logged,
        "top_k": top_k,
    }


def bin_to_jsonl(
    bin_path: str,
    jsonl_path: str,
    model_id: str,
    layer_idx: int,
    ids_dtype=np.int32,
    wts_dtype=np.float16,
    req_id: str = "r1",
    seed: int = 1234,
    record_stride: int = 1,
):
    """
    Converts each record in bin into JSONL.
    - Writes one global meta header first.
    - Then for each token: one JSON line with ids & weights.
    """
    recs = read_records(bin_path, ids_dtype=ids_dtype, wts_dtype=wts_dtype)
    if not recs:
        raise RuntimeError(f"No records found in {bin_path}")

    # Determine top_k from first record (assume consistent)
    top_k = int(recs[0][0]["top_k"])

    Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)

    with open(jsonl_path, "w", encoding="utf-8") as out:
        # meta header line
        meta_header = _get_runtime_meta(
            model_id=model_id,
            layers_logged=[layer_idx],
            top_k=top_k,
            seed=seed,
        )
        out.write(json.dumps(meta_header) + "\n")

        # per-token route lines
        # If you have multiple records (e.g., chunked calls), we continue token_idx across records.
        token_base = 0
        for rec_i, (meta, ids, wts) in enumerate(recs):
            num_tokens = int(meta["num_tokens"])
            top_k_local = int(meta["top_k"])
            if top_k_local != top_k:
                raise RuntimeError(
                    f"Inconsistent top_k: header {top_k} but record {rec_i} has {top_k_local}"
                )

            # Optional: downsample to reduce huge output (e.g., record_stride=10)
            for t in range(0, num_tokens, record_stride):
                ids_list = ids[t].tolist()
                # convert float16 -> python float
                wts_list = [float(x) for x in wts[t]]

                line = {
                    "type": "route",
                    "req_id": req_id,
                    "token_idx": token_base + t,
                    "layer": layer_idx,
                    "topk_ids": ids_list,
                    "topk_weights": wts_list,
                }
                out.write(json.dumps(line) + "\n")

            token_base += num_tokens

    print(f"Wrote JSONL: {jsonl_path}  (records={len(recs)}, top_k={top_k})")


if __name__ == "__main__":
    # Example for your file
    bin_path = "./logs/layer_topK_info.bin"
    jsonl_path = "./logs/moe_routes.jsonl"

    bin_to_jsonl(
        bin_path=bin_path,
        jsonl_path=jsonl_path,
        model_id="Qwen/Qwen1.5-MoE-A2.7B-Chat",
        layer_idx=0,
        ids_dtype=np.int32,
        wts_dtype=np.float16,
        req_id="r1",
        seed=1234,
        record_stride=1,  # set 10/100 if you want fewer lines
    )
