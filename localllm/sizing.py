"""What size of model a GPU suits, how fast it should run, and how long a document it can hold.

Everything here is arithmetic from the card's memory size and bandwidth, calibrated on measured runs:
  dense, whole model in VRAM   decode ≈ 0.6 x bandwidth / bytes read per token   (Qwen3.8-27B Q3 on 640 GB/s: 35 tok/s)
  MoE, whole model in VRAM     decode ≈ 0.3 x bandwidth / active bytes per token (gemma-4-26B-A4B Q4: 69 tok/s)
Estimates are labelled as such; measured numbers live in catalog.py.
"""
from __future__ import annotations

import re

# memory bandwidth in GB/s of common consumer GPUs (vendor specs)
BANDWIDTH = {
    "rtx 5090": 1792, "rtx 5080": 960, "rtx 5070 ti": 896, "rtx 5070": 672, "rtx 5060 ti": 448, "rtx 5060": 448,
    "rtx 4090": 1008, "rtx 4080": 717, "rtx 4070 ti": 504, "rtx 4070": 504, "rtx 4060 ti": 288, "rtx 4060": 272,
    "rtx 3090": 936, "rtx 3080": 760, "rtx 3070": 448, "rtx 3060 ti": 448, "rtx 3060": 360,
    "rx 9070 xt": 640, "rx 9070": 640, "rx 9060 xt": 320, "rx 7900 xtx": 960, "rx 7900 xt": 800, "rx 7800 xt": 624,
    "rx 7700 xt": 432, "rx 7600": 288, "rx 6900 xt": 512, "rx 6800 xt": 512, "rx 6700 xt": 384,
    "arc b580": 456, "arc a770": 560, "arc a750": 512,
}
# bits per weight including quantization overhead
QUANTS = [("Q8", 8.5), ("Q6", 6.6), ("Q4", 4.8), ("Q3", 3.9), ("Q2", 2.9)]
LOW_BITS = 3.0          # below this, measured accuracy drops sharply (Qwen3.8-27B: -17 ThaiExam points at 2.9 bpw)
DESKTOP_GB = 1.0        # what the OS/desktop usually keeps on the card
OVERHEAD_GB = 0.7       # compute buffers + small fixed caches
TOKENS_PER_PAGE = 600   # ~450 English words

RAM_EMBED_SHARE = 0.02  # token embeddings + output tensor stay CPU-mapped even when the model lives in VRAM
RAM_HOST_GB = 0.5       # llama-server process + host-side compute buffers (est.)

# reference shapes people actually download: (label, total params B, active params B)
SHAPES = [("4B", 4, 4), ("8B", 8, 8), ("14B", 14, 14), ("24-32B", 27, 27), ("30B MoE (3B active)", 30, 3),
          ("70B", 70, 70), ("120B MoE (10B active)", 120, 10)]


def bandwidth(gpu_name: str) -> int | None:
    n = gpu_name.lower()
    for key in sorted(BANDWIDTH, key=len, reverse=True):
        if re.search(r"\b" + re.escape(key) + r"\b", n):
            return BANDWIDTH[key]
    return None


def size_gb(params_b: float, bpw: float) -> float:
    return params_b * bpw / 8


def tok_s(total_b: float, active_b: float, bpw: float, bw: int | None) -> int | None:
    if not bw:
        return None
    eff = 0.6 if active_b == total_b else 0.3
    return int(eff * bw / size_gb(active_b, bpw))


def tiers(vram_gb: float, ram_gb: float, gpu_name: str) -> list[dict]:
    """One row per reference shape: best quant that fits the card whole, else whether RAM offload works."""
    usable, bw, rows = vram_gb - DESKTOP_GB - OVERHEAD_GB, bandwidth(gpu_name), []
    for label, total, active in SHAPES:
        fit = next(((q, b) for q, b in QUANTS if size_gb(total, b) <= usable), None)
        moe = active < total
        if fit and fit[1] < LOW_BITS and moe and size_gb(total, 4.8) <= usable + 0.7 * ram_gb:
            fit = None   # a MoE keeps its quality better at Q4 with some experts in RAM than at 2 bits on the GPU
        if fit:
            q, b = fit
            rows.append({"shape": label, "status": "low-bits" if b < LOW_BITS else "fits", "quant": q,
                         "gb": round(size_gb(total, b), 1), "tok_s": tok_s(total, active, b, bw)})
        elif size_gb(total, 4.8) <= usable + 0.7 * ram_gb:
            rows.append({"shape": label, "status": "offload-moe" if active < total else "offload-dense", "quant": "Q4",
                         "gb": round(size_gb(total, 4.8), 1), "tok_s": None})
        else:
            rows.append({"shape": label, "status": "too-big", "quant": None, "gb": round(size_gb(total, 4.8), 1),
                         "tok_s": None})
    return rows


def context_tokens(vram_gb: float, model: dict) -> int:
    """How many tokens of conversation/document fit next to the weights (KV cache at q8)."""
    free = vram_gb - DESKTOP_GB - OVERHEAD_GB - model["gb"] - model.get("fixed_cache_gb", 0)
    return max(0, min(model.get("max_ctx", 131072), int(free * 2**20 / model["kv_kb_per_token"])))


def ram_estimate_gb(model: dict) -> dict:
    """Estimate of the chosen model's system-RAM footprint while the server runs (est., not measured).

    The weights themselves live in VRAM; llama.cpp keeps the token embeddings/output tensor in
    RAM (CPU-mapped), plus the server process and host-side compute buffers on top. The KV cache
    counts against VRAM, not RAM, so it isn't included here.
    """
    embed = model["gb"] * RAM_EMBED_SHARE
    return {"embed_gb": round(embed, 1), "cache_gb": round(RAM_HOST_GB, 1),
            "total_gb": round(embed + RAM_HOST_GB, 1)}
