"""生成 AdaQLoRA-Math 使用的 rank_pattern 和 alpha_pattern。

本文件不导入 PyTorch/PEFT，适合本地轻量运行。生成的 JSON 会在
AutoDL 训练时注入 PEFT LoraConfig。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def parse_indices(raw: str) -> set[int]:
    # 将 "0-3,8,10-12" 解析成层编号集合。
    indices: set[int] = set()
    if not raw:
        return indices
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            indices.update(range(int(start), int(end) + 1))
        else:
            indices.add(int(part))
    return indices


def layer_rank(layer_idx: int, low_layers: set[int], high_layers: set[int], low_r: int, mid_r: int, high_r: int) -> int:
    # 根据层编号决定当前层使用低、中、高哪一档 LoRA rank。
    if layer_idx in high_layers:
        return high_r
    if layer_idx in low_layers:
        return low_r
    return mid_r


def build_patterns(
    num_layers: int,
    low_layers: set[int],
    high_layers: set[int],
    low_r: int,
    mid_r: int,
    high_r: int,
    modules: tuple[str, ...] = DEFAULT_MODULES,
) -> dict[str, dict[str, int]]:
    # 为每个 Transformer 层和目标模块生成 PEFT 可识别的正则 pattern。
    rank_pattern: dict[str, int] = {}
    alpha_pattern: dict[str, int] = {}

    for layer_idx in range(num_layers):
        rank = layer_rank(layer_idx, low_layers, high_layers, low_r, mid_r, high_r)
        for module in modules:
            pattern = rf".*layers\.{layer_idx}\..*{module}$"
            rank_pattern[pattern] = rank
            alpha_pattern[pattern] = rank * 2

    return {"rank_pattern": rank_pattern, "alpha_pattern": alpha_pattern}


def summarize_ranks(rank_pattern: dict[str, int]) -> dict[str, float]:
    # 简单统计 rank pattern，方便检查是否真的生成了分层 rank。
    values = list(rank_pattern.values())
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(sum(values) / len(values), 4),
    }


def main() -> None:
    # 命令行入口：生成 AdaQLoRA-Math 的 rank/alpha JSON。
    parser = argparse.ArgumentParser(description="Generate adaptive LoRA rank/alpha pattern JSON.")
    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument("--low-layers", default="0-3")
    parser.add_argument("--high-layers", default="18-27")
    parser.add_argument("--low-r", type=int, default=8)
    parser.add_argument("--mid-r", type=int, default=16)
    parser.add_argument("--high-r", type=int, default=32)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    patterns = build_patterns(
        num_layers=args.num_layers,
        low_layers=parse_indices(args.low_layers),
        high_layers=parse_indices(args.high_layers),
        low_r=args.low_r,
        mid_r=args.mid_r,
        high_r=args.high_r,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(patterns, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "summary": summarize_ranks(patterns["rank_pattern"])}, indent=2))


if __name__ == "__main__":
    main()
