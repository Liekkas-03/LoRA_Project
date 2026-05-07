"""Generate PEFT rank_pattern and alpha_pattern templates.

This helper is lightweight and does not import PyTorch or PEFT. The generated
JSON can later be converted into a PEFT LoraConfig on the cloud GPU machine.

本文件负责生成自适应 LoRA 的 rank_pattern 和 alpha_pattern 模板，
后续可在 AutoDL 上接入 PEFT 的 LoraConfig，实现不同层使用不同 rank。
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
    # 将 "0-3,8" 这类层编号字符串解析成整数集合。
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


def build_patterns(
    num_layers: int,
    low_layers: set[int],
    high_layers: set[int],
    low_r: int,
    mid_r: int,
    high_r: int,
    modules: tuple[str, ...] = DEFAULT_MODULES,
) -> dict[str, dict[str, int]]:
    # 按层编号和模块名生成 rank/alpha 配置，供 PEFT 后续匹配模型参数名。
    rank_pattern: dict[str, int] = {}
    alpha_pattern: dict[str, int] = {}

    for layer_idx in range(num_layers):
        # 高敏感层用更大 rank，低敏感层用更小 rank，其余层使用默认 rank。
        if layer_idx in high_layers:
            rank = high_r
        elif layer_idx in low_layers:
            rank = low_r
        else:
            rank = mid_r

        for module in modules:
            # pattern 用正则匹配类似 layers.20.xxx.q_proj 的模块名。
            pattern = rf".*layers\.{layer_idx}\..*{module}$"
            rank_pattern[pattern] = rank
            alpha_pattern[pattern] = rank * 2

    return {"rank_pattern": rank_pattern, "alpha_pattern": alpha_pattern}


def main() -> None:
    # 命令行入口：根据层范围和 rank 参数生成 JSON 配置文件。
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
    print(json.dumps({"output": str(output_path), "patterns": len(patterns["rank_pattern"])}, indent=2))


if __name__ == "__main__":
    main()
