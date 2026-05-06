"""Generate PEFT rank_pattern and alpha_pattern templates.

This helper is lightweight and does not import PyTorch or PEFT. The generated
JSON can later be converted into a PEFT LoraConfig on the cloud GPU machine.
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
    rank_pattern: dict[str, int] = {}
    alpha_pattern: dict[str, int] = {}

    for layer_idx in range(num_layers):
        if layer_idx in high_layers:
            rank = high_r
        elif layer_idx in low_layers:
            rank = low_r
        else:
            rank = mid_r

        for module in modules:
            pattern = rf".*layers\.{layer_idx}\..*{module}$"
            rank_pattern[pattern] = rank
            alpha_pattern[pattern] = rank * 2

    return {"rank_pattern": rank_pattern, "alpha_pattern": alpha_pattern}


def main() -> None:
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

