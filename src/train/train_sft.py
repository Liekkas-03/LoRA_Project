"""Cloud GPU SFT entry point placeholder.

This file intentionally keeps imports lazy so the local workspace does not need
PyTorch, Transformers, PEFT, TRL, or bitsandbytes installed. The actual training
implementation should be completed and run on AutoDL.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QLoRA-Math or AdaQLoRA-Math SFT on a cloud GPU host.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "PyYAML is required for cloud training. Install requirements-gpu.txt on AutoDL first."
        ) from exc

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment_name = config.get("experiment", {}).get("name", "unknown")

    raise NotImplementedError(
        "Training dependencies are intentionally not installed locally. "
        f"Config '{experiment_name}' was loaded successfully from {config_path}. "
        "Next step: implement the AutoDL training loop with Transformers + PEFT + TRL."
    )


if __name__ == "__main__":
    main()
