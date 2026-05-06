# Project Working Rules

This repository is for the LoRA/QLoRA mathematical reasoning fine-tuning project.

## Environment Policy

- GPU training will be performed on a cloud GPU platform such as AutoDL.
- The local workspace should be used for lightweight work only: planning, Markdown documents, configuration files, dataset inspection scripts, evaluation utilities, and code organization.
- Do not install PyTorch, CUDA toolkits, large model libraries, or other heavyweight dependencies locally unless the user explicitly approves.
- Before installing any local dependency or package set that may exceed 1 GB, ask the user for approval first.
- Keep heavyweight cloud dependencies in separate files such as `requirements-gpu.txt`, `environment-autodl.yml`, or setup notes for AutoDL.
- Prefer writing portable scripts and configs that can run on AutoDL after upload or Git sync.

## Project Direction

- Research priority: AdaQLoRA-Math, a difficulty-aware adaptive QLoRA method for mathematical reasoning.
- Main datasets: GSM8K and MATH.
- Main method components: difficulty-aware rank allocation, curriculum LoRA, verifier-guided replay, and efficiency analysis.
- Application target: mathematical problem-solving tutor demo.

## Naming Policy

- Use short, readable names that make the meaning clear at a glance.
- Use `AdaQLoRA-Math` as the human-facing method name.
- Use `AdaQLoRA` as the short method name when the math task is already clear from context.
- Use `QLoRA-Math` for the baseline experiment name.
- For local file names, directories, and experiment paths, underscores are allowed when they improve readability.
- Prefer concise names with one or two underscores at most.
- Avoid long underscore chains that concatenate every detail of the experiment.
- Preferred config names:
  - `qlora_math.yaml`
  - `adaqlora_math.yaml`
- Preferred output paths:
  - `outputs/adapters/qlora_math`
  - `outputs/adapters/adaqlora_math`
- Avoid names like `da_qlora_qwen_math` because they use too many fragments and are harder to scan.

## Collaboration Notes

- When starting substantial work, check this file first.
- Keep local changes small and reproducible.
- Avoid assumptions that require local heavyweight installations.
- If a task needs cloud execution, prepare clear AutoDL commands and environment files instead of running heavyweight training locally.
