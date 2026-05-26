"""云端生成 MATH 预测答案。

本文件在 AutoDL 上加载 base model 与 LoRA adapter，对 MATH dev/test
样本生成 prediction 字段。本地不需要安装 PyTorch 或 Transformers。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    # 读取待推理样本；limit 用于 smoke run 快速验证。
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if limit is not None and len(records) >= limit:
                break
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    # 写出带 prediction 字段的 JSONL，供 evaluate_accuracy.py 评测。
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_prompt(record: dict[str, Any]) -> str:
    # 推理时只给题目，不能泄露标准 solution 或 answer。
    question = str(record.get("question", "")).strip()
    return (
        "Question:\n"
        f"{question}\n\n"
        "Solution:\n"
        "Let's solve the problem step by step. End with Final Answer: <answer>.\n"
    )


def load_model(
    base_model: str,
    adapter_path: str | None,
    load_in_4bit: bool,
    torch_dtype_name: str,
) -> tuple[Any, Any]:
    # 加载 tokenizer、base model，并在需要时挂载 LoRA adapter。
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    dtype_map = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(torch_dtype_name.lower())
    if torch_dtype is None:
        raise ValueError(f"Unsupported torch dtype: {torch_dtype_name}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch_dtype,
        quantization_config=quant_config,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model


def generate_one(
    tokenizer: Any,
    model: Any,
    prompt: str,
    max_new_tokens: int,
) -> str:
    # 单条贪心生成，默认不采样，保证评测可复现。
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    input_length = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = outputs[0][input_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def generate_records(
    records: list[dict[str, Any]],
    base_model: str,
    adapter_path: str | None,
    load_in_4bit: bool,
    torch_dtype: str,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    # 批量生成 prediction，并保留原始字段方便分难度评测。
    tokenizer, model = load_model(base_model, adapter_path, load_in_4bit, torch_dtype)
    results: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        prediction = generate_one(tokenizer, model, build_prompt(record), max_new_tokens)
        output_record = dict(record)
        output_record["prediction"] = prediction
        results.append(output_record)
        print(f"generated {index}/{len(records)}", flush=True)
    return results


def main() -> None:
    # 命令行入口：传入 base model、adapter、输入和输出路径。
    parser = argparse.ArgumentParser(description="Generate math predictions with a LoRA adapter.")
    parser.add_argument("--base-model", required=True, help="Base model directory or Hugging Face id.")
    parser.add_argument("--adapter", help="LoRA adapter directory. Omit to evaluate the base model only.")
    parser.add_argument("--input", required=True, help="Input JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSONL path with prediction field.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, help="Optional sample limit for quick smoke generation.")
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"],
    )
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading.")
    args = parser.parse_args()

    records = read_jsonl(Path(args.input), args.limit)
    results = generate_records(
        records=records,
        base_model=args.base_model,
        adapter_path=args.adapter,
        load_in_4bit=not args.no_4bit,
        torch_dtype=args.torch_dtype,
        max_new_tokens=args.max_new_tokens,
    )
    write_jsonl(Path(args.output), results)
    print(f"saved predictions to {args.output}")


if __name__ == "__main__":
    main()
