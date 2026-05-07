"""Cloud GPU SFT entry point.

This file intentionally keeps imports lazy so the local workspace does not need
PyTorch, Transformers, PEFT, TRL, or bitsandbytes installed. The actual training
implementation should be run on AutoDL.

本文件是 AutoDL 云端监督微调入口。它会读取 YAML 配置，在云端延迟导入
PyTorch/Transformers/PEFT/TRL，并启动 QLoRA 或 AdaQLoRA 的 SFT 训练。
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

from src.train.prompting import format_example


def load_config(config_path: Path) -> dict[str, Any]:
    # 读取 YAML 配置；PyYAML 只在训练入口运行时导入，避免本地强依赖。
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "PyYAML is required for cloud training. Install requirements-gpu.txt on AutoDL first."
        ) from exc
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def torch_dtype_from_name(torch_module: Any, name: str | bool | None) -> Any:
    # 将配置中的 dtype 字符串转换成 torch dtype 对象。
    if name is True:
        return torch_module.bfloat16
    if not name:
        return None
    lowered = str(name).lower()
    if lowered in {"bf16", "bfloat16"}:
        return torch_module.bfloat16
    if lowered in {"fp16", "float16"}:
        return torch_module.float16
    if lowered in {"fp32", "float32"}:
        return torch_module.float32
    if lowered == "auto":
        return "auto"
    raise ValueError(f"Unsupported dtype: {name}")


def build_quant_config(config: dict[str, Any], torch_module: Any, bits_config_cls: Any) -> Any:
    # 根据 model 配置构造 4-bit QLoRA 量化配置。
    model_cfg = config.get("model", {})
    if not model_cfg.get("load_in_4bit", False):
        return None
    return bits_config_cls(
        load_in_4bit=True,
        bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=torch_dtype_from_name(
            torch_module, model_cfg.get("bnb_4bit_compute_dtype", "bfloat16")
        ),
        bnb_4bit_use_double_quant=bool(model_cfg.get("bnb_4bit_use_double_quant", True)),
    )


def load_rank_patterns(path: str | None) -> tuple[dict[str, int] | None, dict[str, int] | None]:
    # 读取 rank_allocator.py 生成的 rank_pattern/alpha_pattern JSON。
    if not path:
        return None, None
    pattern_path = Path(path)
    if not pattern_path.exists():
        return None, None
    patterns = json.loads(pattern_path.read_text(encoding="utf-8"))
    return patterns.get("rank_pattern"), patterns.get("alpha_pattern")


def build_lora_config(config: dict[str, Any], lora_config_cls: Any) -> Any:
    # 根据配置构造 PEFT LoraConfig，AdaQLoRA 会额外尝试加载 rank/alpha pattern。
    lora_cfg = config.get("lora", {})
    adaptive_cfg = config.get("adaptive_lora", {})
    method = str(lora_cfg.get("method", "qlora")).lower()

    rank = int(lora_cfg.get("r", lora_cfg.get("base_r", 16)))
    alpha = int(lora_cfg.get("lora_alpha", rank * 2))
    rank_pattern = None
    alpha_pattern = None
    if method == "adaqlora":
        rank_pattern, alpha_pattern = load_rank_patterns(adaptive_cfg.get("rank_pattern_output"))

    kwargs = {
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": float(lora_cfg.get("lora_dropout", 0.05)),
        "bias": lora_cfg.get("bias", "none"),
        "task_type": lora_cfg.get("task_type", "CAUSAL_LM"),
        "target_modules": list(lora_cfg.get("target_modules", [])),
    }
    if rank_pattern:
        kwargs["rank_pattern"] = rank_pattern
    if alpha_pattern:
        kwargs["alpha_pattern"] = alpha_pattern
    return lora_config_cls(**kwargs)


def map_text_field(dataset: Any, template: str) -> Any:
    # 为每条样本增加 text 字段，供 TRL SFTTrainer 直接读取。
    def add_text(record: dict[str, Any]) -> dict[str, str]:
        return {"text": format_example(record, template)}

    return dataset.map(add_text)


def build_sft_config(config: dict[str, Any], sft_config_cls: Any) -> Any:
    # 构造 TRL SFTConfig，并按当前安装版本过滤不支持的参数。
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    experiment_cfg = config.get("experiment", {})
    kwargs = {
        "output_dir": experiment_cfg.get("output_dir", "outputs/adapters/qlora_math"),
        "per_device_train_batch_size": training_cfg.get("per_device_train_batch_size", 1),
        "gradient_accumulation_steps": training_cfg.get("gradient_accumulation_steps", 16),
        "learning_rate": training_cfg.get("learning_rate", 2e-4),
        "num_train_epochs": training_cfg.get("num_train_epochs", 2),
        "warmup_ratio": training_cfg.get("warmup_ratio", 0.03),
        "lr_scheduler_type": training_cfg.get("lr_scheduler_type", "cosine"),
        "bf16": bool(training_cfg.get("bf16", True)),
        "gradient_checkpointing": bool(training_cfg.get("gradient_checkpointing", True)),
        "logging_steps": training_cfg.get("logging_steps", 10),
        "save_steps": training_cfg.get("save_steps", 500),
        "eval_steps": training_cfg.get("eval_steps", 500),
        "evaluation_strategy": "steps",
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "report_to": training_cfg.get("report_to", "tensorboard"),
        "dataset_text_field": data_cfg.get("text_field", "text"),
        "max_length": data_cfg.get("max_length", data_cfg.get("max_seq_length", 2048)),
        "max_seq_length": data_cfg.get("max_seq_length", 2048),
        "packing": bool(data_cfg.get("packing", False)),
        "dataset_num_proc": data_cfg.get("dataset_num_proc", None),
    }
    signature = inspect.signature(sft_config_cls.__init__)
    allowed = set(signature.parameters)
    filtered = {key: value for key, value in kwargs.items() if key in allowed and value is not None}
    return sft_config_cls(**filtered)


def build_trainer(trainer_cls: Any, model: Any, tokenizer: Any, peft_config: Any, sft_args: Any, datasets: Any) -> Any:
    # 兼容不同 TRL 版本的 SFTTrainer 参数名。
    trainer_kwargs = {
        "model": model,
        "args": sft_args,
        "train_dataset": datasets["train"],
        "eval_dataset": datasets.get("validation"),
    }
    signature = inspect.signature(trainer_cls.__init__)
    allowed = set(signature.parameters)
    if "peft_config" in allowed:
        trainer_kwargs["peft_config"] = peft_config
    if "processing_class" in allowed:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in allowed:
        trainer_kwargs["tokenizer"] = tokenizer
    return trainer_cls(**{key: value for key, value in trainer_kwargs.items() if key in allowed and value is not None})


def run_training(config: dict[str, Any]) -> None:
    # 真正的云端训练流程：加载数据、模型、LoRA 配置并启动 SFT。
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from trl import SFTConfig, SFTTrainer

    experiment_cfg = config.get("experiment", {})
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})

    set_seed(int(experiment_cfg.get("seed", 42)))

    data_files = {"train": data_cfg["train_file"]}
    if data_cfg.get("eval_file"):
        data_files["validation"] = data_cfg["eval_file"]
    datasets = load_dataset("json", data_files=data_files)
    datasets = datasets.map(lambda record: {"text": format_example(record, data_cfg.get("prompt_template", "math_cot"))})

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name_or_path"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant_config = build_quant_config(config, torch, BitsAndBytesConfig)
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name_or_path"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        device_map=model_cfg.get("device_map", "auto"),
        torch_dtype=torch_dtype_from_name(torch, model_cfg.get("torch_dtype", "auto")),
        quantization_config=quant_config,
    )
    model.config.use_cache = False
    if model_cfg.get("load_in_4bit", False):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(training_cfg.get("gradient_checkpointing", True)),
        )

    peft_config = build_lora_config(config, LoraConfig)
    sft_args = build_sft_config(config, SFTConfig)
    trainer = build_trainer(SFTTrainer, model, tokenizer, peft_config, sft_args, datasets)

    trainer.train()
    trainer.save_model(experiment_cfg.get("output_dir", "outputs/adapters/qlora_math"))
    tokenizer.save_pretrained(experiment_cfg.get("output_dir", "outputs/adapters/qlora_math"))


def main() -> None:
    # 命令行入口：接收 YAML 配置路径，后续云端训练会从这里启动。
    parser = argparse.ArgumentParser(description="Run QLoRA-Math or AdaQLoRA-Math SFT on a cloud GPU host.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    config = load_config(config_path)
    run_training(config)


if __name__ == "__main__":
    main()
