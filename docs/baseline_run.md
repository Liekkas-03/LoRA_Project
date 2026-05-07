# QLoRA Baseline 云端运行清单

本文档说明如何在 AutoDL 或类似云 GPU 平台上跑通第一版 `QLoRA-Math` baseline。本地仍然不安装 PyTorch、Transformers、PEFT、TRL 或 bitsandbytes。

## 目标

第一版 baseline 只做一件事：在统一后的 `train.jsonl/dev.jsonl` 上，用 `Qwen2.5-Math-7B` 跑通 4-bit QLoRA SFT，并保存 LoRA adapter。

跑通后我们会得到：

- `outputs/adapters/qlora_math/`：baseline adapter 和 tokenizer 文件。
- `outputs/logs/` 或 TensorBoard 日志：训练 loss、eval loss。
- 后续可接推理脚本生成 `prediction`，再进入准确率评测。

## 1. 云端环境

建议选择 AutoDL 中已经预装 PyTorch/CUDA 的镜像。进入项目目录后安装云端依赖：

```bash
pip install -r requirements-gpu.txt
```

如果镜像已经带 PyTorch，不建议手动重装 PyTorch，避免 CUDA 版本冲突。

## 2. 准备原始数据

推荐目录：

```text
data/raw/
  gsm8k/
    train.jsonl
    test.jsonl
  MATH/
    train/
    test/
```

`data/raw/` 不提交 Git，只在云端或本地数据盘保留。

## 3. 转换成统一格式

GSM8K：

```bash
python -m src.data.prepare_gsm8k --input data/raw/gsm8k/train.jsonl --output data/processed/gsm8k_train.jsonl --split train
python -m src.data.prepare_gsm8k --input data/raw/gsm8k/test.jsonl --output data/processed/gsm8k_test.jsonl --split test
```

MATH：

```bash
python -m src.data.prepare_math --input-dir data/raw/MATH/train --output data/processed/math_train.jsonl --split train
python -m src.data.prepare_math --input-dir data/raw/MATH/test --output data/processed/math_test.jsonl --split test
```

## 4. 划分 dev

从训练数据中固定随机种子划分 dev：

```bash
python -m src.data.split_data --input data/processed/gsm8k_train.jsonl --train-output data/processed/gsm8k_train_split.jsonl --dev-output data/processed/gsm8k_dev.jsonl --dev-ratio 0.1 --seed 42
python -m src.data.split_data --input data/processed/math_train.jsonl --train-output data/processed/math_train_split.jsonl --dev-output data/processed/math_dev.jsonl --dev-ratio 0.1 --seed 42
```

## 5. 合并训练和 dev 文件

`configs/qlora_math.yaml` 默认读取：

```text
data/processed/train.jsonl
data/processed/dev.jsonl
```

生成这两个文件：

```bash
python -m src.data.merge_data --inputs data/processed/gsm8k_train_split.jsonl data/processed/math_train_split.jsonl --output data/processed/train.jsonl
python -m src.data.merge_data --inputs data/processed/gsm8k_dev.jsonl data/processed/math_dev.jsonl --output data/processed/dev.jsonl
```

## 6. 运行 QLoRA baseline

```bash
bash scripts/autodl_train.sh configs/qlora_math.yaml
```

等价命令：

```bash
python -m src.train.train_sft --config configs/qlora_math.yaml
```

## 7. 训练完成后检查

确认以下目录存在：

```text
outputs/adapters/qlora_math/
```

至少应包含 adapter 相关文件，例如：

```text
adapter_config.json
adapter_model.safetensors
tokenizer_config.json
```

## 8. 注意事项

- 第一次 baseline 只追求跑通，不要急着调参数。
- 如果显存不足，优先降低 `max_seq_length` 到 `1024`，或增大 `gradient_accumulation_steps` 并保持 batch size 为 `1`。
- 如果下载模型很慢，可以在 AutoDL 上提前配置 Hugging Face 镜像或模型缓存。
- 训练日志、adapter 和 checkpoint 不提交 Git。

