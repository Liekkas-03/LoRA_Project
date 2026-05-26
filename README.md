# AdaQLoRA-Math

本项目研究 **AdaQLoRA-Math：面向 MATH 官方难度等级的自适应 QLoRA 微调方法**。当前科研主线只使用 MATH 数据集，利用其官方 `Level 1-5` 难度标签构建难度感知训练、分层评测和 LoRA rank 分配实验。

本地环境只做轻量开发：文档、配置、数据处理脚本、评测工具和结果整理。PyTorch、Transformers、PEFT、bitsandbytes 等重依赖只在 AutoDL 等云 GPU 环境中安装和运行。

## 研究主线

- 方法名：`AdaQLoRA-Math`
- 主数据集：`MATH`
- 基座模型：`Qwen2.5-Math-7B` 或 `Qwen2.5-Math-7B-Instruct`
- 难度划分：`easy = Level 1-2`，`medium = Level 3`，`hard = Level 4-5`
- 主对比：`QLoRA-Math` vs `AdaQLoRA-Math`
- 当前方法重点：将统一 LoRA rank 改为分层 rank pattern，让不同 Transformer 层使用不同 LoRA 容量
- 后续扩展：参数量公平消融、课程学习、错误样本 replay、效率分析

## 目录结构

```text
LoRA_Project/
  AGENTS.md
  README.md
  Scheme.md
  configs/
    qlora_math.yaml
    adaqlora_math.yaml
  data/
    samples/
      math_preds.jsonl
      math_raw/
  docs/
    autodl_steps.md
    adaqlora_result.md
  scripts/
    local_smoke.ps1
    autodl_train.sh
    autodl_generate.sh
  src/
    data/
      prepare_math.py
      split_data.py
      difficulty_scorer.py
    eval/
      extract_answer.py
      math_verifier.py
      evaluate_accuracy.py
      error_classifier.py
    infer/
      generate_answers.py
    lora/
      rank_allocator.py
    train/
      prompting.py
      train_sft.py
  outputs/
    reports/
```

## 本地轻量检查

本地不安装训练依赖，只运行标准库脚本：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\local_smoke.ps1
```

这个脚本会检查 MATH 样例转换、按官方难度划分 train/dev、小样本评测和难度重标注流程。

## AutoDL 主流程

完整云端步骤见 [docs/autodl_steps.md](docs/autodl_steps.md)。核心流程如下：

```bash
python -m src.data.prepare_math \
  --input-dir data/raw/math/MATH/train \
  --output data/processed/math_train.jsonl \
  --split train

python -m src.data.prepare_math \
  --input-dir data/raw/math/MATH/test \
  --output data/processed/math_test.jsonl \
  --split test

python -m src.data.split_data \
  --input data/processed/math_train.jsonl \
  --train-output data/processed/math_train_split.jsonl \
  --dev-output data/processed/math_dev.jsonl \
  --dev-ratio 0.1 \
  --seed 42 \
  --stratify-field difficulty_group

bash scripts/autodl_train.sh configs/qlora_math.yaml
bash scripts/autodl_train.sh configs/adaqlora_math.yaml
```

训练完成后在云端生成预测并评测：

```bash
bash scripts/autodl_generate.sh \
  /root/models/Qwen/Qwen2.5-Math-7B-Instruct \
  outputs/adapters/adaqlora_math \
  data/processed/math_test.jsonl \
  outputs/reports/math_test_ada_preds.jsonl \
  512

python -m src.eval.evaluate_accuracy \
  --input outputs/reports/math_test_ada_preds.jsonl \
  --output outputs/reports/math_test_ada_metrics.json
```

## 历史结果

根目录下保留的 `qlora_math_formal_*.tar.gz` 和 `adaqlora_math_formal_*.tar.gz` 是此前探索实验的导出结果。它们不会删除，但新科研主线以 MATH-only 重新跑出的结果为准。历史结果解释见 [docs/adaqlora_result.md](docs/adaqlora_result.md)。
