# AdaQLoRA-Math

本项目研究 **AdaQLoRA-Math：面向数学推理的难度感知自适应 QLoRA 微调**，目标是在单卡云 GPU 条件下，用 LoRA/QLoRA 完成可复现实验，并把科研结果转化为数学解题辅导系统。

当前阶段是项目骨架搭建：本地只放轻量脚本、配置、文档和评测工具；PyTorch、Transformers、PEFT、bitsandbytes 等重依赖只在 AutoDL 等云平台安装和运行。

## 研究主线

- 方法名：`AdaQLoRA-Math`
- 主任务：数学推理监督微调
- 主数据集：`GSM8K`、`MATH`
- 主模型：`Qwen2.5-Math-7B` 或 `Qwen2.5-Math-7B-Instruct`
- 核心方法：difficulty-aware rank allocation、curriculum LoRA、verifier-guided replay
- 主要评测：最终答案准确率、分难度准确率、错误类型、训练显存、训练时间、adapter 大小

## 环境原则

- 本地环境不安装 PyTorch/CUDA/大模型训练库。
- 本地依赖只用于轻量数据处理和评测。
- 如果需要在本地安装超过 `1GB` 的依赖，必须先获得用户确认。
- GPU 训练、模型下载和大型依赖安装放到 AutoDL 云端执行。
- 项目长期规则见 [AGENTS.md](AGENTS.md)。

## 目录结构

```text
LoRA_Project/
  AGENTS.md
  README.md
  Scheme.md
  requirements-local.txt
  requirements-gpu.txt
  configs/
    qlora_math.yaml
    adaqlora_math.yaml
  data/
    samples/
      math_preds.jsonl
  docs/
    autodl_setup.md
  scripts/
    local_smoke.ps1
    autodl_train.sh
  src/
    data/
      difficulty_scorer.py
    eval/
      extract_answer.py
      math_verifier.py
      evaluate_accuracy.py
      error_classifier.py
    lora/
      rank_allocator.py
    train/
      train_sft.py
  outputs/
    logs/
    reports/
```

## 本地可做的事

本地只运行标准库脚本，不需要安装重依赖。

```powershell
python -m src.eval.evaluate_accuracy --input data\samples\math_preds.jsonl
python -m src.eval.error_classifier --input data\samples\math_preds.jsonl --output outputs\reports\sample_errors.jsonl
python -m src.data.difficulty_scorer --input data\samples\math_preds.jsonl --output outputs\reports\sample_scored.jsonl
```

如果当前机器的 `python` 命令不可用，运行轻量 smoke test 脚本即可；它会优先使用 `D:\Anaconda3\python.exe`：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\local_smoke.ps1
```

## 云端训练流程

AutoDL 上再安装 GPU 依赖：

```bash
pip install -r requirements-gpu.txt
```

启动 QLoRA baseline：

```bash
bash scripts/autodl_train.sh configs/qlora_math.yaml
```

启动 AdaQLoRA-Math：

```bash
bash scripts/autodl_train.sh configs/adaqlora_math.yaml
```

详细云端说明见 [docs/autodl_setup.md](docs/autodl_setup.md)。

## 第一阶段目标

1. 跑通 GSM8K/MATH 数据读取和格式统一。
2. 跑通答案抽取与准确率评测。
3. 建立 zero-shot/few-shot baseline 评测脚本。
4. 在 AutoDL 上跑通 QLoRA baseline。
5. 加入难度分层、rank pattern、curriculum 和 replay 消融。
