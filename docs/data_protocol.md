# 数据协议

本文档说明 `AdaQLoRA-Math` 科研线的数据准备、统一格式、划分原则和评测协议。当前阶段只做轻量脚本和文档，不在本地安装 PyTorch、Transformers 或其他大模型训练依赖。

## 数据来源

主数据集：

- `GSM8K`：小学数学文字题，原始格式通常是 JSONL，每条样本包含 `question` 和 `answer`。
- `MATH`：竞赛数学题，原始格式通常是目录结构，每个题目是一个 JSON 文件，包含 `problem`、`solution`、`level`、`type` 等字段。

推荐原始数据放置方式：

```text
data/raw/
  gsm8k/
    train.jsonl
    test.jsonl
  MATH/
    train/
      algebra/
        1.json
    test/
      algebra/
        1.json
```

`data/raw/` 和 `data/processed/` 默认不提交到 Git，避免误上传完整数据集或后续生成的大文件。

## 统一样本格式

所有数据集转换后统一写成 JSONL，每行一条样本：

```json
{
  "id": "gsm8k-train-000001",
  "dataset": "gsm8k",
  "split": "train",
  "question": "...",
  "solution": "...",
  "answer": "42",
  "level": "",
  "type": "grade_school_math",
  "difficulty_score": 1.42,
  "difficulty_group": "easy"
}
```

字段说明：

- `id`：项目内部稳定样本 ID。
- `dataset`：数据集名称，例如 `gsm8k` 或 `math`。
- `split`：数据划分，取 `train/dev/test`。
- `question`：题目文本。
- `solution`：标准完整解题过程。
- `answer`：最终答案，优先由 `extract_answer.py` 从标准解答中抽取。
- `level`：数据集自带难度字段，MATH 常见为 `Level 1` 到 `Level 5`。
- `type`：题目类别，例如 `algebra`、`number_theory`。
- `difficulty_score`：启发式连续难度分。
- `difficulty_group`：难度分组，取 `easy/medium/hard`。

## 数据准备命令

GSM8K：

```powershell
python -m src.data.prepare_gsm8k --input data\raw\gsm8k\train.jsonl --output data\processed\gsm8k_train.jsonl --split train
python -m src.data.prepare_gsm8k --input data\raw\gsm8k\test.jsonl --output data\processed\gsm8k_test.jsonl --split test
```

MATH：

```powershell
python -m src.data.prepare_math --input-dir data\raw\MATH\train --output data\processed\math_train.jsonl --split train
python -m src.data.prepare_math --input-dir data\raw\MATH\test --output data\processed\math_test.jsonl --split test
```

如果当前 Windows 环境的 `python` 命令不可用，可以使用：

```powershell
D:\Anaconda3\python.exe -m src.data.prepare_gsm8k --input data\raw\gsm8k\train.jsonl --output data\processed\gsm8k_train.jsonl --split train
```

## 划分原则

建议第一阶段采用：

- `train`：用于 QLoRA/AdaQLoRA 训练。
- `dev`：用于选择 replay 样本、调 curriculum 比例、观察错误类型。
- `test`：只用于最终报告，不参与调参和 replay。

如果原始数据没有 dev 集，可以从 train 中固定随机种子划分 5%-10%。划分脚本后续单独补充，避免现在过早把数据策略写死。

当前提供轻量划分脚本：

```powershell
python -m src.data.split_data --input data\processed\gsm8k_train.jsonl --train-output data\processed\gsm8k_train_split.jsonl --dev-output data\processed\gsm8k_dev.jsonl --dev-ratio 0.1 --seed 42
```

默认按 `difficulty_group` 分层抽取 dev，使 easy/medium/hard 的分布尽量稳定。如果不想分层，可以传空字符串：

```powershell
python -m src.data.split_data --input data\processed\gsm8k_train.jsonl --train-output data\processed\gsm8k_train_split.jsonl --dev-output data\processed\gsm8k_dev.jsonl --stratify-field ""
```

划分后可将多个数据集合并成训练总文件：

```powershell
python -m src.data.merge_data --inputs data\processed\gsm8k_train_split.jsonl data\processed\math_train_split.jsonl --output data\processed\train.jsonl
python -m src.data.merge_data --inputs data\processed\gsm8k_dev.jsonl data\processed\math_dev.jsonl --output data\processed\dev.jsonl
```

## 评测协议

训练或 API 推理完成后，统一保存预测 JSONL：

```json
{
  "id": "...",
  "dataset": "gsm8k",
  "question": "...",
  "answer": "42",
  "prediction": "We compute ... #### 42",
  "difficulty_group": "easy"
}
```

评测命令：

```powershell
python -m src.eval.evaluate_accuracy --input outputs\reports\preds.jsonl --output outputs\reports\eval.json
python -m src.eval.error_classifier --input outputs\reports\preds.jsonl --output outputs\reports\errors.jsonl
```

核心指标：

- `accuracy`：整体最终答案准确率。
- `by_group`：按 `easy/medium/hard` 分组的准确率。
- `error_counts`：错误类型分布。

后续论文实验中，`test` 集只报告最终结果；方法选择和错误重放应只使用 `train/dev` 信息。
