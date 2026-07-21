# QLoRA-Math 与 AdaQLoRA-Math Dev 实验报告

## 1. 实验目标

本实验围绕 `MATH-only` 主线展开，目标是：

1. 在统一基座模型下构建 `QLoRA-Math` baseline。
2. 在相同训练集、相同验证集、相同生成设置下运行 `AdaQLoRA-Math`。
3. 使用更接近主流论文做法的 `Math-Verify` 作为统一评测标准，对两组结果进行最终答案准确率比较。

当前报告中的准确率结果，均以 `Math-Verify` 为准；旧的简化评测结果不再作为正式结论保留。

## 2. 实验设置

### 2.1 模型与方法

- 基座模型：`Qwen2.5-Math-7B-Instruct`
- baseline：`QLoRA-Math`
- 改进方法：`AdaQLoRA-Math`
- 量化方式：`4-bit NF4`

其中：

- `QLoRA-Math` 使用统一 LoRA rank
- `AdaQLoRA-Math` 使用分层 rank 分配，即低层、中层、高层使用不同 rank

### 2.2 数据集设置

- 数据集：`MATH`
- 难度来源：MATH 官方 `Level 1-5`
- 难度分组：
  - `easy = Level 1-2`
  - `medium = Level 3`
  - `hard = Level 4-5`

训练集与验证集规模如下：

| 数据划分 | easy | medium | hard | total |
|---|---:|---:|---:|---:|
| train | 1721 | 1433 | 3595 | 6749 |
| dev | 191 | 159 | 399 | 749 |

说明：

- 当前 `dev` 集由 `MATH train` 分层划分得到，用于方法开发与对照实验。
- 最终正式论文结果仍建议在 `MATH test` 上补充报告。

### 2.3 训练配置

两组实验保持同一训练主设置：

- `max_seq_length = 1024`
- `per_device_train_batch_size = 1`
- `gradient_accumulation_steps = 16`
- `num_train_epochs = 2`
- `learning_rate = 2e-4`
- `bf16 = true`
- `gradient_checkpointing = true`

为了适配 `RTX 4090 24GB` 显存，训练时不做中途 step-based eval，仅保留训练结束后的最终 eval。

## 3. 评测标准

### 3.1 统一评测后端

本报告采用：

- 生成脚本：[generate_answers.py](/D:/LoRA_Project/src/infer/generate_answers.py)
- 评测入口：[evaluate_accuracy.py](/D:/LoRA_Project/src/eval/evaluate_accuracy.py)
- 判题后端：[math_verifier.py](/D:/LoRA_Project/src/eval/math_verifier.py)
- 评测模式：`--backend math_verify`

即使用 `Math-Verify` 对模型输出与标准答案进行最终答案抽取与等价判断。

### 3.2 采用 Math-Verify 的原因

此前项目中的简化评测脚本对部分符号答案、分数形式、LaTeX 表达式处理不够稳健，容易低估模型表现。  
本次报告改用 `Math-Verify` 后，评测方式更接近当前数学推理论文中的主流做法，因此作为当前正式实验记录的唯一标准。

## 4. QLoRA-Math Baseline 结果

### 4.1 训练日志摘要

| 指标 | 数值 |
|---|---:|
| 训练轮数 | 2 |
| 总训练步数 | 844 |
| train runtime | 4560 s |
| 训练时长（约） | 76 分钟 |
| train loss | 0.5048 |
| final eval loss | 0.5473 |
| final eval mean token accuracy | 0.8487 |

### 4.2 Dev 集最终准确率

| 指标 | 数值 |
|---|---:|
| backend | math_verify |
| total | 749 |
| correct | 359 |
| accuracy | 47.93% |
| missing_prediction | 3 |
| missing_reference | 44 |

### 4.3 分难度结果

| 难度 | Correct / Total | Accuracy |
|---|---:|---:|
| easy | 135 / 191 | 70.68% |
| medium | 95 / 159 | 59.75% |
| hard | 129 / 399 | 32.33% |

## 5. AdaQLoRA-Math 结果

### 5.1 方法区别

`AdaQLoRA-Math` 与 baseline 的差别不在于更换数据或更换模型，而在于：

- 保持同一基座模型
- 保持同一训练集与 dev 集
- 保持相同生成脚本与相同评测标准
- 仅在 LoRA 结构上加入分层 rank 分配

当前实现中，高层分配更高 rank，低层分配较低 rank，用于增强复杂数学推理相关层的适配能力。

### 5.2 训练日志摘要

| 指标 | 数值 |
|---|---:|
| 训练轮数 | 2 |
| 总训练步数 | 844 |
| train runtime | 5309 s |
| 训练时长（约） | 88.5 分钟 |
| train loss | 0.4952 |
| final eval loss | 0.5449 |
| final eval mean token accuracy | 0.8492 |

### 5.3 Dev 集最终准确率

| 指标 | 数值 |
|---|---:|
| backend | math_verify |
| total | 749 |
| correct | 381 |
| accuracy | 50.87% |
| missing_prediction | 3 |
| missing_reference | 44 |

### 5.4 分难度结果

| 难度 | Correct / Total | Accuracy |
|---|---:|---:|
| easy | 144 / 191 | 75.39% |
| medium | 96 / 159 | 60.38% |
| hard | 141 / 399 | 35.34% |

## 6. QLoRA 与 AdaQLoRA 对比

### 6.1 总体对比

| 方法 | Correct / Total | Accuracy |
|---|---:|---:|
| QLoRA-Math | 359 / 749 | 47.93% |
| AdaQLoRA-Math | 381 / 749 | 50.87% |
| 提升 | +22 题 | +2.94 个百分点 |

### 6.2 分难度对比

| 难度 | QLoRA | AdaQLoRA | 提升 |
|---|---:|---:|---:|
| easy | 70.68% | 75.39% | +4.71 个百分点 |
| medium | 59.75% | 60.38% | +0.63 个百分点 |
| hard | 32.33% | 35.34% | +3.01 个百分点 |


low_layers: 0-3
mid_layers: 4-17
high_layers: 18-27

low: 8
mid: 16
high: 32


### 6.3 结果解读

当前对照实验说明：

1. `AdaQLoRA-Math` 在统一评测标准下稳定超过 `QLoRA-Math` baseline。
2. 提升不只出现在简单题上，`hard` 子集同样获得了明显增益。
3. `hard` 从 `32.33%` 提升到 `35.34%`，说明当前分层 rank 分配设计对复杂数学推理题是有帮助的。

## 7. 当前阶段结论

截至当前阶段，可以得出以下结论：

1. `MATH-only` 主线已经跑通完整训练、生成、评测闭环。
2. 在 `Math-Verify` 标准下，`QLoRA-Math` 在当前 `dev` 集上的最终准确率为 `47.93%`。
3. 在相同设置下，`AdaQLoRA-Math` 达到 `50.87%`，相对 baseline 提升 `+2.94` 个百分点。
4. 当前改进在 `easy / medium / hard` 三个难度组上均有提升，其中 `hard` 提升最具论文价值。

## 8. 结果文件

本报告使用的主要结果文件包括：

### 8.1 训练日志

- `outputs/logs/qlora_math_formal_noeval.log`
- `outputs/logs/adaqlora_math_formal_noeval.log`

### 8.2 生成结果

- `outputs/reports/math_dev_qlora_preds.jsonl`
- `outputs/reports/math_dev_ada_preds.jsonl`

### 8.3 Math-Verify 评测结果

- `outputs/reports/math_dev_qlora_metrics_math_verify.json`
- `outputs/reports/math_dev_ada_metrics_math_verify.json`

对应导出包为：

```text
qlora_math_dev_eval.tar.gz
ada_dev_eval.tar.gz
math_verify_dev_eval.tar.gz
```

## 9. 下一步建议

当前最值得继续做的实验有两类：

1. 做 rank 分配消融实验，验证提升是否确实来自分层 rank 设计。
2. 检查 `missing_reference = 44` 的样本，进一步确认标准答案解析失败是否还可继续优化。

在此基础上，再决定是否把当前最佳方案扩展到 `MATH test` 作为正式论文主结果。

下一步：
Fair-Ada：低层 8，中层 12，高层 24。平均 rank 约 15.7，和 baseline 16 基本公平。
Reverse-Ada：低层 24，中层 12，高层 8。总预算和 Fair-Ada 一样，但方向反过来，用来验证“高层给更高 rank”是不是关键