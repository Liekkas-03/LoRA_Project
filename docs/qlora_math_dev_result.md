# QLoRA-Math 与 AdaQLoRA-Math Dev 实验记录

## 1. 实验目的

本实验用于验证当前 `MATH-only` 主线是否已经跑通完整后训练闭环，即：

1. 准备 MATH 数据。
2. 使用 `Qwen2.5-Math-7B-Instruct` 进行 QLoRA 微调。
3. 训练完成后单独生成 dev 集预测结果。
4. 通过答案抽取与等价判断脚本计算最终准确率。

这次实验的定位是：

- 作为后续 `AdaQLoRA-Math` 的正式 baseline。
- 验证当前数据处理、训练、推理、评测流程是否可复现。

## 2. 实验设置

### 2.1 模型与方法

- 基座模型：`Qwen2.5-Math-7B-Instruct`
- 微调方法：`QLoRA`
- 量化方式：`4-bit NF4`
- LoRA rank：`r = 16`

### 2.2 数据设置

- 数据集：MATH
- 难度标签：使用 MATH 官方 `Level 1-5`
- 三档划分：
  - `easy = Level 1-2`
  - `medium = Level 3`
  - `hard = Level 4-5`
- 训练集：`math_train_split.jsonl`，共 `6749` 条
- dev 集：`math_dev.jsonl`，共 `749` 条

当前训练集各难度题量如下：

| 难度 | 题目数量 |
|---|---:|
| easy | 1721 |
| medium | 1433 |
| hard | 3595 |
| total | 6749 |

当前 dev 集各难度题量如下：

| 难度 | 题目数量 |
|---|---:|
| easy | 191 |
| medium | 159 |
| hard | 399 |
| total | 749 |

说明：

- 当前 dev 集是从 MATH train 中分层划分出来的验证集，不是最终 test 集。
- 因为 dev 来自 train 划分，所以预测文件中的 `split` 字段仍可能显示为 `train`，这是正常现象，不影响评测。

### 2.3 训练配置

- `max_seq_length = 1024`
- `per_device_train_batch_size = 1`
- `gradient_accumulation_steps = 16`
- `num_train_epochs = 2`
- `learning_rate = 2e-4`
- `bf16 = true`
- `gradient_checkpointing = true`

### 2.4 关于 eval 的处理

原始正式版在中途 `eval_steps = 500` 时会触发显存不足，因此改为：

- 关闭中途 step-based eval
- 保留训练结束后的最终 eval
- 训练完成后单独执行生成式评测

这样更适合 `RTX 4090 24GB` 的显存条件。

## 3. 实验流程

### 3.1 训练

在 AutoDL 上使用 `configs/qlora_math_formal_noeval.yaml` 训练 QLoRA baseline。

训练完成后输出目录为：

```text
outputs/adapters/qlora_math_formal_noeval
```

### 3.2 生成预测

训练完成后，加载：

- base model：`/root/models/Qwen/Qwen2.5-Math-7B-Instruct`
- adapter：`outputs/adapters/qlora_math_formal_noeval`

对 `math_dev.jsonl` 逐条生成 `prediction` 字段。

### 3.3 计算准确率

评测流程为：

1. 从 `prediction` 中抽取最终答案。
2. 从标准 `answer` 中抽取参考答案。
3. 做字符串标准化、数值等价判断。
4. 输出整体准确率与分难度准确率。

相关脚本：

- [generate_answers.py](/D:/LoRA_Project/src/infer/generate_answers.py)
- [extract_answer.py](/D:/LoRA_Project/src/eval/extract_answer.py)
- [math_verifier.py](/D:/LoRA_Project/src/eval/math_verifier.py)
- [evaluate_accuracy.py](/D:/LoRA_Project/src/eval/evaluate_accuracy.py)

## 4. 训练结果

以下结果来自导出的结果包 `qlora_math_dev_eval.tar.gz` 中的训练日志与评测文件。

### 4.1 训练日志摘要

- 训练轮数：`2`
- 总训练步数：`844`
- 训练总时长：`4560 s`，约 `76 分钟`
- 训练损失：`0.5048`
- 最终 eval loss：`0.5473`
- 最终 eval mean token accuracy：`0.8487`
- 最终 eval runtime：`100.2 s`

这说明：

- 训练过程完整跑通。
- 没有在中途因为显存问题中断。
- 最终阶段性 token-level 指标稳定。

## 5. Dev 集最终准确率

### 5.1 整体结果

| 指标 | 数值 |
|---|---:|
| total | 749 |
| correct | 286 |
| accuracy | 38.18% |
| missing_prediction | 1 |
| missing_reference | 12 |

### 5.2 分难度结果

| 难度 | Correct / Total | Accuracy |
|---|---:|---:|
| easy | 110 / 191 | 57.59% |
| medium | 69 / 159 | 43.40% |
| hard | 107 / 399 | 26.82% |

### 5.3 结果解读

当前 baseline 已经呈现出清晰的难度分层：

```text
easy > medium > hard
```

这和数学推理任务的预期一致，说明：

- 官方 `Level 1-5` 难度映射是有效的；
- 当前 baseline 在高难题上仍有明显提升空间；
- 后续 `AdaQLoRA-Math` 有明确的改进目标，尤其是提升 `hard` 子集表现。

## 6. 样例观察

从预测文件中可以看到几类典型现象：

### 6.1 推理基本正确

有些题模型能够完整写出关键推导，并得到正确答案。例如几何比值、等比数列关系等基础到中等难度题。

### 6.2 组合计数类容易翻倍或漏除

例如“两本不同类别书的配对数”这类题，模型给出了 `54`，而正确答案是 `27`。这说明：

- 模型能识别组合结构；
- 但在“是否需要除以 2 去重”这一类细节上仍容易出错。

### 6.3 部分题会复述题面而非真正作答

例如个别几何题，模型输出更像把题目重说一遍，而没有稳定地给出最终答案。这会直接拉低最终准确率，也是后续可重点改进的问题。

## 7. 当前结论

这次实验可以得出三个直接结论：

1. 当前 `MATH-only + QLoRA` baseline 已经跑通完整实验闭环。
2. 在 dev 集上，baseline 最终准确率为 `38.18%`。
3. `hard` 难度仅有 `26.82%`，说明后续 `AdaQLoRA-Math` 的改进重点应放在高难样本。

## 8. Baseline 阶段后续建议（历史记录）

下一步建议按下面顺序继续：

1. 以当前 baseline 为对照，运行 `AdaQLoRA-Math` 正式实验。
2. 在同一 dev 集上比较整体准确率与分难度准确率。
3. 重点观察 `hard` 子集是否提升。
4. 如有提升，再决定是否继续跑正式 test 集。

## 9. 本次结果文件

本次 dev baseline 的结果来源于：

```text
qlora_math_dev_eval.tar.gz
```

其中包含：

- `outputs/logs/qlora_math_formal_noeval.log`
- `outputs/reports/math_dev_qlora_preds.jsonl`
- `outputs/reports/math_dev_qlora_metrics.json`

## 10. AdaQLoRA-Math 对照实验

### 10.1 实验目的

在保持相同基座模型、相同训练集、相同 dev 集、相同评测脚本的条件下，验证当前 `AdaQLoRA-Math` 是否相对 `QLoRA-Math` baseline 带来稳定提升。

### 10.2 与 baseline 的主要区别

本次 `AdaQLoRA-Math` 不是更换数据集，也不是更换基座模型，而是在 `QLoRA` 的基础上加入了难度感知的分层 rank 分配。

具体来说：

- 仍使用 `Qwen2.5-Math-7B-Instruct`
- 仍使用同一份 `math_train_split.jsonl` 与 `math_dev.jsonl`
- 仍使用相同的答案抽取与准确率评测脚本
- 区别在于：对不同 Transformer 层分配不同的 LoRA rank，而不是全层统一使用 `r = 16`

这一改动对应当前方法设计中的 `difficulty-aware rank allocation`。

### 10.3 训练配置与输出

- 配置文件：`configs/adaqlora_math_formal_noeval.yaml`
- adapter 输出目录：`outputs/adapters/adaqlora_math_formal_noeval`
- rank pattern 输出文件：`outputs/reports/rank_pattern_formal_noeval.json`

训练设置与 baseline 基本保持一致，包括：

- `max_seq_length = 1024`
- `per_device_train_batch_size = 1`
- `gradient_accumulation_steps = 16`
- `num_train_epochs = 2`
- `learning_rate = 2e-4`
- `bf16 = true`
- 训练过程中不做中途 step-based eval，仅保留最终 eval

### 10.4 AdaQLoRA 训练日志摘要

以下结果来自导出的结果包 `ada_dev_eval.tar.gz` 中的训练日志与评测文件。

| 指标 | AdaQLoRA |
|---|---:|
| 训练轮数 | 2 |
| 总训练步数 | 844 |
| 训练总时长 | 5309 s |
| 训练总时长（约） | 88.5 分钟 |
| train loss | 0.4952 |
| final eval loss | 0.5449 |
| final eval mean token accuracy | 0.8492 |
| final eval runtime | 99.16 s |

从 token-level 指标看，`AdaQLoRA-Math` 相比 baseline 有轻微但稳定的优化。

## 11. AdaQLoRA Dev 集最终准确率

### 11.1 整体结果

| 指标 | 数值 |
|---|---:|
| total | 749 |
| correct | 307 |
| accuracy | 40.99% |
| missing_prediction | 0 |
| missing_reference | 12 |

### 11.2 分难度结果

| 难度 | Correct / Total | Accuracy |
|---|---:|---:|
| easy | 117 / 191 | 61.26% |
| medium | 74 / 159 | 46.54% |
| hard | 116 / 399 | 29.07% |

## 12. 与 QLoRA Baseline 的对比

### 12.1 总体对比

| 方法 | Correct / Total | Accuracy |
|---|---:|---:|
| QLoRA-Math | 286 / 749 | 38.18% |
| AdaQLoRA-Math | 307 / 749 | 40.99% |
| 提升 | +21 题 | +2.80 个百分点 |

### 12.2 分难度对比

| 难度 | QLoRA | AdaQLoRA | 提升 |
|---|---:|---:|---:|
| easy | 57.59% | 61.26% | +3.66 个百分点 |
| medium | 43.40% | 46.54% | +3.14 个百分点 |
| hard | 26.82% | 29.07% | +2.26 个百分点 |

### 12.3 结果解读

这次对照实验说明：

1. 当前 `AdaQLoRA-Math` 已经在同一实验设置下稳定超过 `QLoRA-Math` baseline。
2. 提升不是只出现在简单题上，`medium` 和 `hard` 子集也同步提升。
3. `hard` 从 `26.82%` 提升到 `29.07%`，说明难度感知分层 rank 分配对复杂数学推理题有帮助。

虽然当前提升幅度还不算特别大，但已经足够说明方法方向是成立的，后续可以继续通过更细的 rank 分配和消融实验来增强论文说服力。

## 13. 当前阶段结论

到目前为止，可以得出以下阶段性结论：

1. `MATH-only` 主线已经跑通了完整的训练、生成、评测闭环。
2. `QLoRA-Math` baseline 在 dev 集上达到 `38.18%`。
3. `AdaQLoRA-Math` 在相同条件下达到 `40.99%`，相对 baseline 提升 `+2.80` 个百分点。
4. 当前方法在 `easy / medium / hard` 三个难度组上均有提升，说明这不是偶然波动，而是有一致性的改进信号。

## 14. 下一步建议

当前最值得继续做的，不是立刻结束，而是把这条主线补成论文实验：

1. 做 rank 分配消融实验，验证提升是否确实来自分层 rank 设计。
2. 做参数量公平对比，避免因为 adapter 容量变化造成结论不稳。
3. 继续分析 `hard` 子集错误类型，观察改进是否集中在多步推理、代数变形、组合计数等题型。
4. 在 dev 上把方法调稳后，再决定是否跑正式 test 集。

## 15. AdaQLoRA 结果文件

本次 `AdaQLoRA-Math` dev 对照实验的结果来源于：

```text
ada_dev_eval.tar.gz
```

其中包含：

- `outputs/reports/math_dev_ada_metrics.json`
- `outputs/reports/math_dev_ada_preds.jsonl`
- `outputs/logs/adaqlora_math_formal_noeval.log`
- `outputs/reports/rank_pattern_formal_noeval.json`
