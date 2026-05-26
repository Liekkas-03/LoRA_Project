# 历史探索结果归档

本文档保留此前导出的 QLoRA 与 AdaQLoRA 探索实验结果，方便回溯。注意：这些结果来自项目重构前的旧流程，不作为 MATH-only 主线的最终论文结果。新的正式结果需要按 [autodl_steps.md](autodl_steps.md) 重新在 MATH train/test 上跑。

## 1. 已保留的结果包

根目录下保留以下文件：

```text
qlora_math_formal_adapter.tar.gz
qlora_math_formal_eval.tar.gz
adaqlora_math_formal_adapter.tar.gz
adaqlora_math_formal_eval.tar.gz
```

这些文件没有提交到 Git，因为 `.gitignore` 会忽略大体积训练产物。

## 2. 历史实验设置

| 项目 | 内容 |
|---|---|
| 基座模型 | Qwen2.5-Math-7B-Instruct |
| QLoRA 训练样本数 | 6725 |
| QLoRA 验证样本数 | 748 |
| AdaQLoRA 训练样本数 | 6725 |
| AdaQLoRA 验证样本数 | 748 |
| 训练轮数 | 2 epochs |
| 训练步数 | 842 steps |

说明：`6725 / 748` 是旧流程日志中的数据量，不是新的 MATH-only 官方 train/test 设置。

## 3. 历史准确率

| 方法 | Correct / Total | Accuracy |
|---|---:|---:|
| QLoRA baseline | 514 / 748 | 68.72% |
| AdaQLoRA | 551 / 748 | 73.66% |

历史提升：

- 绝对准确率提升：`+4.95` 个百分点
- 额外答对题数：`+37`

## 4. 历史分组结果

| 难度 | QLoRA baseline | AdaQLoRA | 变化 |
|---|---:|---:|---:|
| easy | 13/17 = 76.47% | 12/17 = 70.59% | -1 题 |
| medium | 501/730 = 68.63% | 539/730 = 73.84% | +38 题 |
| hard | 0/1 = 0.00% | 0/1 = 0.00% | 0 题 |

旧流程中的 hard 样本只有 1 条，说明当时的难度划分不适合作为论文分难度结论。MATH-only 重构后将改用 MATH 官方 `Level 1-5`。

## 5. 历史训练报告

| 指标 | QLoRA baseline | AdaQLoRA |
|---|---:|---:|
| `num_train_epochs` | 2 | 2 |
| `max_steps` | 842 | 842 |
| `global_step` | 842 | 842 |
| `step=842 eval_loss` | 0.6251 | 0.6204 |
| `step=842 eval_mean_token_accuracy` | 0.8436 | 0.8449 |
| `total_flos` | 1.068e+17 | 1.070e+17 |

## 6. 如何解释这些结果

可以对外说：

> 项目早期探索实验显示，分层 rank 的 AdaQLoRA 相比统一 rank 的 QLoRA 有初步收益。但旧实验的难度划分不够合理，因此后续主线已经切换到 MATH-only，并采用 MATH 官方 Level 1-5 作为难度依据重新评测。

不要把这组结果写成最终论文结论。它们适合放在实验记录或阶段性汇报中，证明训练与评测闭环已经跑通。
