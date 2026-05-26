# 面向 MATH 官方难度的 AdaQLoRA-Math 研究方案

## 1. 研究题目

中文题目：

《面向数学推理的难度感知自适应 QLoRA 微调研究》

英文题目：

Difficulty-Aware Adaptive QLoRA for Mathematical Reasoning

## 2. 一句话概述

本研究基于 MATH 数据集官方 `Level 1-5` 难度标签，探索如何通过分层 LoRA rank 分配提升 Qwen2.5-Math-7B 在复杂数学推理任务上的微调效果，并系统分析不同难度样本上的准确率、训练成本和 adapter 效率。

## 3. 为什么只用 MATH

MATH 自带官方难度等级和题型标签，适合支撑“难度感知”这个科研叙事。相比之下，GSM8K 没有官方逐题难度标签，如果混入主实验，会让难度划分依据变得不够干净。

本项目后续主实验只使用 MATH：

- 训练集：MATH train，约 7500 题
- 验证集：从 MATH train 按难度分层划分 10%
- 测试集：MATH test，约 5000 题
- 难度：`easy = Level 1-2`，`medium = Level 3`，`hard = Level 4-5`

## 4. 核心方法

### 4.1 QLoRA-Math baseline

baseline 使用统一 rank：

```text
所有目标模块 r = 16
```

训练配置使用 4-bit NF4 量化，目标模块包括：

```text
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

### 4.2 AdaQLoRA-Math

AdaQLoRA-Math 将统一 rank 改成分层 rank pattern。直觉是：低层更偏局部词法和浅层表示，中后层更承担长链路推理、关系整合和中间状态变换，因此可以把更多 LoRA 容量分配给中后层。

当前第一版使用静态分层：

```text
低层：layers 0-3，r = 8
中层：layers 4-17，r = 16
高层：layers 18-27，r = 32
```

这不是完整动态 AdaLoRA，而是一个可复现、容易消融的 difficulty-aware rank allocation 起点。

## 5. 实验设计

### 5.1 主实验

| 方法 | 数据集 | 训练集 | 评测集 | 主要指标 |
|---|---|---:|---:|---|
| QLoRA-Math | MATH | train split | MATH test | Accuracy / 分难度 Accuracy |
| AdaQLoRA-Math | MATH | train split | MATH test | Accuracy / 分难度 Accuracy |

### 5.2 分难度评测

| 分组 | MATH level |
|---|---|
| easy | Level 1-2 |
| medium | Level 3 |
| hard | Level 4-5 |

正式报告中同时保留 Level 1/2/3/4/5 细粒度结果，避免三档合并掩盖真实变化。

### 5.3 必做消融

- `Uniform r=8`
- `Uniform r=16`
- `Uniform r=32`
- `AdaQLoRA static rank`
- `参数量公平 AdaQLoRA`
- 不同随机种子复现实验

其中“参数量公平 AdaQLoRA”非常关键，用于证明提升不是单纯来自更多可训练参数。

## 6. 评测闭环

训练 loss 不能直接代表最终效果。正式评测流程是：

1. 训练 adapter。
2. 加载 base model + adapter。
3. 对 MATH dev/test 生成 `prediction`。
4. 从 `prediction` 中抽取最终答案。
5. 与数据集标准 `answer` 比较。
6. 输出整体准确率和分难度准确率。

相关文件：

- `src/infer/generate_answers.py`
- `src/eval/extract_answer.py`
- `src/eval/math_verifier.py`
- `src/eval/evaluate_accuracy.py`

## 7. 论文贡献写法

第一版可以写成三点贡献：

1. 提出一种面向 MATH 官方难度等级的 AdaQLoRA-Math 框架，将统一 LoRA rank 改为分层自适应 rank pattern。
2. 构建一套基于 MATH Level 1-5 的分难度评测体系，报告整体准确率、分 level 准确率和 easy/medium/hard 准确率。
3. 在单卡云 GPU 条件下比较 QLoRA 与 AdaQLoRA 的准确率、训练时间、显存占用和 adapter 大小，分析效果-成本权衡。

## 8. 当前阶段

当前阶段是 **MATH-only 重构阶段**：清理 GSM8K 混合流程，统一数据协议、配置文件和 AutoDL 运行文档，为后续正式 MATH 实验做准备。

## 9. 后续路线

1. 在 AutoDL 上重新准备 MATH-only 数据。
2. 跑通 QLoRA-Math smoke run。
3. 跑 QLoRA-Math 正式 baseline。
4. 跑 AdaQLoRA-Math 静态分层 rank 实验。
5. 加入参数量公平消融。
6. 在 MATH test 上生成预测并评测。
7. 整理表格，判断是否继续做 curriculum/replay。
