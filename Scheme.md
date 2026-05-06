# 面向数学推理的难度感知自适应 QLoRA 微调研究方案

版本：v0.1  
日期：2026-05-06  
硬件假设：单卡 NVIDIA RTX 5090 32GB  
项目定位：科研优先，同时保留可转化为数学解题辅导系统的工程闭环

## 1. 题目

中文题目：

《面向数学推理的难度感知自适应 QLoRA 微调研究》

英文题目：

Difficulty-Aware Adaptive QLoRA for Mathematical Reasoning

可选方法名：

DA-QLoRA：Difficulty-Aware Adaptive QLoRA

## 2. 一句话概括

本项目面向数学推理任务，研究如何在固定显存和参数预算下，根据样本难度、错误类型和模型层敏感性动态分配 LoRA rank，并结合课程学习与验证器驱动的错误重放，提升大模型在数学推理任务上的准确率、稳定性和训练效率。最终将该方法转化为一个可演示的数学解题辅导系统。

## 3. 研究动机

LoRA/QLoRA 已经成为大模型低成本微调的主流方法，但常见做法通常对所有目标层使用相同 rank 和 alpha。数学推理任务中，不同样本的推理难度差异明显，不同错误类型对模型能力的要求也不同。例如，简单算术题主要依赖基础计算和格式稳定性，而复杂代数、组合、几何题往往需要更长链路的推理、符号变换和条件跟踪。

因此，一个自然问题是：在固定可训练参数预算下，是否可以把 LoRA 的容量更有针对性地分配到更需要适配的层和样本上，从而比统一 rank 的 QLoRA 获得更好的效果-成本比？

本项目不追求从零发明全新的低秩分解形式，而是在 PEFT 框架可实现的范围内，设计一种任务感知、难度感知、错误反馈驱动的自适应 LoRA 微调框架，使研究方案更容易复现、消融和落地。

## 4. 研究目标

1. 提出一种面向数学推理的难度感知自适应 LoRA rank 分配策略。
2. 设计课程学习和错误重放机制，使训练过程更关注高价值难样本。
3. 构建一套数学推理微调评测体系，覆盖准确率、分难度表现、错误类型、训练成本和推理成本。
4. 在单卡 RTX 5090 条件下完成可复现实验，形成论文初稿和项目仓库。
5. 将科研结果转化为数学解题辅导 demo，展示分步推理、答案校验和易错类型分析。

## 5. 核心研究问题

RQ1：统一 rank 的 LoRA/QLoRA 是否会在数学推理任务中造成参数预算浪费？

RQ2：根据题目难度和模型层敏感性进行 rank 分配，能否在相同可训练参数预算下提升准确率？

RQ3：课程学习是否能提升 QLoRA 在数学推理任务上的收敛速度和泛化能力？

RQ4：基于答案验证器的错误样本重放，是否能减少高频错误类型，例如算术错误、推理跳步、最终答案格式错误？

RQ5：相比 LoRA、QLoRA、rsLoRA、DoRA，所提出方法在准确率、显存、训练时间、推理延迟和 adapter 大小上是否具有更好的效果-成本比？

## 6. 数据集

### 6.1 主数据集

GSM8K：

- 类型：小学到初中难度的英文数学文字题。
- 规模：约 8.5K 道题，官方划分为训练集和测试集。
- 优点：格式简单，评测稳定，适合作为第一阶段实验和快速迭代。
- 官方仓库：https://github.com/openai/grade-school-math

MATH：

- 类型：竞赛风格数学题，覆盖代数、数论、几何、组合、概率、预备微积分等。
- 难度：显著高于 GSM8K，适合验证方法在复杂推理上的收益。
- 优点：自带学科类别和难度标签，特别适合做 difficulty-aware 方法。
- 官方仓库：https://github.com/hendrycks/math

### 6.2 可选扩展数据集

SVAMP：

- 用于检验模型是否真正掌握数学推理，而非记忆题型。

ASDiv：

- 可作为小学数学文字题的额外泛化测试集。

中文数学题自建小集：

- 从公开数学教材、习题或自建模板中构造 200-500 条中文样本。
- 用于展示中文数学辅导场景，不作为主论文结论的唯一依据。

## 7. 基座模型选择

### 7.1 主模型

Qwen2.5-Math-7B 或 Qwen2.5-Math-7B-Instruct：

- 优点：数学能力强，支持英文和中文数学问题，适合本项目主题。
- 建议：如果重点是监督微调，优先考虑 base 版本；如果重点是做 demo 和对话式辅导，使用 instruct 版本更省事。
- 模型页：https://huggingface.co/Qwen/Qwen2.5-Math-7B
- Instruct 模型页：https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct

### 7.2 对照模型

Qwen3-8B：

- 用作通用强基座对照，检验方法是否只对 math-specialized 模型有效。
- 模型页：https://huggingface.co/Qwen/Qwen3-8B

DeepSeekMath-7B：

- 可作为数学专用模型的额外对照。

### 7.3 单卡 5090 训练建议

主实验优先使用 7B/8B 模型。单卡 32GB 下，4-bit QLoRA 适合做多轮消融。14B 可作为后期扩展，不建议作为第一阶段主模型，否则会显著降低实验迭代速度。

## 8. 方法设计

本项目方法由三部分组成：

1. Difficulty-Aware Rank Allocation：难度感知 rank 分配。
2. Curriculum LoRA：课程式 QLoRA 训练。
3. Verifier-Guided Replay：验证器驱动的错误重放。

整体流程：

```text
数据集 -> 难度评分 -> 初始 QLoRA 探测训练 -> 层敏感性估计
     -> 生成 rank_pattern/alpha_pattern -> 课程学习训练
     -> 验证器评测 -> 错误类型归因 -> replay 微调
     -> 最终评测与效率分析 -> 数学辅导系统 demo
```

## 9. Difficulty-Aware Rank Allocation

### 9.1 样本难度评分

为每道题构造难度分数：

```text
D(x) = w1 * dataset_level + w2 * solution_steps + w3 * expression_complexity
     + w4 * category_difficulty + w5 * base_model_uncertainty
```

候选特征：

- 数据集来源：GSM8K 低于 MATH。
- MATH 官方难度标签：level 1-5。
- 解题步骤数：根据标准解答中的换行、推导句、公式数量估计。
- 表达式复杂度：数字数量、运算符数量、方程数量、分式/根号/组合符号数量。
- 题目类别：代数、数论、几何、组合等。
- 基座模型不确定性：同一题多次采样答案是否一致。
- 基座模型错误率：zero-shot 或 few-shot 下是否答错。

将样本划分为：

- Easy：低难度题。
- Medium：中等难度题。
- Hard：高难度题。

### 9.2 层敏感性估计

目标是找出哪些层或模块对困难数学推理更敏感。

可实现的估计方式：

- 先训练一个小步数 uniform QLoRA 探测模型。
- 在 Easy/Medium/Hard 三组验证样本上计算 loss。
- 记录不同层 LoRA 梯度范数或参数更新范数。
- 对 Hard 样本 loss 下降贡献更明显的层，分配更高 rank。

关注模块：

- Attention：`q_proj`、`k_proj`、`v_proj`、`o_proj`
- MLP：`gate_proj`、`up_proj`、`down_proj`

直觉假设：

- Attention 层更影响长链路条件跟踪和题干信息关联。
- MLP 层更影响隐式知识、运算模式和中间推理变换。
- 高难题可能更需要中后层和 MLP 模块的适配容量。

### 9.3 rank 分配策略

给定总 rank 预算，例如平均 rank 为 16，不增加总可训练参数。

Uniform LoRA：

```text
所有目标层 r = 16
```

DA-QLoRA：

```text
低敏感层 r = 8
中敏感层 r = 16
高敏感层 r = 32
```

约束条件：

```text
sum(params_lora_adaptive) ≈ sum(params_lora_uniform)
```

在 PEFT 中可通过 `rank_pattern` 和 `alpha_pattern` 实现分层 rank 与 scaling。PEFT 官方文档也支持 `rsLoRA`、`DoRA`、`LoftQ` 等 LoRA 变体，适合作为强 baseline。

参考：https://huggingface.co/docs/peft/main/developer_guides/lora

## 10. Curriculum LoRA

### 10.1 基本思想

直接混合所有样本训练时，模型可能过早被复杂题噪声干扰。课程学习先让模型稳定掌握基础推理格式和算术过程，再逐步增加复杂样本。

### 10.2 训练阶段

Stage 1：Easy Warm-up

- 数据：Easy 样本为主，少量 Medium 样本。
- 目标：稳定输出格式，学习基础分步推理。

Stage 2：Mixed Training

- 数据：Easy/Medium/Hard 混合。
- 目标：建立对多类型题目的泛化能力。

Stage 3：Hard Emphasis

- 数据：提高 Hard 样本比例，同时保留 Easy 样本防止退化。
- 目标：增强复杂题推理能力。

推荐采样比例：

| 阶段 | Easy | Medium | Hard |
|---|---:|---:|---:|
| Stage 1 | 70% | 30% | 0% |
| Stage 2 | 30% | 50% | 20% |
| Stage 3 | 15% | 35% | 50% |

### 10.3 对照设置

- Random：随机打乱所有样本。
- Easy-to-Hard：简单到困难。
- Hard-first：先训练困难样本。
- DA-Curriculum：本项目设计的动态采样策略。

## 11. Verifier-Guided Replay

### 11.1 答案验证器

数学任务的一个优势是最终答案可以被自动校验。

验证方式：

- 从模型输出中抽取最终答案。
- 支持 `\boxed{}`、`####`、最后一行答案等格式。
- 数值题用标准化字符串和数值容差判断。
- 符号题用 SymPy 化简后判断等价。
- 选择题或短答案题使用规则匹配。

### 11.2 错误类型

将错误样本分为：

| 错误类型 | 含义 |
|---|---|
| Format Error | 没有输出可解析最终答案 |
| Arithmetic Error | 推理过程大体正确，但计算错误 |
| Reasoning Error | 解题路径错误或条件理解错误 |
| Step Omission | 跳过关键推理步骤 |
| Overthinking | 输出过长，偏离题意 |
| Refusal/Invalid | 拒答、乱码、无效输出 |

### 11.3 错误重放策略

第一轮训练后，在 dev 集上运行模型并收集错误样本。

构造 replay 数据：

- 高频错误类型优先。
- Hard 样本优先。
- 模型高置信但错误的样本优先。
- 保留一定比例正确样本作为稳定项。

Replay 训练建议：

- 学习率降低到主训练的 30%-50%。
- epoch 控制在 0.2-0.5。
- replay 数据占总训练数据的 10%-30%。
- 避免只记忆 dev 集，可从训练集和 dev-like split 中采样。

## 12. Baseline 设计

至少包含以下 baseline：

| 编号 | 方法 | 说明 |
|---|---|---|
| B0 | Base zero-shot | 基座模型直接推理 |
| B1 | Base few-shot | 加 4-shot 或 8-shot 示例 |
| B2 | LoRA | BF16/FP16 LoRA，不量化或低量化 |
| B3 | QLoRA | 4-bit QLoRA，uniform rank |
| B4 | rsLoRA | rank-stabilized LoRA |
| B5 | DoRA | Weight-Decomposed LoRA |
| B6 | QLoRA + Curriculum | 只加入课程学习 |
| B7 | QLoRA + Replay | 只加入错误重放 |
| Ours | DA-QLoRA | 自适应 rank + curriculum + replay |

其中 B6/B7 是关键消融，用来证明提升不是来自单一训练技巧。

## 13. 实验设计

### 13.1 主实验

数据：

- Train：GSM8K train + MATH train。
- Dev：从 train 中划分 5%-10%。
- Test：GSM8K test + MATH test。

模型：

- Qwen2.5-Math-7B 或 Qwen2.5-Math-7B-Instruct。

比较：

| 方法 | GSM8K Acc | MATH Acc | Easy Acc | Medium Acc | Hard Acc | Trainable Params | Peak VRAM | Train Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot | - | - | - | - | - | 0 | - | 0 |
| Few-shot | - | - | - | - | - | 0 | - | 0 |
| QLoRA r=8 | - | - | - | - | - | - | - | - |
| QLoRA r=16 | - | - | - | - | - | - | - | - |
| rsLoRA r=16 | - | - | - | - | - | - | - | - |
| DoRA r=16 | - | - | - | - | - | - | - | - |
| Ours | - | - | - | - | - | - | - | - |

### 13.2 rank 分配消融

| rank 策略 | 平均 rank | 是否等参数预算 | GSM8K Acc | MATH Acc | Hard Acc |
|---|---:|---|---:|---:|---:|
| Uniform r=8 | 8 | 是 | - | - | - |
| Uniform r=16 | 16 | 是 | - | - | - |
| Uniform r=32 | 32 | 否 | - | - | - |
| Random rank pattern | 16 | 是 | - | - | - |
| Difficulty-aware rank pattern | 16 | 是 | - | - | - |

### 13.3 课程学习消融

| 训练顺序 | GSM8K Acc | MATH Acc | Hard Acc | 收敛步数 | 训练稳定性 |
|---|---:|---:|---:|---:|---|
| Random | - | - | - | - | - |
| Easy-to-Hard | - | - | - | - | - |
| Hard-first | - | - | - | - | - |
| DA-Curriculum | - | - | - | - | - |

### 13.4 replay 消融

| Replay 设置 | GSM8K Acc | MATH Acc | Arithmetic Error | Reasoning Error | Format Error |
|---|---:|---:|---:|---:|---:|
| No Replay | - | - | - | - | - |
| Random Replay | - | - | - | - | - |
| Hard-only Replay | - | - | - | - | - |
| Verifier-guided Replay | - | - | - | - | - |

### 13.5 效率分析

| 方法 | Adapter 大小 | 峰值显存 | 训练 tokens/s | 推理 tokens/s | 单题平均延迟 | 准确率 |
|---|---:|---:|---:|---:|---:|---:|
| QLoRA | - | - | - | - | - | - |
| rsLoRA | - | - | - | - | - | - |
| DoRA | - | - | - | - | - | - |
| Ours | - | - | - | - | - | - |

## 14. 评测指标

主指标：

- Accuracy：最终答案是否正确。
- GSM8K Accuracy。
- MATH Accuracy。
- Hard Accuracy：困难样本准确率。

辅助指标：

- Format Validity：最终答案可解析比例。
- Error Rate by Type：按错误类型统计。
- Self-consistency Accuracy：多次采样投票后的准确率，可作为扩展。
- Calibration：模型置信度和正确率是否匹配，可作为扩展。

效率指标：

- 可训练参数量。
- Adapter 文件大小。
- 峰值显存。
- 训练时间。
- 推理吞吐。
- 单题平均延迟。

## 15. 训练配置建议

### 15.1 QLoRA 配置

建议初始配置：

```yaml
model_name: Qwen/Qwen2.5-Math-7B
load_in_4bit: true
bnb_4bit_quant_type: nf4
bnb_4bit_compute_dtype: bfloat16
bnb_4bit_use_double_quant: true

lora:
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  bias: none
  task_type: CAUSAL_LM

training:
  max_seq_length: 2048
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  learning_rate: 2.0e-4
  num_train_epochs: 2
  warmup_ratio: 0.03
  lr_scheduler_type: cosine
  bf16: true
  gradient_checkpointing: true
  logging_steps: 10
  save_steps: 500
```

### 15.2 方法配置

DA-QLoRA 示例：

```yaml
adaptive_lora:
  budget_mode: equal_trainable_params
  rank_levels:
    low: 8
    mid: 16
    high: 32
  alpha_rule: alpha_equals_2r
  sensitivity_metric:
    - lora_grad_norm
    - hard_loss_delta
  modules:
    attention:
      - q_proj
      - k_proj
      - v_proj
      - o_proj
    mlp:
      - gate_proj
      - up_proj
      - down_proj

curriculum:
  stage_1_ratio: [0.70, 0.30, 0.00]
  stage_2_ratio: [0.30, 0.50, 0.20]
  stage_3_ratio: [0.15, 0.35, 0.50]

replay:
  enabled: true
  replay_ratio: 0.20
  lr_scale: 0.5
  max_replay_epochs: 0.5
  priority:
    - hard_wrong
    - high_confidence_wrong
    - arithmetic_error
    - reasoning_error
```

## 16. 推荐代码目录

```text
LoRA_Project/
  README.md
  configs/
    qlora_qwen_math.yaml
    da_qlora_qwen_math.yaml
  data/
    raw/
    processed/
    splits/
  src/
    data/
      prepare_gsm8k.py
      prepare_math.py
      difficulty_scorer.py
    lora/
      build_lora_config.py
      sensitivity_probe.py
      rank_allocator.py
    train/
      train_sft.py
      train_curriculum.py
      train_replay.py
    eval/
      extract_answer.py
      math_verifier.py
      evaluate_accuracy.py
      error_classifier.py
    demo/
      app.py
  outputs/
    checkpoints/
    adapters/
    logs/
    reports/
  docs/
    paper_outline.md
    experiment_plan.md
```

## 17. 论文贡献写法

建议贡献点写成三条：

1. 提出一种难度感知自适应 QLoRA 框架，根据数学样本难度和层敏感性，在固定参数预算下动态分配 LoRA rank。
2. 设计课程学习与验证器驱动错误重放策略，增强模型对复杂数学推理样本的学习效率和错误修复能力。
3. 构建覆盖准确率、错误类型和效率成本的评测体系，并在数学解题辅导场景中验证方法的应用价值。

## 18. 论文结构

```text
1. Introduction
   - 数学推理对大模型的重要性
   - LoRA/QLoRA 的低成本优势
   - 统一 rank 在复杂推理任务中的潜在不足
   - 本文贡献

2. Related Work
   - Mathematical Reasoning with LLMs
   - Parameter-Efficient Fine-Tuning
   - Curriculum Learning
   - Verifier-guided Training

3. Method
   - Problem Formulation
   - Difficulty Scoring
   - Layer Sensitivity Estimation
   - Adaptive Rank Allocation
   - Curriculum LoRA
   - Verifier-Guided Replay

4. Experiments
   - Datasets
   - Models and Baselines
   - Implementation Details
   - Main Results
   - Ablation Study
   - Efficiency Analysis

5. Analysis
   - Error Type Analysis
   - Rank Pattern Analysis
   - Case Study
   - Limitations

6. Application
   - 数学解题辅导系统
   - 答案校验与易错点提示

7. Conclusion
```

## 19. 项目落地：数学解题辅导系统

系统功能：

- 输入数学题。
- 输出分步推理过程。
- 输出最终答案。
- 自动校验最终答案格式。
- 给出错误类型提示，例如“计算错误”“推理跳步”“最终答案缺失”。
- 展示模型是否使用 DA-QLoRA adapter。

Demo 形态：

- 第一阶段：命令行推理脚本。
- 第二阶段：Gradio 或 FastAPI + 简单前端。
- 第三阶段：加入错题本功能，收集用户错题并映射到 replay 数据格式。

科研转化叙事：

本文提出的 DA-QLoRA 方法可作为数学教育场景中的低成本能力增强模块，使学校或培训机构能够在单卡环境下对开源数学模型进行个性化微调，并根据学生错题类型持续优化模型。

## 20. 8 周计划

| 周次 | 目标 | 产出 |
|---|---|---|
| 第 1 周 | 跑通 GSM8K/MATH 数据处理与 zero-shot/few-shot 评测 | 数据处理脚本、baseline 结果 |
| 第 2 周 | 跑通 QLoRA 微调 | 第一版 adapter、训练日志 |
| 第 3 周 | 实现难度评分器和分层评测 | difficulty split、分难度结果 |
| 第 4 周 | 实现 sensitivity probe 和 rank allocator | rank_pattern、参数预算统计 |
| 第 5 周 | 训练 DA-QLoRA 主模型 | 主方法结果 |
| 第 6 周 | 实现 curriculum 和 verifier-guided replay | 消融实验结果 |
| 第 7 周 | 做完整评测、效率分析和错误分析 | 实验表格、错误案例 |
| 第 8 周 | 写论文初稿和 demo | 论文初稿、README、演示系统 |

## 21. 风险与应对

风险 1：方法提升不明显。

应对：

- 优先看 Hard 子集提升，而不是只看整体平均准确率。
- 保证自适应 rank 与 uniform rank 的参数预算一致。
- 增加错误类型分析，展示方法在哪些能力上有效。

风险 2：MATH 太难，7B 模型提升有限。

应对：

- 先在 GSM8K 上验证方法有效性。
- MATH 按 level 1-3 和 level 4-5 分开报告。
- 后期可增加 Qwen3-8B 或更大模型作为扩展。

风险 3：错误分类不够准确。

应对：

- 第一版只做粗粒度错误分类：格式错误、答案错误、不可解析。
- 第二版再加入算术错误、推理错误、跳步错误。

风险 4：adaptive rank 的实现复杂。

应对：

- 第一版用人工规则 rank_pattern。
- 第二版加入 sensitivity probe。
- 第三版再做自动 rank allocator。

## 22. 最小可行版本

如果时间紧，最低限度完成：

1. GSM8K + MATH 数据处理。
2. Qwen2.5-Math-7B 的 QLoRA baseline。
3. 难度分层评测。
4. 手工 difficulty-aware rank_pattern。
5. Curriculum 和 replay 各做一个简单版本。
6. 主结果表 + 消融表 + 错误分析。

这样已经可以形成一篇完整技术报告或论文初稿。

## 23. 简历写法

项目名称：

面向数学推理的难度感知 QLoRA 微调系统

简历描述示例：

- 基于 Qwen2.5-Math-7B 和 QLoRA 构建数学推理微调框架，完成 GSM8K/MATH 数据处理、监督微调、自动答案校验和分难度评测闭环。
- 提出 difficulty-aware rank allocation 与 verifier-guided replay 策略，在固定可训练参数预算下优化 LoRA adapter 分配，并系统比较 LoRA、QLoRA、rsLoRA、DoRA 的准确率和效率成本。
- 搭建数学解题辅导 demo，支持分步推理生成、最终答案抽取、错误类型分析和错题重放数据构造。

## 24. 参考资料

- GSM8K 官方仓库：https://github.com/openai/grade-school-math
- MATH 官方仓库：https://github.com/hendrycks/math
- Qwen2.5-Math-7B：https://huggingface.co/Qwen/Qwen2.5-Math-7B
- Qwen2.5-Math-7B-Instruct：https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct
- Qwen3-8B：https://huggingface.co/Qwen/Qwen3-8B
- PEFT LoRA Guide：https://huggingface.co/docs/peft/main/developer_guides/lora
- TRL SFTTrainer：https://huggingface.co/docs/trl/main/sft_trainer
- bitsandbytes QLoRA 文档：https://huggingface.co/docs/bitsandbytes/main/en/index
