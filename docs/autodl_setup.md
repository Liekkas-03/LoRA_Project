# AutoDL 云端运行说明

本项目本地只负责轻量开发，GPU 训练放到 AutoDL 或类似云平台。以下命令默认在 Linux 云主机中执行。

## 1. 选择镜像

建议选择已经包含 CUDA、PyTorch、Python 3.10/3.11 的深度学习镜像。这样可以减少本项目需要额外安装的大包数量。

推荐硬件：

- 单卡 RTX 5090 32GB 或同级 24GB/32GB/48GB 显存 GPU。
- 系统盘建议不低于 50GB。
- 数据盘建议不低于 100GB，模型和缓存会占用较多空间。

## 2. 上传项目

方式一：使用 Git 同步项目。

```bash
git clone <your-repo-url> LoRA_Project
cd LoRA_Project
```

方式二：本地打包上传到 AutoDL，再解压。

```bash
cd LoRA_Project
```

## 3. 安装云端依赖

在 AutoDL 上执行：

```bash
pip install -r requirements-gpu.txt
```

如果镜像中已经安装了 PyTorch，不建议重复安装 PyTorch。优先使用镜像自带版本。

## 4. 准备数据

第一阶段建议用 Hugging Face datasets 下载 GSM8K；MATH 可从官方仓库或 Hugging Face 镜像准备。数据处理脚本后续会统一输出 JSONL。

推荐统一样本格式：

```json
{"id":"gsm8k-train-000001","dataset":"gsm8k","question":"...","answer":"...","solution":"...","difficulty_group":"easy"}
```

## 5. 运行训练

QLoRA baseline：

```bash
bash scripts/autodl_train.sh configs/qlora_math.yaml
```

AdaQLoRA-Math：

```bash
bash scripts/autodl_train.sh configs/adaqlora_math.yaml
```

## 6. 输出位置

建议统一输出到：

```text
outputs/
  adapters/
  checkpoints/
  logs/
  reports/
```

## 7. 注意事项

- 大模型权重、训练 checkpoint 和 adapter 不建议提交到 Git。
- 每次实验要保留 config、日志、评测结果和随机种子。
- 如果云端磁盘紧张，优先保留 adapter、config 和 eval report，删除中间 checkpoint。
- 本地只同步代码、配置和小型结果文件。
