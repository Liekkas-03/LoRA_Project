# QLoRA Baseline 云端运行清单

本文档说明如何在 AutoDL 或类似云 GPU 平台上跑通第一版 `QLoRA-Math` baseline。本地仍然不安装 PyTorch、Transformers、PEFT、TRL 或 bitsandbytes。

## 目标

第一版 baseline 只做一件事：在统一后的 `train.jsonl/dev.jsonl` 上，用 `Qwen2.5-Math-7B` 跑通 4-bit QLoRA SFT，并保存 LoRA adapter。

跑通后我们会得到：

- `outputs/adapters/qlora_math/`：baseline adapter 和 tokenizer 文件。
- `outputs/logs/` 或 TensorBoard 日志：训练 loss、eval loss。
- 后续可接推理脚本生成 `prediction`，再进入准确率评测。

## 1. 云端环境

建议选择 AutoDL 中已经配好 Qwen 7B 的镜像。不要直接在系统盘默认 miniconda 环境里安装项目依赖，推荐在数据盘安装自己的 Miniconda，再把 PyTorch 和项目依赖都装到数据盘 conda 环境：

```bash
cd /root/autodl-tmp
mkdir -p /root/autodl-tmp/installers /root/autodl-tmp/pip_cache /root/autodl-tmp/hf_cache /root/autodl-tmp/conda_pkgs

if [ ! -d /root/autodl-tmp/miniconda3 ]; then
  wget -O /root/autodl-tmp/installers/miniconda.sh https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash /root/autodl-tmp/installers/miniconda.sh
fi

source /root/autodl-tmp/miniconda3/bin/activate
conda init
source ~/.bashrc
export CONDA_PKGS_DIRS=/root/autodl-tmp/conda_pkgs
if ! conda env list | grep -q "/root/autodl-tmp/miniconda3/envs/lora"; then
  conda create -n lora python=3.10 -y
fi
conda activate lora

export PIP_CACHE_DIR=/root/autodl-tmp/pip_cache
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com

python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

cd /root/autodl-tmp/LoRA_Project
pip install --no-cache-dir -r requirements-gpu.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

运行安装脚本时，前面一路按 Enter 或输入 yes；当它提示安装路径并默认显示 `/root/miniconda3` 时，必须改成 `/root/autodl-tmp/miniconda3`。这样 Miniconda、环境、PyTorch、项目依赖和缓存都会尽量留在 `/root/autodl-tmp`，减少系统盘占用。

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

## 8. 生成答案并评测准确率

训练只会保存 adapter 和训练日志。要得到数学题准确率，需要在云端加载 base model + adapter 生成预测：

```bash
cd /root/autodl-tmp/LoRA_Project

bash scripts/autodl_generate.sh \
  /root/Qwen2.5/text-generation-webui/models/Qwen2.5-7B-Instruct \
  outputs/adapters/qlora_math \
  data/processed/dev.jsonl \
  outputs/reports/dev_preds.jsonl \
  256
```

然后计算最终答案准确率：

```bash
python -m src.eval.evaluate_accuracy \
  --input outputs/reports/dev_preds.jsonl \
  --output outputs/reports/dev_metrics.json
```

`dev_preds.jsonl` 是模型逐题生成结果，`dev_metrics.json` 是总体和分难度准确率结果。

## 9. 导出训练结果到本地

训练结果不建议直接提交 GitHub。先在 AutoDL 上压缩 adapter：

```bash
cd /root/autodl-tmp/LoRA_Project
tar -czf qlora_math_adapter.tar.gz outputs/adapters/qlora_math
ls -lh qlora_math_adapter.tar.gz
```

然后可以在 AutoDL JupyterLab 左侧文件栏中右键下载，也可以在本地 PowerShell 使用 `scp`：

```powershell
scp -P 端口 root@服务器IP:/root/autodl-tmp/LoRA_Project/qlora_math_adapter.tar.gz D:\LoRA_Project\outputs\adapters\
```

下载到本地后解压：

```powershell
cd D:\LoRA_Project
tar -xzf outputs\adapters\qlora_math_adapter.tar.gz
```

## 10. 注意事项

- 第一次 baseline 只追求跑通，不要急着调参数。
- 如果显存不足，优先降低 `max_seq_length` 到 `1024`，或增大 `gradient_accumulation_steps` 并保持 batch size 为 `1`。
- 如果下载模型很慢，可以在 AutoDL 上提前配置 Hugging Face 镜像或模型缓存。
- 训练日志、adapter 和 checkpoint 不提交 Git。
