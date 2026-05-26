# AutoDL MATH-only 实验步骤

本文档用于在 AutoDL 上运行新的 MATH-only 主线实验。当前主线不再准备 GSM8K，所有训练、验证和测试都围绕 MATH 官方 `Level 1-5` 难度标签展开。

## 1. 进入数据盘并克隆项目

```bash
cd /root/autodl-tmp
git clone https://github.com/Liekkas-03/LoRA_Project.git
cd LoRA_Project
```

如果仓库已经存在，进入目录后更新即可：

```bash
cd /root/autodl-tmp/LoRA_Project
git pull
```

## 2. 激活数据盘 conda 环境

如果你已经在 `/root/autodl-tmp/miniconda3` 配好环境，直接执行：

```bash
source /root/autodl-tmp/miniconda3/bin/activate
export CONDA_PKGS_DIRS=/root/autodl-tmp/conda_pkgs
conda activate lora
export PIP_CACHE_DIR=/root/autodl-tmp/pip_cache
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
```

确认 Python 和 pip 都在数据盘环境：

```bash
which python
which pip
python -c "import sys; print(sys.prefix)"
```

期望路径类似：

```text
/root/autodl-tmp/miniconda3/envs/lora/bin/python
/root/autodl-tmp/miniconda3/envs/lora/bin/pip
/root/autodl-tmp/miniconda3/envs/lora
```

如果还没有环境，先按下面方式安装到数据盘：

```bash
cd /root/autodl-tmp
mkdir -p installers pip_cache hf_cache conda_pkgs
wget -O installers/miniconda.sh https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash installers/miniconda.sh
```

安装路径提示默认 `/root/miniconda3` 时，不要直接回车，手动输入：

```text
/root/autodl-tmp/miniconda3
```

然后创建环境：

```bash
source /root/autodl-tmp/miniconda3/bin/activate
conda init
source ~/.bashrc
export CONDA_PKGS_DIRS=/root/autodl-tmp/conda_pkgs
conda create -n lora python=3.10 -y
conda activate lora
```

安装 GPU 依赖：

```bash
cd /root/autodl-tmp/LoRA_Project
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install --no-cache-dir -r requirements-gpu.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 3. 找到本地 Qwen 模型路径

如果镜像已经内置 Qwen 模型，先查找：

```bash
find /root -maxdepth 6 -type f -name config.json 2>/dev/null | grep -i qwen
find /root/autodl-tmp -maxdepth 6 -type f -name config.json 2>/dev/null | grep -i qwen
```

如果你看到：

```text
/root/models/Qwen/Qwen2.5-Math-7B-Instruct/config.json
```

那么 yaml 里的模型路径应改为：

```yaml
name_or_path: /root/models/Qwen/Qwen2.5-Math-7B-Instruct
```

## 4. 下载并准备 MATH 数据

```bash
mkdir -p data/raw data/processed
cd data/raw
git clone https://github.com/hendrycks/math.git
cd /root/autodl-tmp/LoRA_Project
```

转换 MATH train/test：

```bash
python -m src.data.prepare_math \
  --input-dir data/raw/math/MATH/train \
  --output data/processed/math_train.jsonl \
  --split train

python -m src.data.prepare_math \
  --input-dir data/raw/math/MATH/test \
  --output data/processed/math_test.jsonl \
  --split test
```

从 MATH train 中按 `difficulty_group` 分层划分 dev：

```bash
python -m src.data.split_data \
  --input data/processed/math_train.jsonl \
  --train-output data/processed/math_train_split.jsonl \
  --dev-output data/processed/math_dev.jsonl \
  --dev-ratio 0.1 \
  --seed 42 \
  --stratify-field difficulty_group
```

这三个文件是后续主线使用的数据：

```text
data/processed/math_train_split.jsonl
data/processed/math_dev.jsonl
data/processed/math_test.jsonl
```

## 5. 先跑 smoke run

复制一份云端本地配置：

```bash
cp configs/qlora_math.yaml configs/qlora_math_local.yaml
sed -i 's#Qwen/Qwen2.5-Math-7B#/root/models/Qwen/Qwen2.5-Math-7B-Instruct#g' configs/qlora_math_local.yaml
```

创建小样本：

```bash
head -n 512 data/processed/math_train_split.jsonl > data/processed/math_train_smoke.jsonl
head -n 64 data/processed/math_dev.jsonl > data/processed/math_dev_smoke.jsonl
```

把配置改成 smoke run：

```bash
sed -i 's#outputs/adapters/qlora_math#outputs/adapters/qlora_math_smoke#g' configs/qlora_math_local.yaml
sed -i 's#data/processed/math_train_split.jsonl#data/processed/math_train_smoke.jsonl#g' configs/qlora_math_local.yaml
sed -i 's#data/processed/math_dev.jsonl#data/processed/math_dev_smoke.jsonl#g' configs/qlora_math_local.yaml
sed -i 's#max_seq_length: 2048#max_seq_length: 1024#g' configs/qlora_math_local.yaml
sed -i 's#gradient_accumulation_steps: 16#gradient_accumulation_steps: 8#g' configs/qlora_math_local.yaml
sed -i 's#num_train_epochs: 2#num_train_epochs: 1#g' configs/qlora_math_local.yaml
sed -i 's#logging_steps: 10#logging_steps: 1#g' configs/qlora_math_local.yaml
sed -i 's#save_steps: 500#save_steps: 50#g' configs/qlora_math_local.yaml
sed -i 's#eval_steps: 500#eval_steps: 50#g' configs/qlora_math_local.yaml
```

启动：

```bash
bash scripts/autodl_train.sh configs/qlora_math_local.yaml
```

看到 `adapter_model.safetensors`、`adapter_config.json`、`tokenizer.json` 就说明训练闭环跑通。

## 6. 跑正式 QLoRA-Math baseline

```bash
cp configs/qlora_math.yaml configs/qlora_math_formal.yaml
sed -i 's#Qwen/Qwen2.5-Math-7B#/root/models/Qwen/Qwen2.5-Math-7B-Instruct#g' configs/qlora_math_formal.yaml
sed -i 's#outputs/adapters/qlora_math#outputs/adapters/qlora_math_formal#g' configs/qlora_math_formal.yaml
sed -i 's#max_seq_length: 2048#max_seq_length: 1024#g' configs/qlora_math_formal.yaml

mkdir -p outputs/logs
bash scripts/autodl_train.sh configs/qlora_math_formal.yaml 2>&1 | tee outputs/logs/qlora_math_formal.log
```

## 7. 跑正式 AdaQLoRA-Math

```bash
cp configs/adaqlora_math.yaml configs/adaqlora_math_formal.yaml
sed -i 's#Qwen/Qwen2.5-Math-7B#/root/models/Qwen/Qwen2.5-Math-7B-Instruct#g' configs/adaqlora_math_formal.yaml
sed -i 's#outputs/adapters/adaqlora_math#outputs/adapters/adaqlora_math_formal#g' configs/adaqlora_math_formal.yaml
sed -i 's#max_seq_length: 2048#max_seq_length: 1024#g' configs/adaqlora_math_formal.yaml
sed -i 's#outputs/reports/rank_pattern.json#outputs/reports/rank_pattern_formal.json#g' configs/adaqlora_math_formal.yaml

mkdir -p outputs/logs
bash scripts/autodl_train.sh configs/adaqlora_math_formal.yaml 2>&1 | tee outputs/logs/adaqlora_math_formal.log
```

训练启动后日志里应出现：

```text
Loaded AdaQLoRA rank patterns: 196 modules
```

这说明分层 rank pattern 已经注入，不是普通 QLoRA。

## 8. 生成预测并评测

QLoRA：

```bash
bash scripts/autodl_generate.sh \
  /root/models/Qwen/Qwen2.5-Math-7B-Instruct \
  outputs/adapters/qlora_math_formal \
  data/processed/math_test.jsonl \
  outputs/reports/math_test_qlora_preds.jsonl \
  512

python -m src.eval.evaluate_accuracy \
  --input outputs/reports/math_test_qlora_preds.jsonl \
  --output outputs/reports/math_test_qlora_metrics.json
```

AdaQLoRA：

```bash
bash scripts/autodl_generate.sh \
  /root/models/Qwen/Qwen2.5-Math-7B-Instruct \
  outputs/adapters/adaqlora_math_formal \
  data/processed/math_test.jsonl \
  outputs/reports/math_test_ada_preds.jsonl \
  512

python -m src.eval.evaluate_accuracy \
  --input outputs/reports/math_test_ada_preds.jsonl \
  --output outputs/reports/math_test_ada_metrics.json
```

## 9. 导出结果

只导出 adapter、日志和评测结果，不导出 base model。

```bash
cd /root/autodl-tmp/LoRA_Project
tar -czf qlora_math_formal_adapter.tar.gz outputs/adapters/qlora_math_formal
tar -czf qlora_math_formal_eval.tar.gz outputs/logs/qlora_math_formal.log outputs/reports/math_test_qlora_preds.jsonl outputs/reports/math_test_qlora_metrics.json

tar -czf adaqlora_math_formal_adapter.tar.gz outputs/adapters/adaqlora_math_formal
tar -czf adaqlora_math_formal_eval.tar.gz outputs/logs/adaqlora_math_formal.log outputs/reports/math_test_ada_preds.jsonl outputs/reports/math_test_ada_metrics.json
```

然后可以用 AutoDL JupyterLab 右键下载，或在本地 PowerShell 用 `scp` 下载。
