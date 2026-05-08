# AutoDL Baseline 运行步骤

本文档用于在 AutoDL 上跑通 `QLoRA-Math` baseline。当前默认你选择的是已经预装 Qwen 7B 或 Qwen2.5-Math-7B 的镜像，所以优先使用镜像里的本地模型路径，不重新从 Hugging Face 下载 7B 模型。

本地电脑仍然只做轻量开发，不安装 PyTorch、CUDA、Transformers、PEFT、bitsandbytes 等重依赖。

## 1. 创建实例

在 AutoDL 控制台选择：

- GPU：优先 `RTX 5090 32GB`。
- 镜像：优先选择 `PyTorch + CUDA 12.8 + Python 3.10/3.11`。
- 如果镜像说明里写了已配置 `Qwen 7B`、`Qwen2.5-Math-7B` 或类似模型，优先选择这种镜像。
- 代码、数据、缓存和训练输出都尽量放到 `/root/autodl-tmp` 数据盘。

进入实例后先检查 GPU：

```bash
nvidia-smi
```

## 2. 克隆项目

```bash
cd /root/autodl-tmp
git clone https://github.com/Liekkas-03/LoRA_Project.git
cd LoRA_Project
```

如果 GitHub 要求输入密码，不要输入 GitHub 登录密码。公开仓库一般不需要密码；私有仓库请使用 Personal Access Token 或 SSH key。

## 3. 安装云端依赖到数据盘

不要直接在系统盘默认 miniconda 环境里安装 `requirements-gpu.txt`。AutoDL 的系统盘通常较小，这样会把依赖装到 `/root/miniconda3`，很容易空间不足。

推荐方案是在数据盘 `/root/autodl-tmp` 中安装一套自己的 Miniconda，之后 conda 环境、pip 包、conda 缓存、pip 缓存都尽量留在数据盘。

如果网络慢，可以先开启 AutoDL 网络加速：

```bash
source /etc/network_turbo
```

下载 Miniconda 安装脚本到数据盘：

```bash
cd /root/autodl-tmp
mkdir -p /root/autodl-tmp/installers
mkdir -p /root/autodl-tmp/pip_cache
mkdir -p /root/autodl-tmp/hf_cache
mkdir -p /root/autodl-tmp/conda_pkgs

if [ ! -d /root/autodl-tmp/miniconda3 ]; then
  wget -O /root/autodl-tmp/installers/miniconda.sh https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh
fi
```

如果清华源下载失败，换官方源重新下载：

```bash
rm -f /root/autodl-tmp/installers/miniconda.sh
wget -O /root/autodl-tmp/installers/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

运行安装脚本。这里建议按提示交互式安装：前面一路按 Enter 或输入 yes；当它提示安装路径并默认显示 `/root/miniconda3` 时，不要直接回车，要改成 `/root/autodl-tmp/miniconda3`。

```bash
if [ ! -d /root/autodl-tmp/miniconda3 ]; then
  bash /root/autodl-tmp/installers/miniconda.sh
fi
```

安装完成后初始化数据盘里的 conda：

```bash
source /root/autodl-tmp/miniconda3/bin/activate
conda init
source ~/.bashrc
```

创建项目环境。后续的环境和依赖都应该创建在 `/root/autodl-tmp/miniconda3` 下：

```bash
export CONDA_PKGS_DIRS=/root/autodl-tmp/conda_pkgs

if ! conda env list | grep -q "/root/autodl-tmp/miniconda3/envs/lora"; then
  conda create -n lora python=3.10 -y
fi
conda activate lora

which conda
which python
which pip
python -c "import sys; print(sys.prefix)"
```

确认输出路径都在 `/root/autodl-tmp` 下，类似：

```text
/root/autodl-tmp/miniconda3/bin/conda
/root/autodl-tmp/miniconda3/envs/lora/bin/python
/root/autodl-tmp/miniconda3/envs/lora/bin/pip
/root/autodl-tmp/miniconda3/envs/lora
```

如果 `which python` 或 `which pip` 仍然指向 `/root/miniconda3`，说明还没有切换到数据盘 conda，不要继续安装依赖。先执行：

```bash
source /root/autodl-tmp/miniconda3/bin/activate
conda activate lora
hash -r
which python
which pip
```

确认路径已经切换到 `/root/autodl-tmp/miniconda3/envs/lora` 后，再继续下面的安装。

设置 pip 和 Hugging Face 缓存到数据盘：

```bash
export PIP_CACHE_DIR=/root/autodl-tmp/pip_cache
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
```

注意：`requirements-gpu.txt` 里不包含 PyTorch。因为我们现在使用的是自己安装在数据盘里的 Miniconda，所以需要先把 PyTorch 安装到这个数据盘 conda 环境中。

RTX 5090 推荐先用 CUDA 12.8 版本：

```bash
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

然后安装项目依赖：

```bash
cd /root/autodl-tmp/LoRA_Project
pip install --no-cache-dir -r requirements-gpu.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

验证依赖和 GPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -c "import transformers, peft, trl, bitsandbytes, accelerate; print('deps ok')"
```

每次重新打开终端后，都需要重新激活环境：

```bash
source /root/autodl-tmp/miniconda3/bin/activate
export CONDA_PKGS_DIRS=/root/autodl-tmp/conda_pkgs
conda activate lora
export PIP_CACHE_DIR=/root/autodl-tmp/pip_cache
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
```

如果你想让这些设置自动生效，可以追加到 `~/.bashrc`：

```bash
echo 'export CONDA_PKGS_DIRS=/root/autodl-tmp/conda_pkgs' >> ~/.bashrc
echo 'conda activate lora' >> ~/.bashrc
echo 'export PIP_CACHE_DIR=/root/autodl-tmp/pip_cache' >> ~/.bashrc
echo 'export HF_HOME=/root/autodl-tmp/hf_cache' >> ~/.bashrc
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

## 3.1 找到镜像中的 Qwen 本地路径

如果镜像已经配好 Qwen 7B，先找模型目录：

```bash
find /root -maxdepth 5 -type f -name config.json 2>/dev/null | grep -i qwen
find /root/autodl-tmp -maxdepth 6 -type f -name config.json 2>/dev/null | grep -i qwen
```

如果上面没找到，再全盘找一次：

```bash
find / -type f -name config.json 2>/dev/null | grep -i qwen
```

找到目录后检查是否包含这些文件：

```bash
ls /path/to/Qwen-model-dir
```

常见文件包括：

```text
config.json
tokenizer_config.json
model.safetensors.index.json
model-00001-of-00004.safetensors
```

复制一份配置，改成本地模型路径：

```bash
cp configs/qlora_math.yaml configs/qlora_math_local.yaml
vim configs/qlora_math_local.yaml
```

把：

```yaml
name_or_path: Qwen/Qwen2.5-Math-7B
```

改成你的真实模型目录，例如：

```yaml
name_or_path: /root/autodl-tmp/models/Qwen2.5-Math-7B
```

后续训练优先使用这个本地配置：

```bash
bash scripts/autodl_train.sh configs/qlora_math_local.yaml
```

## 4. 准备数据目录

```bash
mkdir -p data/raw data/processed
```

下载数据：

```bash
cd /root/autodl-tmp/LoRA_Project/data/raw
git clone https://github.com/openai/grade-school-math.git
git clone https://github.com/hendrycks/math.git
cd /root/autodl-tmp/LoRA_Project
```

## 5. 转换 GSM8K

```bash
python -m src.data.prepare_gsm8k --input data/raw/grade-school-math/grade_school_math/data/train.jsonl --output data/processed/gsm8k_train.jsonl --split train

python -m src.data.prepare_gsm8k --input data/raw/grade-school-math/grade_school_math/data/test.jsonl --output data/processed/gsm8k_test.jsonl --split test
```

## 6. 转换 MATH

```bash
python -m src.data.prepare_math --input-dir data/raw/math/MATH/train --output data/processed/math_train.jsonl --split train

python -m src.data.prepare_math --input-dir data/raw/math/MATH/test --output data/processed/math_test.jsonl --split test
```

## 7. 划分 dev

```bash
python -m src.data.split_data --input data/processed/gsm8k_train.jsonl --train-output data/processed/gsm8k_train_split.jsonl --dev-output data/processed/gsm8k_dev.jsonl --dev-ratio 0.1 --seed 42

python -m src.data.split_data --input data/processed/math_train.jsonl --train-output data/processed/math_train_split.jsonl --dev-output data/processed/math_dev.jsonl --dev-ratio 0.1 --seed 42
```

## 8. 合并训练集和 dev 集

训练脚本默认读取：

```text
data/processed/train.jsonl
data/processed/dev.jsonl
```

生成这两个文件：

```bash
python -m src.data.merge_data --inputs data/processed/gsm8k_train_split.jsonl data/processed/math_train_split.jsonl --output data/processed/train.jsonl

python -m src.data.merge_data --inputs data/processed/gsm8k_dev.jsonl data/processed/math_dev.jsonl --output data/processed/dev.jsonl
```

## 9. 运行 QLoRA baseline

如果你已经把配置改成本地模型路径：

```bash
bash scripts/autodl_train.sh configs/qlora_math_local.yaml
```

如果你确实要从 Hugging Face 或镜像站下载模型：

```bash
bash scripts/autodl_train.sh configs/qlora_math.yaml
```

等价命令：

```bash
python -m src.train.train_sft --config configs/qlora_math.yaml
```

如果显存不足，先把 `configs/qlora_math.yaml` 或 `configs/qlora_math_local.yaml` 中的 `max_seq_length` 从 `2048` 改成：

```yaml
max_seq_length: 1024
```

## 10. 检查训练输出

```bash
ls outputs/adapters/qlora_math
```

正常应看到类似：

```text
adapter_config.json
adapter_model.safetensors
tokenizer_config.json
```

## 11. 云端生成答案并评测准确率

训练完成后，需要继续在 AutoDL 上加载 base model + LoRA adapter，给 dev/test 样本生成 `prediction` 字段。这个步骤需要 GPU，建议不要放到本地做。

smoke run 可以先对 `dev_smoke.jsonl` 生成预测：

```bash
cd /root/autodl-tmp/LoRA_Project

bash scripts/autodl_generate.sh \
  /root/Qwen2.5/text-generation-webui/models/Qwen2.5-7B-Instruct \
  outputs/adapters/qlora_math_smoke \
  data/processed/dev_smoke.jsonl \
  outputs/reports/dev_smoke_preds.jsonl \
  256
```

然后在云端或本地都可以计算准确率。云端直接执行：

```bash
python -m src.eval.evaluate_accuracy \
  --input outputs/reports/dev_smoke_preds.jsonl \
  --output outputs/reports/dev_smoke_metrics.json
```

如果要评测正式 dev 集，把输入和输出路径换成：

```bash
bash scripts/autodl_generate.sh \
  /root/Qwen2.5/text-generation-webui/models/Qwen2.5-7B-Instruct \
  outputs/adapters/qlora_math \
  data/processed/dev.jsonl \
  outputs/reports/dev_preds.jsonl \
  256

python -m src.eval.evaluate_accuracy \
  --input outputs/reports/dev_preds.jsonl \
  --output outputs/reports/dev_metrics.json
```

生成好的 `outputs/reports/*_preds.jsonl` 和 `*_metrics.json` 可以下载回本地保存。它们比 adapter 小很多，更适合进入实验记录。

## 12. 导出训练结果到本地

训练完成后，优先导出 LoRA adapter，不要把大模型本体或 checkpoint 全部提交到 GitHub。

先在 AutoDL 上压缩 adapter 目录。如果是 smoke run，例如：

```bash
cd /root/autodl-tmp/LoRA_Project
tar -czf qlora_math_smoke_adapter.tar.gz outputs/adapters/qlora_math_smoke
ls -lh qlora_math_smoke_adapter.tar.gz
```

如果是正式 baseline，改成：

```bash
cd /root/autodl-tmp/LoRA_Project
tar -czf qlora_math_adapter.tar.gz outputs/adapters/qlora_math
ls -lh qlora_math_adapter.tar.gz
```

下载方式一：用 AutoDL JupyterLab 的文件栏下载。在左侧找到 `/root/autodl-tmp/LoRA_Project/qlora_math_smoke_adapter.tar.gz`，右键 Download 即可。

下载方式二：用 `scp` 下载。在本地 PowerShell 执行，把 `端口` 和 `服务器IP` 换成 AutoDL 页面给你的 SSH 信息：

```powershell
scp -P 端口 root@服务器IP:/root/autodl-tmp/LoRA_Project/qlora_math_smoke_adapter.tar.gz D:\LoRA_Project\outputs\adapters\
```

下载到本地后，在 PowerShell 里解压：

```powershell
cd D:\LoRA_Project
tar -xzf outputs\adapters\qlora_math_smoke_adapter.tar.gz
```

注意：adapter、checkpoint、TensorBoard 日志属于训练产物，默认不提交到 GitHub。GitHub 上只保存代码、配置、文档和小型评测结果。

## 13. 快速 smoke run

如果只是确认流程能不能跑通，可以先把配置临时改小：

```yaml
max_seq_length: 1024
num_train_epochs: 0.05
save_steps: 50
eval_steps: 50
```

如果后续训练脚本支持 `max_steps`，可以用 `max_steps: 50` 做更快的 smoke baseline。

## 14. 报错时先收集这些信息

```bash
nvidia-smi
which python
which pip
python -c "import sys; print(sys.prefix)"
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
pip show transformers peft trl bitsandbytes accelerate
```

把完整报错和这些输出贴回来，我再继续帮你改训练脚本或配置。
