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

## 3. 安装云端依赖到数据盘

不要直接在系统盘默认 miniconda 环境里执行 `pip install -r requirements-gpu.txt`，否则依赖会装到 `/root/miniconda3`，容易把系统盘占满。

推荐在数据盘 `/root/autodl-tmp` 中安装自己的 Miniconda，之后 conda 环境和 pip 包都安装到数据盘：

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

运行安装脚本时，前面一路按 Enter 或输入 yes；当它提示安装路径并默认显示 `/root/miniconda3` 时，必须改成 `/root/autodl-tmp/miniconda3`。安装依赖前确认 `which python`、`which pip` 和 `python -c "import sys; print(sys.prefix)"` 都指向 `/root/autodl-tmp/miniconda3/envs/lora`。

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
