#!/usr/bin/env bash
set -euo pipefail

# 新机器 1-step smoke test 脚本。
#
# 用途：
#   在新机器上快速验证 CLARE 主训练链路是否可用：
#   数据集缓存 -> checkpoint 加载 -> PEFT 包装 -> adapter 训练 1 step
#   -> discriminator 训练 1 step -> checkpoint 保存。
#
# 最常用运行方式：
#   cd /新机器/上的/实际路径/clare
#   ENV=/新机器/上的/conda环境路径/_008n_clare \
#     bash _001n_my/_004n_bashs/_001n_new_machine_smoke_test.sh
#
# 指定使用哪一张 GPU：
#   先用 nvidia-smi 查看物理 GPU 编号，例如 0、1、2、3。
#   如果只想使用物理 GPU 1，把 CUDA_VISIBLE_DEVICES 放在 bash 命令前面：
#   CUDA_VISIBLE_DEVICES=1 ENV=/新机器/上的/conda环境路径/_008n_clare \
#     bash _001n_my/_004n_bashs/_001n_new_machine_smoke_test.sh
#
#   注意：设置 CUDA_VISIBLE_DEVICES=1 后，训练进程内部通常仍显示 cuda:0。
#   这是正常现象，意思是“当前进程可见的第 0 张 GPU”，实际对应物理 GPU 1。
#   MUJOCO_EGL_DEVICE_ID 默认保持 0 即可，因为它也使用进程内的可见 GPU 编号。
#
# 可选覆盖变量：
#   REPO=/path/to/clare                 # 仓库路径；默认从脚本位置自动推断
#   RUNTIME=/path/to/cache_root         # HF/datasets/LeRobot/tmp/pip 缓存根目录
#   SHORT_TMPDIR=/tmp/clare_0           # Python/PyTorch 多进程临时目录，必须是短路径
#   PRETRAIN=/path/to/pretrain          # 基础 checkpoint 路径
#   OUT=/path/to/output                 # smoke test 输出目录
#   CUDA_VISIBLE_DEVICES=1              # 只让脚本看到物理 GPU 1
#   MUJOCO_EGL_DEVICE_ID=0              # 可见 GPU 内部编号；通常保持 0
#   OVERWRITE=1                         # 如果 OUT 已存在，允许删除后重跑

# 1. 推断路径。
# 脚本放在 clare/_001n_my/_004n_bashs 下，所以 ../.. 正好是 clare 仓库根目录。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RUNTIME="${RUNTIME:-$REPO/_001n_my/_003n_clare_download}"
SHORT_TMPDIR="${SHORT_TMPDIR:-/tmp/clare_$(id -u)}"
ENV="${ENV:-${CLARE_ENV:-}}"

# 2. 检查 conda 环境路径。
# 新机器路径不同，必须显式传 ENV，或提前 export CLARE_ENV。
if [[ -z "$ENV" ]]; then
  echo "ERROR: Please set ENV to the conda environment path on the new machine." >&2
  echo "Example: ENV=/path/to/_008n_clare bash $0" >&2
  exit 2
fi

# 3. 检查 Python 是否存在。
# 这里不用 conda activate，直接调用 "$ENV/bin/python"，更适合脚本和远程机器使用。
PYTHON="$ENV/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python not found or not executable: $PYTHON" >&2
  exit 2
fi

cd "$REPO"

# 4. 创建缓存目录和输出目录。
# RUNTIME 只放缓存；训练输出和日志放在 $REPO/outputs。
mkdir -p "$RUNTIME"/{hf_home,hf_datasets,lerobot,tmp,pip_cache} "$SHORT_TMPDIR" "$REPO/outputs/_smoke" "$REPO/outputs/logs"

# 5. 设置运行时环境变量。
# 这些变量避免 Hugging Face / datasets / LeRobot / 临时文件写到默认位置。
# Python/PyTorch multiprocessing 会在 TMPDIR 下创建 AF_UNIX socket。
# Linux 对 AF_UNIX 路径长度有限制，所以 TMPDIR 不能使用很长的项目路径。
# HF_HOME、HF_DATASETS_CACHE、HF_LEROBOT_HOME 仍然放在 RUNTIME，避免大缓存写到 /tmp。
# CUDA_VISIBLE_DEVICES 控制脚本能看到哪些物理 GPU；默认只使用物理 GPU 0。
# 例如命令前加 CUDA_VISIBLE_DEVICES=1，则脚本只会看到物理 GPU 1。
# 在这种情况下，PyTorch 日志里的 cuda:0 指的是“可见 GPU 0”，不是物理 GPU 0。
# MUJOCO_EGL_DEVICE_ID 使用可见 GPU 的内部编号；只暴露一张卡时通常保持 0。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export HF_HOME="$RUNTIME/hf_home"
export HF_DATASETS_CACHE="$RUNTIME/hf_datasets"
export HF_LEROBOT_HOME="$RUNTIME/lerobot"
export TMPDIR="$SHORT_TMPDIR"
export PIP_CACHE_DIR="$RUNTIME/pip_cache"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

# 6. 检查基础预训练 checkpoint。
# 如果没有复制这个目录，可以按 requirements_and_run.md 第 14.6 节重新 hf download。
PRETRAIN="${PRETRAIN:-$REPO/outputs/dit_flow_mt_libero_90_pretrain}"
if [[ ! -f "$PRETRAIN/config.json" || ! -f "$PRETRAIN/model.safetensors" || ! -f "$PRETRAIN/train_config.json" ]]; then
  echo "ERROR: Pretrained checkpoint is incomplete or missing: $PRETRAIN" >&2
  echo "Expected config.json, model.safetensors, and train_config.json." >&2
  exit 2
fi

# 7. 设置本次 smoke test 输出目录和日志文件。
# 默认输出带时间戳，避免重复运行时误覆盖旧结果。
RUN_NAME="${RUN_NAME:-clare_cu117_new_machine_task0_$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-$REPO/outputs/_smoke/$RUN_NAME}"
LOG_FILE="${LOG_FILE:-$REPO/outputs/logs/${RUN_NAME}.log}"

# 8. 输出目录已存在时默认拒绝覆盖。
# 如果确认旧结果不要了，运行时加 OVERWRITE=1。
if [[ -e "$OUT" ]]; then
  if [[ "${OVERWRITE:-0}" == "1" ]]; then
    rm -rf "$OUT"
  else
    echo "ERROR: Output directory already exists: $OUT" >&2
    echo "Set OVERWRITE=1 to remove it, or set OUT to another path." >&2
    exit 2
  fi
fi

echo "REPO=$REPO"
echo "ENV=$ENV"
echo "RUNTIME=$RUNTIME"
echo "TMPDIR=$TMPDIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "MUJOCO_EGL_DEVICE_ID=$MUJOCO_EGL_DEVICE_ID"
echo "PRETRAIN=$PRETRAIN"
echo "OUT=$OUT"
echo "LOG_FILE=$LOG_FILE"

# 9. 先做轻量 import / CUDA 检查。
# 这里能提前发现 torch CUDA、diffusers 版本或 editable 路径错误。
"$PYTHON" - <<'PY'
import torch
import diffusers
import huggingface_hub
import lerobot.scripts.clare as clare

print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
print("diffusers", diffusers.__version__)
print("huggingface_hub", huggingface_hub.__version__)
print("clare", clare.__file__)
PY

# 10. 跑最小训练链路。
# 只训练 adapter 1 step 和 discriminator 1 step，目标是验证代码能跑通，不评估性能。
"$PYTHON" ./lerobot_lsy/src/lerobot/scripts/clare.py \
  --seed=42 \
  --job_name=clare_smoke_cu117_new_machine_task0 \
  --output_dir="$OUT" \
  --dataset.repo_id=continuallearning/libero_10_image_task_0 \
  --dataset.video_backend=pyav \
  --policy.path="$PRETRAIN" \
  --policy.push_to_hub=false \
  --batch_size=1 \
  --num_workers=0 \
  --steps=1 \
  --env.type=libero \
  --env.benchmark=libero_10 \
  --env.task=Libero_10_Task_0 \
  --eval_freq=0 \
  --save_freq=1 \
  --log_freq=1 \
  --peft_cfg_path=./peft_lsy/peft_config/clare_dit_flow_encoder_adapter \
  --expand_threshold=1.0 \
  --detect_distribution_shift_steps=1 \
  --detect_distribution_shift_batch_size=1 \
  --detect_distribution_shift_num_workers=0 \
  --detect_distribution_shift_log_freq=1 \
  --train_discriminators_steps=1 \
  --train_discriminators_batch_size=1 \
  --train_discriminators_num_workers=0 \
  --train_discriminators_log_freq=1 \
  --train_discriminators_eval_freq=0 \
  --train_discriminators_save_freq=1 \
  --wandb.enable=false 2>&1 | tee "$LOG_FILE"

# 11. 打印 checkpoint 文件，方便确认训练和保存都完成。
echo "Smoke test finished."
echo "Checkpoint files:"
find "$OUT/checkpoints" -maxdepth 4 -type f | sort | sed -n '1,80p'
