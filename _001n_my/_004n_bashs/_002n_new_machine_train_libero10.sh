#!/usr/bin/env bash
set -euo pipefail

# 新机器 LIBERO-10 正式训练脚本。
#
# 用途：
#   按 LIBERO-10 的 task 顺序运行 CLARE 训练。默认跑 task 0 到 task 9；
#   task 1 之后会自动读取上一 task 的 checkpoints/last/adapter，
#   并通过 --peft_weight_path 传给当前 task，实现持续学习串联。
#
# 最常用运行方式：
#   cd /新机器/上的/实际路径/clare
#   ENV=/新机器/上的/conda环境路径/_008n_clare \
#     bash _001n_my/_004n_bashs/_002n_new_machine_train_libero10.sh
#
# 指定使用哪一张 GPU：
#   先用 nvidia-smi 查看物理 GPU 编号，例如 0、1、2、3。
#   如果只想使用物理 GPU 1，把 CUDA_VISIBLE_DEVICES 放在 bash 命令前面：
#   CUDA_VISIBLE_DEVICES=1 ENV=/新机器/上的/conda环境路径/_008n_clare \
#     bash _001n_my/_004n_bashs/_002n_new_machine_train_libero10.sh
#
#   注意：设置 CUDA_VISIBLE_DEVICES=1 后，训练进程内部通常仍显示 cuda:0。
#   这是正常现象，意思是“当前进程可见的第 0 张 GPU”，实际对应物理 GPU 1。
#   MUJOCO_EGL_DEVICE_ID 默认保持 0 即可，因为它也使用进程内的可见 GPU 编号。
#
# 只跑单个 task，例如 task 0：
#   ENV=/新机器/上的/conda环境路径/_008n_clare TASK_START=0 TASK_END=0 \
#     bash _001n_my/_004n_bashs/_002n_new_machine_train_libero10.sh
#
# 常用覆盖变量：
#   REPO=/path/to/clare                 # 仓库路径；默认从脚本位置自动推断
#   RUNTIME=/path/to/cache_root         # HF/datasets/LeRobot/tmp/pip 缓存根目录
#   PRETRAIN=/path/to/pretrain          # 基础 checkpoint 路径
#   TASK_START=0 TASK_END=9             # 要训练的 task 范围
#   BATCH_SIZE=16 NUM_WORKERS=8         # 显存或 IO 紧张时可调小
#   STEPS=20000 DISC_STEPS=2000         # adapter 训练步数和 discriminator 训练步数
#   CUDA_VISIBLE_DEVICES=1              # 只让脚本看到物理 GPU 1
#   MUJOCO_EGL_DEVICE_ID=0              # 可见 GPU 内部编号；通常保持 0
#   WANDB_ENABLE=true WANDB_ENTITY=...  # 需要 W&B 时再打开
#   OVERWRITE=1                         # 如果输出目录已存在，允许删除后重跑

# 1. 推断路径。
# 脚本放在 clare/_001n_my/_004n_bashs 下，所以 ../.. 正好是 clare 仓库根目录。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RUNTIME="${RUNTIME:-$REPO/_001n_my/_003n_clare_download}"
ENV="${ENV:-${CLARE_ENV:-}}"

# 2. 检查 conda 环境路径。
# 新机器路径不同，必须显式传 ENV，或提前 export CLARE_ENV。
if [[ -z "$ENV" ]]; then
  echo "ERROR: Please set ENV to the conda environment path on the new machine." >&2
  echo "Example: ENV=/path/to/_008n_clare bash $0" >&2
  exit 2
fi

# 3. 检查 Python 是否存在。
# 不依赖 conda activate，直接调用 "$ENV/bin/python"。
PYTHON="$ENV/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python not found or not executable: $PYTHON" >&2
  exit 2
fi

cd "$REPO"

# 4. 创建缓存目录和日志目录。
# RUNTIME 只放缓存；训练 checkpoint 和日志放到 $REPO/outputs。
mkdir -p "$RUNTIME"/{hf_home,hf_datasets,lerobot,tmp,pip_cache} "$REPO/outputs/logs"

# 5. 设置运行时环境变量。
# 这些变量避免 Hugging Face / datasets / LeRobot / 临时文件写到默认位置。
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
export TMPDIR="$RUNTIME/tmp"
export PIP_CACHE_DIR="$RUNTIME/pip_cache"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

# 6. 训练配置。
# 默认值来自 README/已跑通配置；都可以在命令行前用环境变量覆盖。
SEED="${SEED:-42}"
TASK_START="${TASK_START:-0}"
TASK_END="${TASK_END:-9}"
STEPS="${STEPS:-20000}"
DISC_STEPS="${DISC_STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-16}"
DETECT_STEPS="${DETECT_STEPS:-200}"
LOG_FREQ="${LOG_FREQ:-100}"
DISC_LOG_FREQ="${DISC_LOG_FREQ:-50}"
EVAL_FREQ="${EVAL_FREQ:-0}"
DISC_EVAL_FREQ="${DISC_EVAL_FREQ:-0}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
WANDB_DISABLE_ARTIFACT="${WANDB_DISABLE_ARTIFACT:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-clare_experiments}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
OVERWRITE="${OVERWRITE:-0}"

# 7. 输出根目录和基础 checkpoint。
# ROOT 下会为每个 task 建一个独立输出目录；PRETRAIN 是 LIBERO-90 预训练模型。
ROOT="${ROOT:-$REPO/outputs/libero_10/clare}"
PRETRAIN="${PRETRAIN:-$REPO/outputs/dit_flow_mt_libero_90_pretrain}"

# 8. 检查基础预训练 checkpoint。
# 如果缺文件，先按 requirements_and_run.md 第 14.6 节复制或下载 checkpoint。
if [[ ! -f "$PRETRAIN/config.json" || ! -f "$PRETRAIN/model.safetensors" || ! -f "$PRETRAIN/train_config.json" ]]; then
  echo "ERROR: Pretrained checkpoint is incomplete or missing: $PRETRAIN" >&2
  echo "Expected config.json, model.safetensors, and train_config.json." >&2
  exit 2
fi

# 9. 检查 task 范围。
# 当前脚本只针对 LIBERO-10，因此 task id 必须在 0..9。
if (( TASK_START < 0 || TASK_END > 9 || TASK_START > TASK_END )); then
  echo "ERROR: Expected 0 <= TASK_START <= TASK_END <= 9, got TASK_START=$TASK_START TASK_END=$TASK_END" >&2
  exit 2
fi

mkdir -p "$ROOT"

echo "REPO=$REPO"
echo "ENV=$ENV"
echo "RUNTIME=$RUNTIME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "MUJOCO_EGL_DEVICE_ID=$MUJOCO_EGL_DEVICE_ID"
echo "ROOT=$ROOT"
echo "PRETRAIN=$PRETRAIN"
echo "TASK_START=$TASK_START TASK_END=$TASK_END"
echo "SEED=$SEED STEPS=$STEPS DISC_STEPS=$DISC_STEPS BATCH_SIZE=$BATCH_SIZE NUM_WORKERS=$NUM_WORKERS"

# 10. 先做轻量 import / CUDA 检查。
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

# 11. 逐 task 训练。
# task 0 从基础 checkpoint 开始；task 1..9 会接上前一个 task 的 adapter。
for TASK in $(seq "$TASK_START" "$TASK_END"); do
  OUT="$ROOT/dit_flow_mt_cl_seed_${SEED}_libero_10_task_${TASK}_encoder_mlp_adapter_threshold_1_0"
  LOG_FILE="$REPO/outputs/logs/clare_libero_10_task_${TASK}_seed_${SEED}_$(date +%Y%m%d_%H%M%S).log"

# 12. 输出目录已存在时默认拒绝覆盖。
# 如果确认旧结果不要了，运行时加 OVERWRITE=1。
  if [[ -e "$OUT" ]]; then
    if [[ "$OVERWRITE" == "1" ]]; then
      rm -rf "$OUT"
    else
      echo "ERROR: Output directory already exists: $OUT" >&2
      echo "Set OVERWRITE=1 to remove it, or set ROOT/SEED to another output path." >&2
      exit 2
    fi
  fi

# 13. 组装当前 task 的训练参数。
# 默认关闭仿真评测和 W&B，优先保证离线训练主链路稳定。
  ARGS=(
    --seed="$SEED"
    --job_name="clare_libero_10_task_${TASK}"
    --output_dir="$OUT"
    --dataset.repo_id="continuallearning/libero_10_image_task_${TASK}"
    --dataset.video_backend=pyav
    --policy.path="$PRETRAIN"
    --policy.push_to_hub=false
    --batch_size="$BATCH_SIZE"
    --num_workers="$NUM_WORKERS"
    --steps="$STEPS"
    --env.type=libero
    --env.benchmark=libero_10
    --env.task="Libero_10_Task_${TASK}"
    --eval_freq="$EVAL_FREQ"
    --save_freq="$STEPS"
    --log_freq="$LOG_FREQ"
    --peft_cfg_path=./peft_lsy/peft_config/clare_dit_flow_encoder_adapter
    --expand_threshold=1.0
    --detect_distribution_shift_steps="$DETECT_STEPS"
    --detect_distribution_shift_batch_size="$BATCH_SIZE"
    --detect_distribution_shift_num_workers="$NUM_WORKERS"
    --detect_distribution_shift_log_freq=10
    --train_discriminators_steps="$DISC_STEPS"
    --train_discriminators_batch_size="$BATCH_SIZE"
    --train_discriminators_num_workers="$NUM_WORKERS"
    --train_discriminators_log_freq="$DISC_LOG_FREQ"
    --train_discriminators_eval_freq="$DISC_EVAL_FREQ"
    --train_discriminators_save_freq="$DISC_STEPS"
    --wandb.enable="$WANDB_ENABLE"
    --wandb.disable_artifact="$WANDB_DISABLE_ARTIFACT"
    --wandb.project="$WANDB_PROJECT"
  )

# 14. 可选启用 W&B entity。
# 只有 WANDB_ENABLE=true 且 WANDB_ENTITY 非空时才传 entity，避免 placeholder 造成报错。
  if [[ "$WANDB_ENABLE" == "true" && -n "$WANDB_ENTITY" ]]; then
    ARGS+=(--wandb.entity="$WANDB_ENTITY")
  fi

# 15. 持续学习串联：从 task 1 开始加载上一 task 的 adapter。
# 如果从中间 task 开始跑，必须保证上一 task 的 checkpoints/last/adapter 已存在。
  if (( TASK > 0 )); then
    PREV=$((TASK - 1))
    PREV_OUT="$ROOT/dit_flow_mt_cl_seed_${SEED}_libero_10_task_${PREV}_encoder_mlp_adapter_threshold_1_0"
    PREV_ADAPTER="$PREV_OUT/checkpoints/last/adapter"
    if [[ ! -d "$PREV_ADAPTER" ]]; then
      echo "ERROR: Previous adapter not found for task $TASK: $PREV_ADAPTER" >&2
      echo "Run previous tasks first, or set TASK_START=0." >&2
      exit 2
    fi
    ARGS+=(--peft_weight_path="$PREV_ADAPTER")
  fi

  echo "==== Running LIBERO-10 task $TASK ===="
  echo "OUT=$OUT"
  echo "LOG_FILE=$LOG_FILE"
# 16. 启动训练，并同时把日志写到文件和终端。
  "$PYTHON" ./lerobot_lsy/src/lerobot/scripts/clare.py "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"
done

echo "LIBERO-10 training command finished."
