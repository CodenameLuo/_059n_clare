# CLARE 已跑通环境与正式训练指令

本文档记录当前这台 AutoDL 机器上已经验证能跑通 CLARE 训练主链路的环境、缓存路径、checkpoint 路径和运行命令。

当前约束是：必须使用 CUDA 11.7 对应的 `torch==2.0.1`。因此这里的依赖组合不是 README 原始声明的最新 LeRobot 组合，而是为了兼容 CUDA 11.7 做过固定和补丁后的组合。

## 1. 关键路径

仓库路径：

```bash
REPO=/root/autodl-tmp/_000n_00000000/_000n_uni/_001n_my_codes/_001n_projects/_005n_claude_code_projs/_001n_cc_translation/_010n_codes/_026n_clare/clare
```

conda 环境：

```bash
ENV=/root/autodl-tmp/_000n_00000000/_003n_backend/_001n_z001/_001n_miniconda/_002n_conda_list/_008n_clare
```

运行缓存根目录：

```bash
RUNTIME=/root/autodl-tmp/_000n_00000000/_000n_uni/_001n_my_codes/_001n_projects/_005n_claude_code_projs/_001n_cc_translation/_010n_codes/_026n_clare/clare/_001n_my/_003n_clare_download
```

训练输出和 checkpoint 统一放在仓库的 `outputs` 目录：

```bash
$REPO/outputs
```

当前不再保留 `$RUNTIME/outputs` 兼容软链接；新命令直接写 `$REPO/outputs/...`。

基础 checkpoint 本地路径：

```bash
$REPO/outputs/dit_flow_mt_libero_90_pretrain
```

注意：README 里写的 `dit_flow_mt_libero_90_pretrain_new` 当前未验证可匿名访问；本机实际下载并跑通的是：

```text
continuallearning/dit_flow_mt_libero_90_pretrain
https://huggingface.co/continuallearning/dit_flow_mt_libero_90_pretrain/tree/main
```

## 2. 当前已验证的核心版本

当前环境中已验证：

```text
torch==2.0.1
torchvision==0.15.2
torchaudio==2.0.2
diffusers==0.32.2
huggingface_hub==0.36.2
transformers==4.48.3
tokenizers==0.21.4
numpy==1.26.4
datasets==3.6.0
av==17.1.0
peft==0.17.1.dev0
accelerate==1.14.0
safetensors==0.8.0
draccus==0.10.0
wandb==0.28.0
lerobot==0.1.0
```

CUDA 检查结果：

```text
torch.version.cuda == 11.7
torch.cuda.is_available() == True
GPU: NVIDIA GeForce RTX 3090
```

可用下面命令复查：

```bash
$ENV/bin/python - <<'PY'
import importlib.metadata as md
import torch

for p in [
    "torch", "torchvision", "torchaudio", "diffusers", "huggingface_hub",
    "transformers", "tokenizers", "numpy", "datasets", "av", "peft",
    "accelerate", "safetensors", "draccus", "wandb", "lerobot",
]:
    try:
        print(f"{p}=={md.version(p)}")
    except md.PackageNotFoundError:
        print(f"{p} not installed")

print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

## 3. 已知依赖取舍

`pip check` 仍可能报告这些声明冲突：

```text
lerobot 0.1.0 requires torchcodec, which is not installed.
lerobot 0.1.0 has requirement torch>=2.2.1, but you have torch 2.0.1.
lerobot 0.1.0 has requirement torchvision>=0.21.0, but you have torchvision 0.15.2.
opencv-python-headless 4.13.0.92 requires numpy>=2, but numpy is 1.26.4.
rerun-sdk 0.33.1 requires numpy>=2, but numpy is 1.26.4.
```

这些冲突目前是有意保留的兼容边界：

- `torch==2.0.1` / `torchvision==0.15.2` 是为了匹配 CUDA 11.7。
- 当前新版 `torchcodec` 需要更新的 torch API，在 `torch==2.0.1` 下不兼容；训练时显式使用 `--dataset.video_backend=pyav`。
- `numpy==1.26.4` 是为了避免部分旧栈 ABI 问题；当前 CLARE 离线训练 smoke test 已通过。

不要随手执行会升级 torch 的安装命令。尤其不要让 `pip install -e ./lerobot_lsy` 自动拉依赖把 torch 升到 2.2+。

## 4. 一次性环境准备

如果只是继续使用当前环境，不需要重装，直接从第 5 节开始。

如果需要复现当前固定版本，可按下面思路操作。注意 `--no-deps` 是为了避免 pip 自动升级 torch。

```bash
cd "$REPO"

mkdir -p "$RUNTIME"/{hf_home,hf_datasets,lerobot,tmp,pip_cache} "$REPO/outputs"
export PIP_CACHE_DIR="$RUNTIME/pip_cache"
export TMPDIR="$RUNTIME/tmp"

$ENV/bin/python -m pip install \
  torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
  --index-url https://download.pytorch.org/whl/cu117

$ENV/bin/python -m pip install --no-deps \
  diffusers==0.32.2 \
  huggingface_hub==0.36.2

$ENV/bin/python -m pip install -e ./peft_lsy --no-deps
$ENV/bin/python -m pip install -e ./lerobot_lsy --no-deps

$ENV/bin/python -m pip uninstall -y torchcodec
```

如果已经装好环境，只需要确认 import：

```bash
cd "$REPO"

$ENV/bin/python - <<'PY'
import torch
import diffusers, huggingface_hub
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import lerobot.scripts.clare as clare

print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
print("diffusers", diffusers.__version__)
print("huggingface_hub", huggingface_hub.__version__)
print("clare import ok:", clare.__file__)
print("scheduler imports ok:", get_scheduler, DDIMScheduler, DDPMScheduler)
PY
```

## 5. 每次运行前都设置的环境变量

建议每次跑训练前先执行：

```bash
ENV=/root/autodl-tmp/_000n_00000000/_003n_backend/_001n_z001/_001n_miniconda/_002n_conda_list/_008n_clare
REPO=/root/autodl-tmp/_000n_00000000/_000n_uni/_001n_my_codes/_001n_projects/_005n_claude_code_projs/_001n_cc_translation/_010n_codes/_026n_clare/clare
RUNTIME=/root/autodl-tmp/_000n_00000000/_000n_uni/_001n_my_codes/_001n_projects/_005n_claude_code_projs/_001n_cc_translation/_010n_codes/_026n_clare/clare/_001n_my/_003n_clare_download

cd "$REPO"

mkdir -p "$RUNTIME"/{hf_home,hf_datasets,lerobot,tmp,outputs,pip_cache}

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export HF_HOME="$RUNTIME/hf_home"
export HF_DATASETS_CACHE="$RUNTIME/hf_datasets"
export HF_LEROBOT_HOME="$RUNTIME/lerobot"
export TMPDIR="$RUNTIME/tmp"
export PIP_CACHE_DIR="$RUNTIME/pip_cache"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

这些变量的目的：

- Hugging Face 模型缓存写到 `autodl-tmp`，避免写爆容器根目录。
- datasets Arrow cache 写到 `autodl-tmp`。
- LeRobot 视频/数据缓存写到 `autodl-tmp`。
- 临时文件写到 `autodl-tmp`。
- 训练输出和 checkpoint 写到 `$REPO/outputs`。

## 6. 下载基础 checkpoint

当前已经下载过；如需重新下载：

```bash
cd "$REPO"
mkdir -p ./outputs/dit_flow_mt_libero_90_pretrain

$ENV/bin/hf download continuallearning/dit_flow_mt_libero_90_pretrain \
  --local-dir ./outputs/dit_flow_mt_libero_90_pretrain
```

确认文件：

```bash
find ./outputs/dit_flow_mt_libero_90_pretrain -maxdepth 1 -type f -print | sort
du -sh ./outputs/dit_flow_mt_libero_90_pretrain
```

当前本机应包含：

```text
config.json
model.safetensors
train_config.json
README.md
.gitattributes
```

## 7. 已跑通的 smoke test

这个命令用于确认训练主链路是否正常，不是正式实验。它只训练 1 step adapter 和 1 step discriminator，已在本机跑通，退出码为 0。

```bash
ENV=/root/autodl-tmp/_000n_00000000/_003n_backend/_001n_z001/_001n_miniconda/_002n_conda_list/_008n_clare
REPO=/root/autodl-tmp/_000n_00000000/_000n_uni/_001n_my_codes/_001n_projects/_005n_claude_code_projs/_001n_cc_translation/_010n_codes/_026n_clare/clare
RUNTIME=/root/autodl-tmp/_000n_00000000/_000n_uni/_001n_my_codes/_001n_projects/_005n_claude_code_projs/_001n_cc_translation/_010n_codes/_026n_clare/clare/_001n_my/_003n_clare_download

cd "$REPO"

mkdir -p "$RUNTIME"/{hf_home,hf_datasets,lerobot,tmp} "$REPO/outputs/_smoke"

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export HF_HOME="$RUNTIME/hf_home"
export HF_DATASETS_CACHE="$RUNTIME/hf_datasets"
export HF_LEROBOT_HOME="$RUNTIME/lerobot"
export TMPDIR="$RUNTIME/tmp"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

$ENV/bin/python ./lerobot_lsy/src/lerobot/scripts/clare.py \
  --seed=42 \
  --job_name=clare_smoke_cu117_task0 \
  --output_dir="$REPO/outputs/_smoke/clare_cu117_task0_smoke" \
  --dataset.repo_id=continuallearning/libero_10_image_task_0 \
  --dataset.video_backend=pyav \
  --policy.path="$REPO/outputs/dit_flow_mt_libero_90_pretrain" \
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
  --wandb.enable=false
```

已验证关键日志类似：

```text
step:1 ... loss:0.293 ...
Checkpoint policy after step 1
Training discriminator
step:2 ... loss:0.309 ...
Checkpoint policy after step 2
End of training
```

如果重复运行同一个 `--output_dir`，且没有 `--resume=true`，脚本会因为输出目录已存在而拒绝覆盖。换一个输出目录或删掉旧 smoke 输出即可。

## 8. 正式训练：单个 LIBERO-10 task

下面是更接近 README 的正式训练模板，但默认关闭仿真评测，只跑离线训练和保存 checkpoint。这是当前环境已验证主链路的稳妥跑法。

默认参数：

```bash
SEED=42
TASK=0
STEPS=20000
DISC_STEPS=2000
BATCH_SIZE=32
NUM_WORKERS=16
```

如果 3090 显存不够，先把 `BATCH_SIZE` 改成 `16` 或 `8`，把 `NUM_WORKERS` 改成 `8` 或 `4`。

运行 task 0：

```bash
ENV=/root/autodl-tmp/_000n_00000000/_003n_backend/_001n_z001/_001n_miniconda/_002n_conda_list/_008n_clare
REPO=/root/autodl-tmp/_000n_00000000/_000n_uni/_001n_my_codes/_001n_projects/_005n_claude_code_projs/_001n_cc_translation/_010n_codes/_026n_clare/clare
RUNTIME=/root/autodl-tmp/_000n_00000000/_000n_uni/_001n_my_codes/_001n_projects/_005n_claude_code_projs/_001n_cc_translation/_010n_codes/_026n_clare/clare/_001n_my/_003n_clare_download

cd "$REPO"

mkdir -p "$RUNTIME"/{hf_home,hf_datasets,lerobot,tmp} "$REPO/outputs"

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export HF_HOME="$RUNTIME/hf_home"
export HF_DATASETS_CACHE="$RUNTIME/hf_datasets"
export HF_LEROBOT_HOME="$RUNTIME/lerobot"
export TMPDIR="$RUNTIME/tmp"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

SEED=42
TASK=0
STEPS=20000
DISC_STEPS=2000
BATCH_SIZE=32
NUM_WORKERS=16

OUT="$REPO/outputs/libero_10/clare/dit_flow_mt_cl_seed_${SEED}_libero_10_task_${TASK}_encoder_mlp_adapter_threshold_1_0"

$ENV/bin/python ./lerobot_lsy/src/lerobot/scripts/clare.py \
  --seed="$SEED" \
  --job_name="clare_libero_10_task_${TASK}" \
  --output_dir="$OUT" \
  --dataset.repo_id="continuallearning/libero_10_image_task_${TASK}" \
  --dataset.video_backend=pyav \
  --policy.path="$REPO/outputs/dit_flow_mt_libero_90_pretrain" \
  --policy.push_to_hub=false \
  --batch_size="$BATCH_SIZE" \
  --num_workers="$NUM_WORKERS" \
  --steps="$STEPS" \
  --env.type=libero \
  --env.benchmark=libero_10 \
  --env.task="Libero_10_Task_${TASK}" \
  --eval_freq=0 \
  --save_freq="$STEPS" \
  --log_freq=100 \
  --peft_cfg_path=./peft_lsy/peft_config/clare_dit_flow_encoder_adapter \
  --expand_threshold=1.0 \
  --detect_distribution_shift_steps=200 \
  --detect_distribution_shift_batch_size="$BATCH_SIZE" \
  --detect_distribution_shift_num_workers="$NUM_WORKERS" \
  --detect_distribution_shift_log_freq=10 \
  --train_discriminators_steps="$DISC_STEPS" \
  --train_discriminators_batch_size="$BATCH_SIZE" \
  --train_discriminators_num_workers="$NUM_WORKERS" \
  --train_discriminators_log_freq=50 \
  --train_discriminators_eval_freq=0 \
  --train_discriminators_save_freq="$DISC_STEPS" \
  --wandb.enable=false
```

输出重点看：

```bash
find "$OUT/checkpoints" -maxdepth 3 -type f | sort | sed -n '1,80p'
```

正常会有：

```text
checkpoints/last/adapter
checkpoints/last/pretrained_model
checkpoints/last/training_state
```

## 9. 正式训练：LIBERO-10 连续 10 个任务

下面这个循环按 task 0 到 task 9 顺序训练。task 1 之后会把上一任务的 adapter 通过 `--peft_weight_path` 传给下一任务。

默认关闭 W&B 和仿真评测，优先保证离线训练主链路稳定。需要 W&B 时，把 `--wandb.enable=false` 改成 `true`，并补 `--wandb.project` / `--wandb.entity`。

```bash
ENV=/root/autodl-tmp/_000n_00000000/_003n_backend/_001n_z001/_001n_miniconda/_002n_conda_list/_008n_clare
REPO=/root/autodl-tmp/_000n_00000000/_000n_uni/_001n_my_codes/_001n_projects/_005n_claude_code_projs/_001n_cc_translation/_010n_codes/_026n_clare/clare
RUNTIME=/root/autodl-tmp/_000n_00000000/_000n_uni/_001n_my_codes/_001n_projects/_005n_claude_code_projs/_001n_cc_translation/_010n_codes/_026n_clare/clare/_001n_my/_003n_clare_download

cd "$REPO"

mkdir -p "$RUNTIME"/{hf_home,hf_datasets,lerobot,tmp} "$REPO/outputs"

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export HF_HOME="$RUNTIME/hf_home"
export HF_DATASETS_CACHE="$RUNTIME/hf_datasets"
export HF_LEROBOT_HOME="$RUNTIME/lerobot"
export TMPDIR="$RUNTIME/tmp"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

SEED=42
STEPS=20000
DISC_STEPS=2000
BATCH_SIZE=32
NUM_WORKERS=16
ROOT="$REPO/outputs/libero_10/clare"
PRETRAIN="$REPO/outputs/dit_flow_mt_libero_90_pretrain"

for TASK in $(seq 0 9); do
  OUT="$ROOT/dit_flow_mt_cl_seed_${SEED}_libero_10_task_${TASK}_encoder_mlp_adapter_threshold_1_0"

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
    --eval_freq=0
    --save_freq="$STEPS"
    --log_freq=100
    --peft_cfg_path=./peft_lsy/peft_config/clare_dit_flow_encoder_adapter
    --expand_threshold=1.0
    --detect_distribution_shift_steps=200
    --detect_distribution_shift_batch_size="$BATCH_SIZE"
    --detect_distribution_shift_num_workers="$NUM_WORKERS"
    --detect_distribution_shift_log_freq=10
    --train_discriminators_steps="$DISC_STEPS"
    --train_discriminators_batch_size="$BATCH_SIZE"
    --train_discriminators_num_workers="$NUM_WORKERS"
    --train_discriminators_log_freq=50
    --train_discriminators_eval_freq=0
    --train_discriminators_save_freq="$DISC_STEPS"
    --wandb.enable=false
  )

  if [ "$TASK" -gt 0 ]; then
    PREV=$((TASK - 1))
    PREV_OUT="$ROOT/dit_flow_mt_cl_seed_${SEED}_libero_10_task_${PREV}_encoder_mlp_adapter_threshold_1_0"
    ARGS+=(--peft_weight_path="$PREV_OUT/checkpoints/last/adapter")
  fi

  echo "==== Running LIBERO-10 task ${TASK} ===="
  "$ENV/bin/python" ./lerobot_lsy/src/lerobot/scripts/clare.py "${ARGS[@]}"
done
```

## 10. 使用仓库自带 bash 脚本

仓库实际存在的脚本是：

```text
bash/clare/clare_libero_10.sh
bash/clare/clare_libero_goal.sh
bash/clare/clare_libero_spatial.sh
bash/clare/clare_libero_40_10_goal_spatial_object.sh
```

这些脚本目前不建议直接原样运行，原因：

- 默认 `PRETRAIN_PATH` 仍是 `./outputs/dit_flow_mt_libero_90_pretrain_new`，需要改成 `./outputs/dit_flow_mt_libero_90_pretrain`。
- 输出目录默认在 `./outputs`，这正是当前推荐位置；需要确认脚本里的 `CHECKPOINT_ROOT` 也指向 `$REPO/outputs`。
- 脚本里有 literal `<YOUR_WANDB_ENTITY>`，直接 shell 执行会出问题；需要替换成真实 entity，或把 W&B 关掉。
- 脚本默认会开启 discriminator 阶段评测，当前环境未单独验证仿真评测依赖。

如果要基于原脚本跑，至少先复制一份再改，不要直接改原始脚本：

```bash
cd "$REPO"
mkdir -p _001n_my/_002n_requirements_and_run/scripts
cp bash/clare/clare_libero_10.sh _001n_my/_002n_requirements_and_run/scripts/clare_libero_10_local.sh
```

然后手动修改复制出的脚本：

```text
PRETRAIN_PATH=./outputs/dit_flow_mt_libero_90_pretrain
CHECKPOINT_ROOT=$REPO/outputs
--wandb.enable=false
--train_discriminators_eval_freq=0
```

或者直接使用第 9 节的循环命令，少踩脚本里的 placeholder。

## 11. 恢复中断训练

普通继续训练使用 `--resume=true` 和 `--config_path`。注意 resume 模式不要再传 `--policy.path`，配置会从 checkpoint 里的 `train_config.json` 读取。

示例：

```bash
OUT="$REPO/outputs/libero_10/clare/dit_flow_mt_cl_seed_42_libero_10_task_0_encoder_mlp_adapter_threshold_1_0"

$ENV/bin/python ./lerobot_lsy/src/lerobot/scripts/clare.py \
  --resume=true \
  --config_path="$OUT/checkpoints/last/pretrained_model/train_config.json"
```

如果想从某个已完成任务的 adapter 开始下一个任务，不用 `--resume=true`，而是新建下一个任务的 `--output_dir`，并传：

```bash
--peft_weight_path="$PREV_OUT/checkpoints/last/adapter"
```

## 12. 开启仿真评测的边界

当前已验证的是离线训练主链路：

```text
dataset -> policy load -> PEFT wrapper -> adapter training -> discriminator training -> checkpoint save
```

当前环境里 `mujoco`、`robosuite`、`libero` 未作为 Python 包元数据检测到。smoke test 和上面的正式模板都设置：

```bash
--eval_freq=0
--train_discriminators_eval_freq=0
```

如果要复现论文里的在线仿真评测，需要先补齐并单独验证 LIBERO / robosuite / MuJoCo 栈，然后再把：

```bash
--eval_freq=200000
--train_discriminators_eval_freq=2000
```

或其他评测频率打开。

## 13. 常见问题

### 13.1 根目录空间不足

如果看到：

```text
OSError: [Errno 28] No space left on device
```

优先检查是否忘了设置这些变量：

```bash
echo "$HF_HOME"
echo "$HF_DATASETS_CACHE"
echo "$HF_LEROBOT_HOME"
echo "$TMPDIR"
```

它们都应该指向：

```text
/root/autodl-tmp/_000n_00000000/_000n_uni/_001n_my_codes/_001n_projects/_005n_claude_code_projs/_001n_cc_translation/_010n_codes/_026n_clare/clare/_001n_my/_003n_clare_download/...
```

### 13.2 torch.xpu 报错

如果看到：

```text
AttributeError: module 'torch' has no attribute 'xpu'
```

说明 `diffusers` 太新。固定为：

```bash
$ENV/bin/python -m pip install --no-deps diffusers==0.32.2 huggingface_hub==0.36.2
```

### 13.3 保存 checkpoint 时 NumPy uint32 报错

如果看到：

```text
TypeError: can't convert np.ndarray of type numpy.uint32
```

说明 `lerobot_lsy/src/lerobot/utils/random_utils.py` 的兼容补丁不在当前代码里。当前跑通版本已经修复：保存 NumPy RNG state 时转成 `int64`，恢复时转回 `uint32`。

### 13.4 torchcodec 警告

如果看到：

```text
'torchcodec' is not available in your platform, falling back to 'pyav' as a default decoder
```

这是当前 CUDA 11.7 / torch 2.0.1 方案下的预期警告。运行命令里保留：

```bash
--dataset.video_backend=pyav
```

### 13.5 输出目录已存在

如果看到：

```text
Output directory ... already exists and resume is False
```

解决方式三选一：

```bash
# 方式 1：换一个新的 output_dir
--output_dir="$REPO/outputs/..."

# 方式 2：确认不要旧结果后删除旧目录
rm -rf "$OUT"

# 方式 3：按第 11 节 resume
--resume=true --config_path="$OUT/checkpoints/last/pretrained_model/train_config.json"
```

## 14. 迁移到新机器运行

新机器硬件配置相同，只是路径不同。因此迁移时重点是：

1. 保留当前已经打过兼容补丁的源码。
2. 使用相同 Python / PyTorch / CUDA wheel / 关键 Python 包版本。
3. 在新机器上重新设置 `REPO`、`ENV`、`RUNTIME` 三个路径变量。
4. 先跑 smoke test，再跑正式训练。

### 14.1 需要复制哪些东西

推荐至少复制整个 CLARE 仓库目录：

```text
clare/
```

这个目录里包含当前已修改并跑通的源码补丁，例如：

```text
lerobot_lsy/src/lerobot/scripts/clare.py
lerobot_lsy/src/lerobot/utils/random_utils.py
peft_lsy/src/peft/tuners/...
```

建议一并复制：

```text
clare/outputs/dit_flow_mt_libero_90_pretrain
```

这是当前已跑通的基础 checkpoint。如果不复制，也可以在新机器上重新下载。

可选复制：

```text
clare/_001n_my/_003n_clare_download
```

这是 Hugging Face / datasets / LeRobot / pip 的缓存目录，复制后可减少重新下载。它不是必须的；如果新机器上缓存报错，可以删掉这个目录后按本文档重新下载。

如果只想传最小体积，至少保留：

```text
clare/lerobot_lsy
clare/peft_lsy
clare/README.md
clare/outputs/dit_flow_mt_libero_90_pretrain
clare/_001n_my/_002n_requirements_and_run/requirements_and_run.md
```

### 14.2 新机器路径变量

新机器上不要照抄旧机器的绝对路径。先按新机器实际位置设置：

```bash
NEW_REPO=/新机器/上的/实际路径/clare
ENV=/新机器/上的/conda环境路径/_008n_clare
RUNTIME="$NEW_REPO/_001n_my/_003n_clare_download"
```

之后本文档所有命令里的：

```bash
REPO=...
```

都换成：

```bash
REPO="$NEW_REPO"
```

### 14.3 推荐方式：打包迁移当前 conda 环境

如果两台机器系统和硬件相同，最稳妥的方法是把当前已跑通的 conda 环境打包过去。这样最容易保证版本一致。

旧机器上执行：

```bash
OLD_ENV=/root/autodl-tmp/_000n_00000000/_003n_backend/_001n_z001/_001n_miniconda/_002n_conda_list/_008n_clare
PACK_OUT=/root/autodl-tmp/_000n_00000000/clare_cu117_env.tar.gz

python -m pip install conda-pack
conda-pack -p "$OLD_ENV" -o "$PACK_OUT"
```

把下面两个东西传到新机器：

```text
clare/                         # 当前代码目录
clare_cu117_env.tar.gz          # 打包后的环境
```

新机器上解包：

```bash
ENV=/新机器/上的/conda环境路径/_008n_clare
mkdir -p "$ENV"
tar -xzf clare_cu117_env.tar.gz -C "$ENV"
"$ENV/bin/conda-unpack"
```

由于 editable 安装会记录源码路径，新机器路径不同后需要重新安装本地 editable 包：

```bash
REPO=/新机器/上的/实际路径/clare
cd "$REPO"

"$ENV/bin/python" -m pip install -e ./peft_lsy --no-deps
"$ENV/bin/python" -m pip install -e ./lerobot_lsy --no-deps
```

然后执行第 14.5 节的检查。

### 14.4 备选方式：在新机器重新建环境

如果不迁移 conda 环境，就在新机器上重建 Python 3.10 环境，并固定当前已验证的版本。

```bash
ENV=/新机器/上的/conda环境路径/_008n_clare
REPO=/新机器/上的/实际路径/clare

conda create -p "$ENV" python=3.10 -y
"$ENV/bin/python" -m pip install -U pip setuptools wheel
```

安装 PyTorch CUDA 11.7 wheel：

```bash
"$ENV/bin/python" -m pip install \
  torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
  --index-url https://download.pytorch.org/whl/cu117
```

安装当前已验证的关键版本：

```bash
"$ENV/bin/python" -m pip install \
  numpy==1.26.4 \
  datasets==3.6.0 \
  diffusers==0.32.2 \
  huggingface_hub==0.36.2 \
  transformers==4.48.3 \
  tokenizers==0.21.4 \
  av==17.1.0 \
  accelerate==1.14.0 \
  safetensors==0.8.0 \
  draccus==0.10.0 \
  einops==0.8.2 \
  wandb==0.28.0 \
  deepdiff \
  flask \
  gdown \
  gymnasium \
  h5py \
  imageio[ffmpeg] \
  jsonlines \
  omegaconf \
  packaging \
  pymunk \
  pynput \
  pyserial \
  pyzmq \
  termcolor \
  zarr
```

当前已跑通环境里 `opencv-python-headless==4.13.0.92`、`rerun-sdk==0.33.1` 与 `numpy==1.26.4` 在声明依赖上有冲突，但没有阻塞 CLARE smoke test。为了保持当前版本而不让 pip 自动升级 NumPy，这两个包用 `--no-deps` 安装：

```bash
"$ENV/bin/python" -m pip install --no-deps \
  opencv-python-headless==4.13.0.92 \
  rerun-sdk==0.33.1
```

安装本地修改版 PEFT 和 LeRobot：

```bash
cd "$REPO"
"$ENV/bin/python" -m pip install -e ./peft_lsy --no-deps
"$ENV/bin/python" -m pip install -e ./lerobot_lsy --no-deps
```

确保不要安装 `torchcodec`：

```bash
"$ENV/bin/python" -m pip uninstall -y torchcodec
```

如果 `pip` 后续自动升级了 torch，需要重新执行 PyTorch CUDA 11.7 的安装命令，把版本拉回：

```bash
"$ENV/bin/python" -m pip install \
  torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
  --index-url https://download.pytorch.org/whl/cu117
```

如果 `pip` 后续自动升级了 NumPy，也要拉回：

```bash
"$ENV/bin/python" -m pip install --no-deps numpy==1.26.4
```

### 14.5 新机器环境检查

在新机器上执行：

```bash
REPO=/新机器/上的/实际路径/clare
ENV=/新机器/上的/conda环境路径/_008n_clare
RUNTIME="$REPO/_001n_my/_003n_clare_download"

cd "$REPO"

mkdir -p "$RUNTIME"/{hf_home,hf_datasets,lerobot,tmp,pip_cache} "$REPO/outputs"

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export HF_HOME="$RUNTIME/hf_home"
export HF_DATASETS_CACHE="$RUNTIME/hf_datasets"
export HF_LEROBOT_HOME="$RUNTIME/lerobot"
export TMPDIR="$RUNTIME/tmp"
export PIP_CACHE_DIR="$RUNTIME/pip_cache"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

"$ENV/bin/python" - <<'PY'
import importlib.metadata as md
import torch
import diffusers, huggingface_hub
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import lerobot.scripts.clare as clare

for p in [
    "torch", "torchvision", "torchaudio", "diffusers", "huggingface_hub",
    "transformers", "tokenizers", "numpy", "datasets", "av", "accelerate",
    "safetensors", "draccus", "wandb", "lerobot", "peft",
]:
    try:
        print(f"{p}=={md.version(p)}")
    except md.PackageNotFoundError:
        print(f"{p} not installed")

print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

print("diffusers scheduler ok:", get_scheduler, DDIMScheduler, DDPMScheduler)
print("clare import ok:", clare.__file__)
PY
```

期望至少看到：

```text
torch==2.0.1
torchvision==0.15.2
torchaudio==2.0.2
diffusers==0.32.2
huggingface_hub==0.36.2
numpy==1.26.4
torch cuda: 11.7
cuda available: True
clare import ok: .../lerobot_lsy/src/lerobot/scripts/clare.py
```

### 14.6 新机器 checkpoint 准备

如果已经复制了：

```text
$REPO/outputs/dit_flow_mt_libero_90_pretrain
```

只需要检查：

```bash
find "$REPO/outputs/dit_flow_mt_libero_90_pretrain" -maxdepth 1 -type f -print | sort
```

应该能看到：

```text
.gitattributes
README.md
config.json
model.safetensors
train_config.json
```

如果没有复制 checkpoint，在新机器上重新下载：

```bash
mkdir -p "$REPO/outputs/dit_flow_mt_libero_90_pretrain"

"$ENV/bin/hf" download continuallearning/dit_flow_mt_libero_90_pretrain \
  --local-dir "$REPO/outputs/dit_flow_mt_libero_90_pretrain"
```

### 14.7 新机器 smoke test

环境检查和 checkpoint 都通过后，先跑 1-step smoke test：

```bash
REPO=/新机器/上的/实际路径/clare
ENV=/新机器/上的/conda环境路径/_008n_clare
RUNTIME="$REPO/_001n_my/_003n_clare_download"

cd "$REPO"

mkdir -p "$RUNTIME"/{hf_home,hf_datasets,lerobot,tmp,pip_cache} "$REPO/outputs/_smoke"

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export HF_HOME="$RUNTIME/hf_home"
export HF_DATASETS_CACHE="$RUNTIME/hf_datasets"
export HF_LEROBOT_HOME="$RUNTIME/lerobot"
export TMPDIR="$RUNTIME/tmp"
export PIP_CACHE_DIR="$RUNTIME/pip_cache"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

OUT="$REPO/outputs/_smoke/clare_cu117_new_machine_task0"
rm -rf "$OUT"

"$ENV/bin/python" ./lerobot_lsy/src/lerobot/scripts/clare.py \
  --seed=42 \
  --job_name=clare_smoke_cu117_new_machine_task0 \
  --output_dir="$OUT" \
  --dataset.repo_id=continuallearning/libero_10_image_task_0 \
  --dataset.video_backend=pyav \
  --policy.path="$REPO/outputs/dit_flow_mt_libero_90_pretrain" \
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
  --wandb.enable=false
```

跑通标志：

```text
step:1 ... loss:...
Checkpoint policy after step 1
Training discriminator
step:2 ... loss:...
Checkpoint policy after step 2
End of training
```

并确认 checkpoint 文件存在：

```bash
find "$OUT/checkpoints" -maxdepth 4 -type f | sort | sed -n '1,80p'
```

### 14.8 新机器正式训练

smoke test 通过后，直接使用第 8 节或第 9 节的正式训练命令。只需要替换：

```bash
REPO=/新机器/上的/实际路径/clare
ENV=/新机器/上的/conda环境路径/_008n_clare
RUNTIME="$REPO/_001n_my/_003n_clare_download"
```

训练输出仍统一写到：

```bash
$REPO/outputs
```

缓存仍统一写到：

```bash
$RUNTIME/hf_home
$RUNTIME/hf_datasets
$RUNTIME/lerobot
$RUNTIME/tmp
$RUNTIME/pip_cache
```

### 14.9 新机器常见迁移问题

如果 `import lerobot.scripts.clare` 指向旧机器路径，说明 editable 包没有在新机器重新安装。重新执行：

```bash
cd "$REPO"
"$ENV/bin/python" -m pip install -e ./peft_lsy --no-deps
"$ENV/bin/python" -m pip install -e ./lerobot_lsy --no-deps
```

如果 `torch.cuda.is_available()` 是 `False`，先检查：

```bash
nvidia-smi
"$ENV/bin/python" - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
PY
```

如果 `torch.version.cuda` 不是 `11.7`，重新安装 CUDA 11.7 wheel：

```bash
"$ENV/bin/python" -m pip install \
  torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
  --index-url https://download.pytorch.org/whl/cu117
```

如果新机器路径不同导致缓存异常，删除缓存后重建即可：

```bash
rm -rf "$RUNTIME/hf_home" "$RUNTIME/hf_datasets" "$RUNTIME/lerobot" "$RUNTIME/tmp"
mkdir -p "$RUNTIME"/{hf_home,hf_datasets,lerobot,tmp,pip_cache}
```

### 14.10 已整理好的新机器 bash 脚本

已将新机器 smoke test 和正式训练命令分别写成 bash，放在：

```text
clare/_001n_my/_004n_bashs
```

两个脚本都会从自身位置自动推断 `REPO`，因此新机器上通常只需要设置 `ENV`。

Smoke test：

```bash
cd /新机器/上的/实际路径/clare
ENV=/新机器/上的/conda环境路径/_008n_clare \
  bash _001n_my/_004n_bashs/_001n_new_machine_smoke_test.sh
```

正式训练 LIBERO-10 task 0 到 task 9：

```bash
cd /新机器/上的/实际路径/clare
ENV=/新机器/上的/conda环境路径/_008n_clare \
  bash _001n_my/_004n_bashs/_002n_new_machine_train_libero10.sh
```

只跑单个 task，例如 task 0：

```bash
ENV=/新机器/上的/conda环境路径/_008n_clare \
TASK_START=0 \
TASK_END=0 \
  bash _001n_my/_004n_bashs/_002n_new_machine_train_libero10.sh
```

显存不足时可降低 batch size 和 worker 数：

```bash
ENV=/新机器/上的/conda环境路径/_008n_clare \
BATCH_SIZE=16 \
NUM_WORKERS=8 \
  bash _001n_my/_004n_bashs/_002n_new_machine_train_libero10.sh
```

如果输出目录已存在，脚本默认拒绝覆盖。确认旧输出不要后再设置：

```bash
OVERWRITE=1
```
