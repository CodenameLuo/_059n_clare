# CLARE：通过自主适配器路由与扩展实现视觉-语言-动作模型的持续学习

[Ralf Römer](https://ralfroemer.com)<sup>1,\*</sup>,
[Yi Zhang](https://www.linkedin.com/in/yi-zhang-01a8aa245/)<sup>1,\*</sup>,
[Angela P. Schoellig](https://www.dynsyslab.org/prof-angela-schoellig/)<sup>1</sup>,

<sup>1</sup>慕尼黑工业大学

[![arXiv](https://img.shields.io/badge/arXiv-2601.09512-red)](https://arxiv.org/abs/2601.09512)
[![Website](https://img.shields.io/badge/Website-CLARE-blue)](https://tum-lsy.github.io/clare/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Models%20%26%20Datasets-yellow)](https://huggingface.co/continuallearning)
[![PyTorch](https://img.shields.io/badge/Python-PyTorch-orange.svg)](https://www.pytorch.org)

这是论文 *“CLARE: Continual Learning for Vision-Language-Action Models via Autonomous Adapter Routing and Expansion”* 的官方代码仓库。

<p align="center">
  <img src="../../clare_overview.png" alt="CLARE" width="50%"/>
</p>

> 摘要：为了让机器人学会复杂的操作任务，当前一种常见做法是使用任务特定数据对预训练的视觉-语言-动作模型（VLA）进行微调。然而，由于这种方案会更新已有表征，它并不适合真实世界中的长期运行场景。在这些场景中，机器人必须在保留已获得知识的同时，持续适应新的任务和环境。现有面向机器人的持续学习方法通常需要存储历史数据（样例），难以应对较长的任务序列，或者在部署时依赖任务标识符。为解决这些限制，我们提出了 CLARE，一个通用且参数高效的 VLA 非样例持续学习框架。CLARE 在选定的前馈层中引入轻量级模块化适配器，并在学习新任务时根据逐层特征相似度，只在必要位置自主扩展模型。部署阶段，基于自编码器的路由机制会动态激活最相关的适配器，而无需任务标签。通过在 LIBERO 基准上的大量实验，我们表明 CLARE 能够在学习新任务时保持高性能，同时避免早期任务发生灾难性遗忘，并显著优于甚至包含样例数据的方法。

## 项目结构

本代码库构建在 Hugging Face 的两个开源框架之上：

- **`lerobot_lsy/`**：修改版 [LeRobot](https://github.com/huggingface/lerobot)，用于设计、训练和微调视觉-语言-动作（VLA）模型
- **`peft_lsy/`**：修改版 [PEFT](https://github.com/huggingface/peft)，将 CLARE 算法实现为与 LoRA 兼容的适配器

## 预训练 Checkpoint 与数据集

我们在 🤗 Hugging Face 上提供了预训练模型 checkpoint 和 LIBERO 数据集：**[huggingface.co/continuallearning](https://huggingface.co/continuallearning)**

可用资源：
- **`dit_flow_mt_libero_90_pretrain_new`**：论文中描述的、在 LIBERO-90 上预训练的基础 VLA checkpoint
- **`libero_10_image_task_0` 到 `libero_10_image_task_9`**：LIBERO-10 基准数据集（也提供 Goal、Spatial、Object）

## 安装

### 前置条件
- Python 3.8+
- 支持 CUDA 的 GPU（推荐）
- Conda 或 Miniconda

### 配置步骤

1. **创建并激活 conda 环境**
   ```bash
   conda create -n clare python=3.10
   conda activate clare
   ```

2. **以 editable 模式安装 PEFT-LSY**
   ```bash
   cd peft_lsy
   pip install -e .
   cd ..
   ```

3. **以 editable 模式安装 LeRobot-LSY**
   ```bash
   cd lerobot_lsy
   pip install -e .
   cd ..
   ```

4. **安装额外依赖**（如有需要）
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## PEFT 配置

CLARE 使用自定义 PEFT 适配器配置。下面是一个 CLARE 适配器配置示例：

```json
{
  "peft_type": "CLARE",
  "task_type": null,
  "auto_mapping": {
    "base_model_class": "PeftWrapperPolicy",
    "parent_library": "__main__"
  },
  "base_model_name_or_path": null,
  "revision": null,
  "target_modules": ".*velocity_net.cond_proj",
  "inference_mode": true,
  "batch_first": true,
  "num_learned_task": 0,
  "feature_dim": 2576,
  "out_feature_dim": 512,
  "use_trainable_copy": false,
  "add_zero_init_conv_layer": false,
  "structure": {},
  "discriminator_cfg": {
    "type": "autoencoder",
    "batch_first": true,
    "feature_dim": 2576,
    "feature_fusion": false,
    "fused_feature_dim": null,
    "hidden_dim": 256,
    "latent_dim": 128,
    "num_tokens": 16,
    "lora_rank": 32,
    "lora_alpha": 32,
    "use_lora": false,
    "use_momentum": true,
    "momentum": 0.1,
    "max_batches_tracked": 2000
  },
  "func_adapter_cfg": {
    "hidden_dim": 1024,
    "lora_rank": 32,
    "lora_alpha": 32,
    "use_lora": false
  }
}
```

### 关键配置参数

- **`target_modules`**：用于匹配模型中需要应用适配器的层的正则表达式模式
- **`feature_dim`**：适配器输入特征的维度
- **`out_feature_dim`**：适配器变换后的输出维度
- **`discriminator_cfg`**：基于自编码器的路由机制配置
  - `type`：判别器类型（自编码器或其他类型）
  - `hidden_dim`：隐藏层维度
  - `latent_dim`：自编码器潜在空间维度
  - `use_momentum`：是否对特征统计使用动量
- **`func_adapter_cfg`**：功能适配器模块配置
- **`num_learned_task`**：目前已经学习过的任务数量

## 使用方法

### 在 LIBERO 基准上使用 CLARE 训练

主训练脚本位于 `lerobot_lsy/src/lerobot/scripts/clare.py`。下面给出在 LIBERO-10 上进行持续学习第一阶段训练的示例命令：

```bash
python ./lerobot_lsy/src/lerobot/scripts/clare.py \
    --seed=42 \
    --job_name=clare_libero_10_task_0 \
    --output_dir=./outputs/libero_10/clare/dit_flow_mt_cl_seed_42_libero_10_task_0 \
    --dataset.repo_id=continuallearning/libero_10_image_task_0 \
    --policy.path=./outputs/dit_flow_mt_libero_90_pretrain_new \
    --policy.push_to_hub=false \
    --batch_size=32 \
    --num_workers=16 \
    --steps=20000 \
    --env.type=libero \
    --env.benchmark=libero_10 \
    --env.task=Libero_10_Task_0 \
    --eval.batch_size=50 \
    --eval.n_episodes=100 \
    --eval.max_episodes_rendered=4 \
    --eval_freq=200000 \
    --save_freq=20000 \
    --log_freq=100 \
    --peft_cfg_path=./peft_lsy/peft_config/clare_dit_flow_encoder_adapter \
    --expand_threshold=1.0 \
    --detect_distribution_shift_steps=200 \
    --detect_distribution_shift_batch_size=32 \
    --detect_distribution_shift_num_workers=16 \
    --detect_distribution_shift_log_freq=10 \
    --train_discriminators_steps=2000 \
    --train_discriminators_batch_size=32 \
    --train_discriminators_num_workers=16 \
    --train_discriminators_log_freq=50 \
    --train_discriminators_eval_freq=2000 \
    --train_discriminators_save_freq=2000 \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=clare_experiments \
    --wandb.entity=<your-wandb-entity>
```

如需运行完整的持续学习实验，请使用仓库提供的 bash 脚本。先设置你的 W&B entity，然后运行：

```bash
# LIBERO-10（10 个任务）
bash bash/clare/dit_dec_libero_10.sh

# LIBERO-Goal（10 个任务）
bash bash/clare/dit_dec_libero_goal.sh

# LIBERO-Spatial（10 个任务）
bash bash/clare/dit_dec_libero_spatial.sh

# LIBERO-40（4 个 suite × 10 个任务）
bash bash/clare/dit_dec_libero_40_10_goal_spatial_object.sh
```

每个脚本都会按顺序运行完整任务序列，并将上一阶段的适配器 checkpoint 传递给下一阶段。

### 关键训练参数

#### 数据集与模型
- `--dataset.repo_id`：Hugging Face 数据集仓库 ID
- `--policy.path`：预训练 VLA 模型路径（本地路径或 Hugging Face repo ID）

#### 训练配置
- `--batch_size`：训练 batch size
- `--num_workers`：数据加载 worker 数量
- `--steps`：总训练步数
- `--seed`：用于复现实验的随机种子

#### CLARE 特定参数
- `--peft_cfg_path`：PEFT 配置 JSON 文件路径
- `--expand_threshold`：自主适配器扩展阈值（基于特征相似度）
- `--detect_distribution_shift_steps`：分布偏移检测步数
- `--train_discriminators_steps`：基于自编码器的路由机制训练步数

#### 评估
- `--env.type`：环境类型（例如 `libero`）
- `--env.benchmark`：基准套件（例如 `libero_10`、`libero_goal`、`libero_spatial`）
- `--env.task`：到目前为止已经见过的任务名称，以逗号分隔（例如 `Libero_10_Task_0,Libero_10_Task_1`）
- `--eval.n_episodes`：评估 episode 数量
- `--eval_freq`：评估频率（以训练 step 为单位）

#### 日志
- `--wandb.enable`：启用 Weights & Biases 日志
- `--wandb.project`：W&B 项目名称
- `--log_freq`：日志记录频率

<!-- ## 环境变量

运行实验前设置以下环境变量：

```bash
export DATASET_ROOT=/path/to/your/datasets
export POLICY_ROOT=/path/to/your/pretrained/policy
``` -->

## 评估

可以直接使用包含 LIBERO 的最新版 LeRobot 在 LIBERO 中运行评估，也可以单独安装 LIBERO：
```bash
cd ..
git clone git@github.com:ZhangYi1999/gym-libero.git
cd gym-libero
pip install -e .
python test/test_gym_libero.py  # 验证安装
```

## 引用

如果你觉得这项工作有帮助，请考虑引用我们的论文：

```bibtex
@article{clare,
  title={CLARE: Continual Learning for Vision-Language-Action Models via Autonomous Adapter Routing and Expansion},
  author={Ralf R{\"o}mer and Yi Zhang and Angela P. Schoellig},
  journal={arXiv preprint arXiv:2601.09512},
  year={2026}
}
```

## 许可证

本项目遵循与原始 LeRobot 和 PEFT 框架相同的许可证发布。

## 致谢

本工作构建于以下项目之上：
- Hugging Face 的 [LeRobot](https://github.com/huggingface/lerobot)
- Hugging Face 的 [PEFT](https://github.com/huggingface/peft)
- Bo Liu 等人的 [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)

我们感谢这些项目的作者所作出的开源贡献。
