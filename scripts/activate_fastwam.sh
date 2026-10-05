#!/usr/bin/env bash
# Source this file before running FastWAM from the repository root.
source /groups/yshang/an221229/envs/FastWAM/bin/activate
export DIFFSYNTH_MODEL_BASE_PATH=/groups/yshang/an221229/checkpoints/FastWAM
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export LIBERO_CONFIG_PATH=/groups/yshang/an221229/config/FastWAM
export HF_HOME=/groups/yshang/an221229/cache/huggingface
export MUJOCO_GL=egl
