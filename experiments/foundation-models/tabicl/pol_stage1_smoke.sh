#!/usr/bin/env bash
# TabICL v2 proof-of-life: stage-1 classifier pretraining, trimmed to a short smoke run.
# Verbatim arg set from upstream scripts/train_v2_clf_stage1.sh (soda-inria/tabicl@main,
# fetched 2026-07-17) — only these deviate: max_steps 500000→$MAX_STEPS, n_jobs 16→$N_JOBS,
# save_temp_every 500→50, save_perm_every 5000→$MAX_STEPS, checkpoint dir, and metrics go
# to MLflow instead of wandb (train_with_mlflow.py wrapper + --wandb_log True; the shim
# redirects wandb.init/log to the ambient AIR MLflow run — real wandb never imported).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPROC="${NPROC_PER_NODE:-1}"                 # 1 for A10/1xH100 smoke; 8 on GPU_8xH100
CKPT_DIR="${CKPT_DIR:-/tmp/tabicl-pol/ckpt}" # point at a UC volume for the restart test
MAX_STEPS="${MAX_STEPS:-100}"
N_JOBS="${N_JOBS:-8}"                        # CPU prior-gen workers PER RANK — watch GPU util vs this
# Continued pretraining (T1 / the customer's fusion->continue path): set CHECKPOINT_PATH to
# start from existing weights (upstream: --checkpoint_path + --only_load_model True chains stages)
CONTINUE_ARGS=""
if [ -n "${CHECKPOINT_PATH:-}" ]; then
  CONTINUE_ARGS="--checkpoint_path $CHECKPOINT_PATH --only_load_model True"
  echo "CONTINUING from checkpoint: $CHECKPOINT_PATH"
fi

nvidia-smi || true
python - <<'EOF'
import torch, os
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| gpus", torch.cuda.device_count())
print("cpus", os.cpu_count())
EOF

torchrun --standalone --nproc_per_node="$NPROC" "$SCRIPT_DIR/train_with_mlflow.py" \
            --wandb_log True \
            --device cuda \
            --dtype float32 \
            --np_seed 42 \
            --torch_seed 42 \
            --max_steps "$MAX_STEPS" \
            --batch_size 64 \
            --micro_batch_size 4 \
            --lr 8e-4 \
            --muon True \
            --beta1 0.9 \
            --weight_decay 0.01 \
            --use_cautious_wd False \
            --scheduler cosine_with_restarts \
            --warmup_proportion 0.01 \
            --cosine_num_cycles 1 \
            --cosine_amplitude_decay 1 \
            --cosine_lr_end 1e-7 \
            --gradient_clipping 10.0 \
            --prior_type graph_scm \
            --prior_device cpu \
            --n_jobs "$N_JOBS" \
            --batch_size_per_gp 4 \
            --min_features 1 \
            --max_features 100 \
            --max_classes 10 \
            --max_seq_len 1024 \
            --min_train_size 0.3 \
            --max_train_size 0.9 \
            --seq_len_per_gp True \
            --graph_noise False \
            --filter_unpredictable_graphs True \
            --filter_unpredictable_datasets True \
            --allow_act_warping False \
            --min_n_nodes 2 \
            --max_n_nodes 32 \
            --cauchy_dag_offset 0.0 \
            --embed_dim 128 \
            --col_num_blocks 3 \
            --col_nhead 8 \
            --col_num_inds 128 \
            --col_affine False \
            --col_feature_group same \
            --col_feature_group_size 3 \
            --col_target_aware True \
            --col_ssmax True \
            --row_num_blocks 3 \
            --row_nhead 8 \
            --row_num_cls 4 \
            --row_rope_base 100000 \
            --row_rope_interleaved False \
            --icl_num_blocks 12 \
            --icl_nhead 8 \
            --icl_ssmax True \
            --ssmax_type qassmax-mlp-elementwise \
            --ff_factor 2 \
            --norm_first True \
            --zero_init False \
            --use_flash_attn3 False \
            --checkpoint_dir "$CKPT_DIR" \
            --save_temp_every 50 \
            --save_perm_every "$MAX_STEPS" \
            $CONTINUE_ARGS

echo "--- checkpoints written:"
ls -la "$CKPT_DIR" || true
