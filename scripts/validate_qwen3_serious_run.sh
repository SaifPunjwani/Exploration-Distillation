#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[validate-qwen3] python syntax"
python3 -m py_compile \
  tmx/config.py \
  tmx/wandb_metrics.py \
  tmx/explorer.py \
  tmx/distill.py \
  tmx/pipeline.py \
  scripts/run_vllm_aime_benchmark.py

echo "[validate-qwen3] shell syntax"
bash -n \
  scripts/launch_dual_slice_training.sh \
  scripts/launch_qwen3_explorer_novelty_saturation.sh \
  scripts/launch_qwen3_explorer_novelty_checked.sh \
  scripts/launch_qwen3_explorer_fit_probe.sh \
  scripts/launch_qwen3_one_round_full_dapo.sh \
  scripts/launch_qwen3_one_round_checked.sh \
  scripts/run_qwen3_saved_aime24_avg32.sh \
  scripts/run_qwen3_saved_aime24_avg32_sharded.sh \
  scripts/sync_tpu_slice_repo.sh \
  scripts/qwen3_serious_remote_state.sh \
  scripts/show_qwen3_serious_run_commands.sh \
  scripts/launch_ew4a_qwen3_thinking_pipeline.sh \
  scripts/run_aime24_avg32_eval.sh \
  scripts/run_use1d_vllm_remote_eval.sh \
  scripts/run_tpu_slice_vllm_sharded_aime.sh

echo "[validate-qwen3] launcher geometry"
rg -n 'MODEL_NAME="\$\{MODEL_NAME:-Qwen/Qwen3-1\.7B\}"' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'MODEL_NAME="\$\{MODEL_NAME:-Qwen/Qwen3-1\.7B\}"' scripts/launch_qwen3_one_round_full_dapo.sh >/dev/null
rg -n 'DATASET_NAME="\$\{DATASET_NAME:-dapo_math_17k\}"' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'DATASET_NAME="\$\{DATASET_NAME:-dapo_math_17k\}"' scripts/launch_qwen3_one_round_full_dapo.sh >/dev/null
rg -n 'VLLM_ENV_CACHE_GCS_URI="\$\{VLLM_ENV_CACHE_GCS_URI:-gs://llmvae-checkpoints/two-model-exploration/shared_envs/v5lite_vllm_tpu_0\.13\.3_py311\.tar\.gz\}"' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'VLLM_ENV_CACHE_GCS_URI="\$\{VLLM_ENV_CACHE_GCS_URI:-gs://llmvae-checkpoints/two-model-exploration/shared_envs/v5lite_vllm_tpu_0\.13\.3_py311\.tar\.gz\}"' scripts/launch_qwen3_one_round_full_dapo.sh >/dev/null
rg -n 'SKIP_INFERENCE_SLICE="\$\{SKIP_INFERENCE_SLICE:-0\}"' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'SYNC_INFER_REPO="\$\{SYNC_INFER_REPO:-1\}"' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'BOOTSTRAP_VLLM="\$\{BOOTSTRAP_VLLM:-1\}"' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'VLLM_PREFETCH_MODEL="\$\{VLLM_PREFETCH_MODEL:-1\}"' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'LAUNCH_VLLM="\$\{LAUNCH_VLLM:-1\}"' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'MIN_READY_SERVERS="\$\{MIN_READY_SERVERS:-4\}"' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'ALLOW_PARTIAL_VLLM_SLICE="\$\{ALLOW_PARTIAL_VLLM_SLICE:-1\}"' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'PARTIAL_READY_DELAY_ATTEMPTS="\$\{PARTIAL_READY_DELAY_ATTEMPTS:-6\}"' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'GRPO_BATCH_SIZE="\$\{GRPO_BATCH_SIZE:-1\}"' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'GRPO_NUM_GENERATIONS="\$\{GRPO_NUM_GENERATIONS:-8\}"' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'ACTUAL_GRPO_BATCH_SIZE="\$\{ACTUAL_GRPO_BATCH_SIZE:-1\}"' scripts/launch_qwen3_one_round_full_dapo.sh >/dev/null
rg -n 'TRAIN_LAUNCH_MODE=ddp_local' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'TRAIN_LAUNCH_MODE=ddp_local' scripts/launch_qwen3_one_round_full_dapo.sh >/dev/null
rg -n 'DDP_VISIBLE_CHIPS="\$\{DDP_VISIBLE_CHIPS:-0\}"' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'DDP_NUM_PROCESSES="\$\{DDP_NUM_PROCESSES:-1\}"' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'DDP_VISIBLE_CHIPS="\$\{DDP_VISIBLE_CHIPS:-0\}"' scripts/launch_qwen3_one_round_full_dapo.sh >/dev/null
rg -n 'DDP_NUM_PROCESSES="\$\{DDP_NUM_PROCESSES:-1\}"' scripts/launch_qwen3_one_round_full_dapo.sh >/dev/null
rg -n 'BOOTSTRAP_PARALLELISM="\$\{BOOTSTRAP_PARALLELISM:-8\}"' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'VLLM_PREFETCH_PARALLELISM="\$\{VLLM_PREFETCH_PARALLELISM:-8\}"' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'VLLM_LAUNCH_PARALLELISM="\$\{VLLM_LAUNCH_PARALLELISM:-8\}"' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'BOOTSTRAP_PARALLELISM="\$\{BOOTSTRAP_PARALLELISM:-8\}"' scripts/launch_qwen3_one_round_full_dapo.sh >/dev/null
rg -n 'VLLM_PREFETCH_PARALLELISM="\$\{VLLM_PREFETCH_PARALLELISM:-8\}"' scripts/launch_qwen3_one_round_full_dapo.sh >/dev/null
rg -n 'VLLM_LAUNCH_PARALLELISM="\$\{VLLM_LAUNCH_PARALLELISM:-8\}"' scripts/launch_qwen3_one_round_full_dapo.sh >/dev/null
rg -n 'GRPO_GRAD_ACCUM="\$\{GRPO_GRAD_ACCUM:-2\}"' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'GRPO_GRAD_ACCUM="\$\{GRPO_GRAD_ACCUM:-2\}"' scripts/launch_qwen3_one_round_full_dapo.sh >/dev/null
rg -n 'ACTUAL_GRPO_GRAD_ACCUM="\$\{ACTUAL_GRPO_GRAD_ACCUM:-2\}"' scripts/launch_qwen3_one_round_full_dapo.sh >/dev/null
rg -n 'MAX_PROMPT_LEN="\$\{MAX_PROMPT_LEN:-2048\}"' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'MAX_COMPLETION_LEN="\$\{MAX_COMPLETION_LEN:-8192\}"' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'MAX_MODEL_LEN="\$\{MAX_MODEL_LEN:-10240\}"' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'MAX_PROMPT_LEN="\$\{MAX_PROMPT_LEN:-2048\}"' scripts/launch_qwen3_one_round_full_dapo.sh >/dev/null
rg -n 'MAX_COMPLETION_LEN="\$\{MAX_COMPLETION_LEN:-8192\}"' scripts/launch_qwen3_one_round_full_dapo.sh >/dev/null
rg -n 'MAX_MODEL_LEN="\$\{MAX_MODEL_LEN:-10240\}"' scripts/launch_qwen3_one_round_full_dapo.sh >/dev/null
rg -n 'TMX_STOP_AFTER_EXPLORER=1' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'TMX_STOP_AFTER_TRAJECTORIES=0' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'TMX_STOP_AFTER_EXPLORER=1' scripts/launch_qwen3_explorer_fit_probe.sh >/dev/null
rg -n 'TMX_TRAJ_USE_VLLM="\$\{TMX_TRAJ_USE_VLLM:-1\}"' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'TMX_GRPO_USE_VLLM="\$\{TMX_GRPO_USE_VLLM:-1\}"' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'TMX_GRPO_USE_VLLM="\$\{TMX_GRPO_USE_VLLM:-1\}"' scripts/launch_qwen3_one_round_full_dapo.sh >/dev/null
rg -n 'TMX_OPTIM="\$\{TMX_OPTIM:-adafactor\}"' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'TMX_OPTIM="\$\{TMX_OPTIM:-adafactor\}"' scripts/launch_qwen3_one_round_full_dapo.sh >/dev/null
rg -n 'TMX_EXPLORER_VLLM_TOP_P="\$\{TMX_EXPLORER_VLLM_TOP_P:-0.95\}"' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'TMX_EXPLORER_VLLM_TOP_K="\$\{TMX_EXPLORER_VLLM_TOP_K:-20\}"' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'TMX_TRAJ_VLLM_TOP_P="\$\{TMX_TRAJ_VLLM_TOP_P:-0.95\}"' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'TMX_TRAJ_VLLM_TOP_K="\$\{TMX_TRAJ_VLLM_TOP_K:-20\}"' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'TMX_EXPLORER_TRAIN_TOP_P="\$\{TMX_EXPLORER_TRAIN_TOP_P:-0.95\}"' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'TMX_EXPLORER_TRAIN_TOP_K="\$\{TMX_EXPLORER_TRAIN_TOP_K:-20\}"' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'TMX_EXPLORER_VLLM_TOP_P="\$\{TMX_EXPLORER_VLLM_TOP_P:-0.95\}"' scripts/launch_qwen3_one_round_full_dapo.sh >/dev/null
rg -n 'TMX_EXPLORER_VLLM_TOP_K="\$\{TMX_EXPLORER_VLLM_TOP_K:-20\}"' scripts/launch_qwen3_one_round_full_dapo.sh >/dev/null
rg -n 'TMX_TRAJ_VLLM_TOP_P="\$\{TMX_TRAJ_VLLM_TOP_P:-0.95\}"' scripts/launch_qwen3_one_round_full_dapo.sh >/dev/null
rg -n 'TMX_TRAJ_VLLM_TOP_K="\$\{TMX_TRAJ_VLLM_TOP_K:-20\}"' scripts/launch_qwen3_one_round_full_dapo.sh >/dev/null
rg -n 'TMX_EXPLORER_TRAIN_TOP_P="\$\{TMX_EXPLORER_TRAIN_TOP_P:-0.95\}"' scripts/launch_qwen3_one_round_full_dapo.sh >/dev/null
rg -n 'TMX_EXPLORER_TRAIN_TOP_K="\$\{TMX_EXPLORER_TRAIN_TOP_K:-20\}"' scripts/launch_qwen3_one_round_full_dapo.sh >/dev/null
rg -n 'TMX_EXPLORER_DECOUPLED_GRPO=1' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'TMX_EXPLORER_DECOUPLED_GRPO=1' scripts/launch_qwen3_one_round_full_dapo.sh >/dev/null
rg -n 'TMX_EXPLORER_DECOUPLED_TRAIN_FSDP=0' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'TMX_EXPLORER_DECOUPLED_TRAIN_FSDP=0' scripts/launch_qwen3_one_round_full_dapo.sh >/dev/null
rg -n 'TMX_DECOUPLED_TRAIN_TOKEN_CHUNK="\$\{TMX_DECOUPLED_TRAIN_TOKEN_CHUNK:-256\}"' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'TMX_DECOUPLED_TRAIN_TOKEN_CHUNK="\$\{TMX_DECOUPLED_TRAIN_TOKEN_CHUNK:-256\}"' scripts/launch_qwen3_one_round_full_dapo.sh >/dev/null
rg -n 'TMX_DECOUPLED_GRAD_CKPT="\$\{TMX_DECOUPLED_GRAD_CKPT:-0\}"' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'TMX_DECOUPLED_GRAD_CKPT="\$\{TMX_DECOUPLED_GRAD_CKPT:-0\}"' scripts/launch_qwen3_one_round_full_dapo.sh >/dev/null
rg -n 'TMX_DECOUPLED_ACCUM_MARK_STEP="\$\{TMX_DECOUPLED_ACCUM_MARK_STEP:-0\}"' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'TMX_DECOUPLED_ACCUM_MARK_STEP="\$\{TMX_DECOUPLED_ACCUM_MARK_STEP:-0\}"' scripts/launch_qwen3_one_round_full_dapo.sh >/dev/null
rg -n 'TMX_DECOUPLED_GRAD_CKPT=1' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'TMX_DECOUPLED_GRAD_CKPT=1' scripts/launch_qwen3_one_round_full_dapo.sh >/dev/null
rg -n -- '--train-selection-policy random' scripts/launch_dual_slice_training.sh >/dev/null
rg -n "DDP_VISIBLE_CHIPS='\\\$\\{DDP_VISIBLE_CHIPS:-0,1,2,3\\}'" scripts/launch_dual_slice_training.sh >/dev/null
rg -n "DDP_NUM_PROCESSES='\\\$\\{DDP_NUM_PROCESSES:-4\\}'" scripts/launch_dual_slice_training.sh >/dev/null
rg -n "TPU_SKIP_MDS_QUERY=1" scripts/launch_dual_slice_training.sh >/dev/null
rg -n "TPU_ACCELERATOR_TYPE='\\\$TRAIN_ACCELERATOR_TYPE'" scripts/launch_dual_slice_training.sh >/dev/null
rg -n "TMX_TRAJ_USE_VLLM='\\\$\\{TMX_TRAJ_USE_VLLM:-0\\}'" scripts/launch_dual_slice_training.sh >/dev/null
rg -n 'SKIP_INFERENCE_SLICE="\$\{SKIP_INFERENCE_SLICE:-0\}"' scripts/launch_dual_slice_training.sh >/dev/null
rg -n "TMX_GRPO_LOGPROB_CHUNK_SIZE='\\\$\\{TMX_GRPO_LOGPROB_CHUNK_SIZE:-2\\}'" scripts/launch_dual_slice_training.sh >/dev/null
rg -n "TMX_DECOUPLED_ACCUM_MARK_STEP='\\\$\\{TMX_DECOUPLED_ACCUM_MARK_STEP:-1\\}'" scripts/launch_dual_slice_training.sh >/dev/null
rg -n "TMX_DECOUPLED_ACCUM_MARK_RESET_SCOPE='\\\$\\{TMX_DECOUPLED_ACCUM_MARK_RESET_SCOPE:-0\\}'" scripts/launch_dual_slice_training.sh >/dev/null

echo "[validate-qwen3] prompt contract"
rg -n 'EXPLORATION_THINKING_MODE=think' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'EVAL_THINKING_MODE=think' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'ACTUAL_THINKING_MODE=think' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'EXPLORATION_RESPONSE_STYLE=reasoning_final' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'EVAL_RESPONSE_STYLE=reasoning_final' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'ANSWER_CONSTRAINT_MODE=none' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'ACTUAL_TARGET_CONTRACT="\$\{ACTUAL_TARGET_CONTRACT:-full_trace\}"' scripts/launch_qwen3_one_round_full_dapo.sh >/dev/null
rg -n -F '\boxed{}' tmx/prompting.py >/dev/null
rg -n 'enable_thinking' tmx/prompting.py >/dev/null

echo "[validate-qwen3] benchmark path"
rg -n 'BENCHMARK_NUM_ROLLOUTS=32' scripts/run_qwen3_saved_aime24_avg32.sh >/dev/null
rg -n 'BENCHMARK_NUM_ROLLOUTS=32' scripts/run_qwen3_saved_aime24_avg32_sharded.sh >/dev/null
rg -n 'THINKING_MODE=think' scripts/run_qwen3_saved_aime24_avg32.sh >/dev/null
rg -n 'THINKING_MODE=think' scripts/run_qwen3_saved_aime24_avg32_sharded.sh >/dev/null
rg -n 'BENCHMARK_DATASET_NAME=aime_2024' scripts/run_qwen3_saved_aime24_avg32_sharded.sh >/dev/null
rg -n 'BENCHMARK_ROLLOUT_PASS_K=1,4,8,16,32' scripts/run_qwen3_saved_aime24_avg32_sharded.sh >/dev/null
rg -n 'SKIP_SERVER_LAUNCH="\$\{SKIP_SERVER_LAUNCH:-1\}"' scripts/run_qwen3_saved_aime24_avg32_sharded.sh >/dev/null
rg -n 'SKIP_SERVER_LAUNCH="\$\{SKIP_SERVER_LAUNCH:-0\}"' scripts/run_tpu_slice_vllm_sharded_aime.sh >/dev/null
rg -n 'SKIP_SERVER_LAUNCH="\$\{SKIP_SERVER_LAUNCH:-0\}"' scripts/run_use1d_vllm_remote_eval.sh >/dev/null
rg -n 'rollout_accuracy' scripts/run_vllm_aime_benchmark.py >/dev/null

echo "[validate-qwen3] efficiency defaults"
rg -n 'TMX_SKIP_INTERMEDIATE_EVALS=1' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'TMX_SKIP_INTERMEDIATE_EVALS=1' scripts/launch_qwen3_one_round_full_dapo.sh >/dev/null
rg -n 'TMX_SKIP_FINAL_BENCHMARK_EVALS=1' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'TMX_SKIP_FINAL_BENCHMARK_EVALS=1' scripts/launch_qwen3_one_round_full_dapo.sh >/dev/null
rg -n 'TMX_FINAL_EVAL_ONLY_ACTUAL=1' scripts/launch_qwen3_one_round_full_dapo.sh >/dev/null
rg -n 'TMX_FINAL_EVAL_MAX_EXAMPLES=1' scripts/launch_qwen3_one_round_full_dapo.sh >/dev/null
rg -n 'TMX_WANDB_MINIMAL=1' scripts/launch_qwen3_explorer_novelty_saturation.sh >/dev/null
rg -n 'TMX_WANDB_MINIMAL=1' scripts/launch_qwen3_one_round_full_dapo.sh >/dev/null

echo "[validate-qwen3] guard rails"
rg -n 'TMX_ALLOW_LEGACY_QWEN3_THINKING_PIPELINE' scripts/launch_ew4a_qwen3_thinking_pipeline.sh >/dev/null
rg -n 'TMX_ALLOW_LEGACY_AIME24_AVG32_EVAL' scripts/run_aime24_avg32_eval.sh >/dev/null
rg -n 'CONFIRM_REMOTE_CLEANUP=1 bash scripts/qwen3_serious_remote_state.sh clean' scripts/show_qwen3_serious_run_commands.sh >/dev/null

echo "[validate-qwen3] slice ops"
rg -nF 'INFER_TPU_NAME="${INFER_TPU_NAME:?' scripts/qwen3_serious_remote_state.sh >/dev/null
rg -n 'tmx_gcloud alpha compute tpus tpu-vm ssh "\$INFER_TPU_NAME"' scripts/qwen3_serious_remote_state.sh >/dev/null
rg -n -- '--worker="\$worker"' scripts/qwen3_serious_remote_state.sh >/dev/null
rg -nF 'INFER_SSH_HOST="${INFER_SSH_HOST:?' scripts/launch_dual_slice_training.sh >/dev/null
rg -n 'DIRECT_SSH_HOST="\$INFER_SSH_HOST"' scripts/launch_dual_slice_training.sh >/dev/null
rg -n 'start_seed_archive_server\(\)' scripts/sync_tpu_slice_repo.sh >/dev/null
rg -n 'stop_seed_archive_server\(\)' scripts/sync_tpu_slice_repo.sh >/dev/null
rg -n 'curl -fsS .*seed_worker_ip.*FANOUT_HTTP_PORT' scripts/sync_tpu_slice_repo.sh >/dev/null
rg -n '\[sync\] full-slice sync detected; copying archive once to worker' scripts/sync_tpu_slice_repo.sh >/dev/null

echo "[validate-qwen3] online actual grpo"
rg -n 'actual_online_grpo_max_steps' tmx/config.py tmx/distill.py >/dev/null
rg -n 'actual_train_mode: str = "grpo"' tmx/config.py >/dev/null
rg -n 'sft_online_grpo' tmx/config.py tmx/pipeline.py >/dev/null
rg -n 'stopped_after_explorer' tmx/pipeline.py >/dev/null
rg -n 'online_grpo_uses_dataset_prompts' tmx/pipeline.py >/dev/null
rg -n 'grpo_metric_prefix: str = "grpo"' tmx/explorer.py >/dev/null
rg -n 'train_metric_prefix: Optional\[str\] = "train"' tmx/explorer.py >/dev/null
rg -n 'actual_grpo/reward/' tmx/wandb_metrics.py >/dev/null

echo "[validate-qwen3] checkpoint policy"
rg -n 'TMX_ACTUAL_GRPO_SAVE_STEPS="\$\{TMX_ACTUAL_GRPO_SAVE_STEPS:-25\}"' scripts/launch_qwen3_one_round_full_dapo.sh >/dev/null
rg -n 'TMX_ACTUAL_GRPO_SAVE_TOTAL_LIMIT="\$\{TMX_ACTUAL_GRPO_SAVE_TOTAL_LIMIT:-4\}"' scripts/launch_qwen3_one_round_full_dapo.sh >/dev/null
rg -n '_save_model_to_gcs_and_reload_vllm\(explorer_model, tokenizer, _run_name, role="explorer"\)' tmx/pipeline.py >/dev/null
rg -n '_save_model_to_gcs_and_reload_vllm\(actual_model, tokenizer, _run_name, role="actual"\)' tmx/pipeline.py >/dev/null
rg -n 'TMX_ACTUAL_GRPO_SAVE_STEPS' tmx/distill.py >/dev/null

echo "[validate-qwen3] ok"
