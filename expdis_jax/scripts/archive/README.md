# Retired launchers (provenance only)

These scripts drove specific past campaigns on private TPU allocations or were
superseded by the scripts one directory up. They are kept as a record of how
the reported runs were launched and are not maintained as entry points.

| File | What it was |
|---|---|
| `launch_*_trc256_rollout.sh`, `launch_*_saif_v6e64_ew4.sh`, `launch_lambda075_uc1a_v5lite64_pair.sh` | env presets for `../launch_dapo_drgrpo_tpu.sh` bound to specific slices, λ values, and resume checkpoints |
| `launch_jax_multihost_v5lite.sh`, `launch_jax.sh` | earlier launchers with the retired 8-generation geometry; superseded by `../launch_jax_pipeline*.sh` (use `EXPDIS_PIPELINE_STOP_AFTER_EXPLORER=1` for an Explorer-only run) |
| `launch_lambda05_after_lambda0.sh`, `resume_fast_after_step50.sh`, `run_actual_from_sft_standalone.sh`, `run_actual_after_sft_with_reload.sh` | one-off control-plane helpers for particular runs (the pipeline now performs the Actual stage and the vLLM reload itself) |
| `monitor_jax_slice.sh`, `monitor_tpu_run.sh`, `overnight_status_guard.sh` | read-only campaign monitors |
| `mirror_wandb_from_log.py`, `mirror_gcs_to_hf.py`, `log_eval_progress_wandb.py`, `artifact_snapshot_uploader.sh` | recovery and migration utilities (W&B backfill from console logs, GCS→HF mirror, legacy GCS uploader) |
| `debug_logprob_paths.py` | developer diagnostic comparing log-probability code paths in `train.py` |

Home directories, ssh host aliases, IP lists, and past run identifiers that
these scripts used to default to have been removed; each script now requires
the corresponding environment variable to be set explicitly. The env presets
no longer name the GCP projects or TPU VMs of the original allocations either;
the `TRAIN_*`/`SERVE_*` coordinates must be supplied through the environment.
