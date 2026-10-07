# Concurrent explorers on a fixed allocation

`python -m expdis_jax.parallel_pipeline manifest.json` validates the manifest
and prints commands. Add `--run` to execute them. It launches the round's
explorer groups concurrently, waits for every selected trajectory pool, then
runs the filter, student SFT, and student RL. The next round starts every
model from that student checkpoint. It partitions the same 200/100 update
totals and uses the same deterministic prompt shards and novelty schedule as
the native driver.

## Deployment contract

- The driver runs once, outside the trainer process groups.
- Each slot's `command` is an argv list, executed without a shell. The driver
  appends the normal `expdis_jax.pipeline` CLI arguments. A scheduler wrapper
  must forward those arguments and environment variables to every rank, wait
  for all ranks, and return a nonzero status if any rank fails.
- `devices` identifies the complete training/serving allocation assigned to a
  slot. Explorer slots must be disjoint and together equal the student's
  allocation (the `main` entry in the manifest).
  The scheduler or device-visibility settings must enforce that declaration;
  the driver cannot independently inventory remote hardware.
- Each explorer has separate serving endpoints and reload workers. Configure
  the slot's reload transport and worker selectors in `env`; reloading one
  explorer must never touch another's servers. The launcher sets both sampling
  and reload endpoint URL lists from `serving_urls`. Each JAX process group also
  needs its own coordinator and host-side HTTP ports.
- `output_dir` and `checkpoint_root` are unique, absolute paths on a filesystem
  mounted at the same locations in the driver and all trainer groups. This
  launcher uses filesystem pools; the native pipeline additionally supports
  `gs://` checkpoint roots. Do not reuse an interrupted run's output prefix.
- The server must return sampled token IDs and acknowledge every policy reload.
  Use the same model/tokenizer build on trainer and serving workers.

## Manifest example

Replace the example wrappers and endpoints with your deployment. Each wrapper
must launch `python -m expdis_jax.pipeline "$@"` across its assigned trainer
ranks with correct distributed initialization. For a single-host slot,
`["python", "-m", "expdis_jax.pipeline"]` can be used directly with explicit
device visibility in `env`.

```json
{
  "config": {
    "model_name": "Qwen/Qwen3-1.7B",
    "output_dir": "/shared/expdis/run-001",
    "checkpoint_root": "/shared/expdis/checkpoints-001",
    "explorers_per_round": 3,
    "num_rounds": 4,
    "round_novelty_schedule": "0.75,0.50,0.35,0.25",
    "grpo_max_steps": 200,
    "main_rl_max_steps": 100,
    "seed": 0
  },
  "allocation": ["partition-a", "partition-b", "partition-c"],
  "explorers": [
    {"command": ["/path/to/launch-group-a"], "devices": ["partition-a"],
     "serving_urls": ["http://serve-a:8000/v1"], "env": {}},
    {"command": ["/path/to/launch-group-b"], "devices": ["partition-b"],
     "serving_urls": ["http://serve-b:8000/v1"], "env": {}},
    {"command": ["/path/to/launch-group-c"], "devices": ["partition-c"],
     "serving_urls": ["http://serve-c:8000/v1"], "env": {}}
  ],
  "main": {
    "command": ["/path/to/launch-main-group"],
    "devices": ["partition-a", "partition-b", "partition-c"],
    "serving_urls": ["http://serve-a:8000/v1", "http://serve-b:8000/v1", "http://serve-c:8000/v1"],
    "env": {}
  }
}
```

For this configuration each round launches 17/17/16 explorer updates and 25
student updates. The driver checks each pool contains its expected selected row
count, refuses an existing stage artifact, stops before the student stage if an
explorer fails, and writes `round_XX/parallel_receipt.json` with elapsed times and update
counts. Stage logs are in each output directory's `launcher.log`. Evaluation
is a separate invocation of `expdis_jax.eval` against the final student HF export;
the launcher disables automatic probes at round boundaries.

## Tests

The tests run the process launcher with fixture processes, including a check
that fails if explorers run one after another, and two-process JAX runs of the
RL, SFT, and checkpoint handoffs for R=2, K=2 with fixed completions in place
of vLLM. They do not cover concurrent real-model training or TPU
sub-allocation. Each round's `parallel_receipt.json` records elapsed times.
