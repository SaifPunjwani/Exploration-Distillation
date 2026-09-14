"""Budget/lineage planning and real subprocess ordering for parallel breadth."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from expdis_jax.parallel_pipeline import build_plan, run_plan


def manifest(tmp_path):
    return {
        "config": {"output_dir": str(tmp_path / "run"),
                   "checkpoint_root": str(tmp_path / "checkpoints"),
                   "explorers_per_round": 3, "num_rounds": 4,
                   "round_novelty_schedule": "0.75,0.50,0.35,0.25"},
        "allocation": ["worker-a", "worker-b", "worker-c"],
        "explorers": [{"command": [sys.executable, "-m", "expdis_jax.pipeline"],
                       "devices": [f"worker-{c}"], "serving_urls": [f"http://serve-{c}:8000/v1"]}
                      for c in "abc"],
        "main": {"command": [sys.executable, "-m", "expdis_jax.pipeline"],
                 "devices": [f"worker-{c}" for c in "abc"],
                 "serving_urls": [f"http://serve-{c}:8000/v1" for c in "abc"]},
    }


def flags(job):
    argv = job["command"][3:]
    return dict(zip(argv[::2], argv[1::2]))


def test_parallel_plan_matches_paper_and_carries_only_main(tmp_path):
    m = manifest(tmp_path)
    m["config"]["vllm_server_urls"] = "http://wrong-shared-server:8000/v1"
    plan = build_plan(m)
    assert sum(j["steps"] for r in plan for j in r["explorers"]) == 200
    assert sum(r["main"]["steps"] for r in plan) == 100
    assert sum(j["expected_rows"] for r in plan for j in r["explorers"]) == 12800
    for index, r in enumerate(plan):
        assert [j["steps"] for j in r["explorers"]] == [17, 17, 16]
        assert len({flags(j)["--seed"] for j in r["explorers"]}) == 3
        expected_parent = plan[index - 1]["main"]["artifact"] if index else ""
        for j in [*r["explorers"], r["main"]]:
            assert flags(j)["--vllm-server-urls"] == j["env"]["EXPDIS_VLLM_SERVER_URLS"]
            assert "wrong-shared-server" not in flags(j)["--vllm-server-urls"]
            assert flags(j)["--init-weights-checkpoint"] == expected_parent
            assert flags(j)["--main-parent-checkpoint"] == expected_parent
            assert flags(j)["--resume-checkpoint"] == ""
            assert j["env"]["EXPDIS_DATASET_SHARD_INDEX"] == str(index)
        assert r["main"]["env"]["EXPDIS_EXISTING_TRAJECTORIES"].split(",") == [j["artifact"] for j in r["explorers"]]


@pytest.mark.parametrize("kind", ["devices", "serving_urls"])
def test_rejects_overlapping_explorer_resources(tmp_path, kind):
    m = manifest(tmp_path)
    m["explorers"][1][kind] = m["explorers"][0][kind]
    with pytest.raises(ValueError, match="disjoint"):
        build_plan(m)


def test_rejects_extra_devices_for_main(tmp_path):
    m = manifest(tmp_path)
    m["main"]["devices"].append("another-slice")
    with pytest.raises(ValueError, match="fixed allocation"):
        build_plan(m)


def test_paper_names_and_legacy_configurations_agree(monkeypatch):
    from expdis_jax.config import config_from_dict, parse_args
    paper = ["--explorers-per-round", "3", "--main-rl-max-steps", "100", "--main-parent-checkpoint", "/main"]
    old = ["--scouts-per-round", "3", "--actual-online-grpo-max-steps", "100", "--central-parent-checkpoint", "/main"]
    monkeypatch.setattr(sys, "argv", ["pipeline", *paper])
    expected = parse_args()
    monkeypatch.setattr(sys, "argv", ["pipeline", *old])
    assert parse_args() == expected
    assert config_from_dict({"explorers_per_round": 3, "main_parent_checkpoint": "/main"}) == expected
    with pytest.raises(ValueError, match="both names"):
        config_from_dict({"explorers_per_round": 3, "scouts_per_round": 2})


def test_processes_overlap_and_main_waits_for_both(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text('''import pathlib, sys, time
root, name = pathlib.Path(sys.argv[1]), sys.argv[2]
if name != "main":
    (root / (name + ".started")).touch()
    deadline = time.monotonic() + 5
    while len(list(root.glob("*.started"))) != 2:
        if time.monotonic() > deadline: raise RuntimeError("Explorers did not overlap")
        time.sleep(0.01)
    (root / (name + ".jsonl")).write_text('{"ok": true}\\n')
else:
    assert len(list(root.glob("*.jsonl"))) == 2
    (root / "checkpoint").mkdir()
''')
    jobs = [{"command": [sys.executable, str(worker), str(tmp_path), name],
             "env": {}, "output_dir": str(tmp_path / name), "steps": 1,
             "artifact": str(tmp_path / ("checkpoint" if name == "main" else name + ".jsonl")),
             "expected_rows": None if name == "main" else 1}
            for name in ["explorer-a", "explorer-b", "main"]]
    plan = [{"round": 1, "explorers": jobs[:2], "main": jobs[2]}]
    run_plan(plan, timeout_seconds=10)
    assert json.loads((tmp_path / "main" / "parallel_receipt.json").read_text())["explorer_updates"] == 2
    with pytest.raises(FileExistsError, match="fresh run"):
        run_plan(plan)


def test_failed_explorer_never_launches_main(tmp_path):
    job = {"command": [sys.executable, "-c", "raise SystemExit(7)"], "env": {},
           "output_dir": str(tmp_path / "failed"), "artifact": str(tmp_path / "missing"),
           "expected_rows": 1, "steps": 1}
    main = deepcopy(job)
    main["command"] = [sys.executable, "-c", "raise AssertionError('must not run')"]
    main["output_dir"] = str(tmp_path / "main")
    with pytest.raises(RuntimeError, match="stage failed"):
        run_plan([{"round": 1, "explorers": [job], "main": main}])
    assert not Path(main["output_dir"]).exists()
