"""Deterministic timing and action-execution checks without LIBERO or a GPU."""

from __future__ import annotations

import importlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def benchmark(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts" / "latency_test"
    monkeypatch.syspath_prepend(str(scripts))
    return importlib.import_module("benchmark_libero_feedback")


class FakeClock:
    def __init__(self, *, oversleep=0.0):
        self.now = 100.0
        self.oversleep = oversleep

    def perf_counter(self):
        return self.now

    def sleep(self, seconds):
        assert seconds >= 0
        self.now += seconds + self.oversleep


class FakeEnv:
    def __init__(self, clock, *, step_time=0.005, success_after=None, terminate_after=None):
        self.clock = clock
        self.step_time = step_time
        self.success_after = success_after
        self.terminate_after = terminate_after
        self.actions = []
        self.dispatches = []

    def step(self, action):
        self.dispatches.append(self.clock.perf_counter())
        self.actions.append(np.array(action, copy=True))
        self.clock.now += self.step_time
        step = len(self.actions)
        success = step == self.success_after
        terminated = success or step == self.terminate_after
        return step, 0.0, terminated, False, {"is_success": success}

    def reset(self, *args, **kwargs):
        raise AssertionError("run_episode must not include another episode's reset in completion time")


class FakePlanner:
    def __init__(self, clock, *, latencies=(0.01,), make_chunk=None):
        self.clock = clock
        self.latencies = latencies
        self.make_chunk = make_chunk
        self.observations = []

    def __call__(self, observation):
        call = len(self.observations)
        self.observations.append(observation)
        self.clock.now += self.latencies[min(call, len(self.latencies) - 1)]
        if self.make_chunk is not None:
            return self.make_chunk(call)
        chunk = np.zeros((40, 7), dtype=np.float64)
        chunk[:, 0] = call * 100 + np.arange(40)
        return chunk


@pytest.mark.parametrize("k", [1, 5, 10, 20, 40])
def test_replan_discards_tail_and_uses_latest_observation(benchmark, k):
    clock = FakeClock()
    env = FakeEnv(clock)
    planner = FakePlanner(clock)
    result = benchmark.run_episode(env, planner, 0, k=k, control_hz=20, max_steps=83, clock=clock)

    assert planner.observations == list(range(0, 83, k))
    assert [action[0] for action in env.actions] == [step // k * 100 + step % k for step in range(83)]
    assert result["steps"] == 83
    assert len(result["plans"]) == (83 + k - 1) // k
    assert result["sim_time_s"] == pytest.approx(83 / 20)
    assert result["success"] is False


def test_stall_rebases_dispatches_without_catchup_or_accumulated_misses(benchmark):
    clock = FakeClock()
    env = FakeEnv(clock)
    planner = FakePlanner(clock, latencies=(0.08,))
    result = benchmark.run_episode(env, planner, 0, k=40, control_hz=20, max_steps=3, clock=clock)

    assert env.dispatches == pytest.approx([100.08, 100.13, 100.18])
    assert result["wall_time_s"] == pytest.approx(0.185)
    assert [row["deadline_missed"] for row in result["actions"]] == [True, False, False]
    assert [row["ready_deadline_missed"] for row in result["actions"]] == [True, False, False]
    assert result["plans"][0]["pipeline_ms"] == pytest.approx(80)
    assert result["plans"][0]["e2e_first_action_ms"] == pytest.approx(80)


def test_sensor_age_uses_source_observation_for_whole_executed_chunk(benchmark):
    clock = FakeClock()
    env = FakeEnv(clock)
    planner = FakePlanner(clock)
    result = benchmark.run_episode(env, planner, 0, k=5, control_hz=20, max_steps=6, clock=clock)

    assert [row["observation_age_ms"] for row in result["actions"]] == pytest.approx(
        [50, 100, 150, 200, 250, 45]
    )
    assert [row["observation_age_steps"] for row in result["actions"]] == [0, 1, 2, 3, 4, 0]
    assert [row["is_boundary"] for row in result["actions"]] == [False] * 5 + [True]
    assert [plan["pipeline_ms"] for plan in result["plans"]] == pytest.approx([10, 10])
    assert [plan["e2e_first_action_ms"] for plan in result["plans"]] == pytest.approx([50, 45])


def test_sleep_overshoot_is_an_actual_miss_but_not_a_compute_miss(benchmark):
    clock = FakeClock(oversleep=0.003)
    env = FakeEnv(clock)
    planner = FakePlanner(clock)
    result = benchmark.run_episode(
        env, planner, 0, k=40, control_hz=20, max_steps=2, deadline_tolerance_s=0.001, clock=clock
    )

    assert env.dispatches == pytest.approx([100.053, 100.106])
    assert [row["deadline_missed"] for row in result["actions"]] == [True, True]
    assert [row["ready_deadline_missed"] for row in result["actions"]] == [False, False]


def test_replan_stall_appears_at_block_boundary(benchmark):
    clock = FakeClock()
    env = FakeEnv(clock)
    planner = FakePlanner(clock, latencies=(0.01, 0.08))
    result = benchmark.run_episode(env, planner, 0, k=2, control_hz=20, max_steps=4, clock=clock)

    assert env.dispatches == pytest.approx([100.05, 100.10, 100.185, 100.235])
    assert [row["deadline_missed"] for row in result["actions"]] == [False, False, True, False]
    assert result["actions"][2]["is_boundary"] is True
    assert result["actions"][2]["boundary_interval_error_ms"] == pytest.approx(35)


def test_block_action_discontinuity_excludes_gripper_coordinate(benchmark):
    def make_chunk(call):
        chunk = np.zeros((40, 7), dtype=np.float64)
        chunk[:, :6] = (call * 10 + np.arange(40))[:, None]
        chunk[:, 6] = np.where(np.arange(40) % 2, -1000, 1000)
        return chunk

    clock = FakeClock()
    env = FakeEnv(clock)
    planner = FakePlanner(clock, make_chunk=make_chunk)
    result = benchmark.run_episode(env, planner, 0, k=2, control_hz=20, max_steps=4, clock=clock)

    assert result["actions"][1]["action_delta_l2"] == pytest.approx(np.sqrt(6))
    assert result["actions"][2]["action_delta_l2"] == pytest.approx(9 * np.sqrt(6))
    assert result["actions"][2]["action_accel_l2"] == pytest.approx(8 * np.sqrt(6))


@pytest.mark.parametrize("successful", [True, False])
def test_terminal_episode_stops_before_reset_or_further_planning(benchmark, successful):
    clock = FakeClock()
    env = FakeEnv(clock, success_after=3 if successful else None, terminate_after=3)
    planner = FakePlanner(clock)
    result = benchmark.run_episode(env, planner, 0, k=5, control_hz=20, max_steps=520, clock=clock)

    assert result["success"] is successful
    assert result["steps"] == 3
    assert result["wall_time_s"] == pytest.approx(0.155)
    assert result["sim_time_s"] == pytest.approx(0.15)
    assert len(env.actions) == 3
    assert planner.observations == [0]


@pytest.mark.parametrize(
    "bad_chunk",
    [
        np.zeros((39, 7)),
        np.zeros((40, 6)),
        np.zeros((1, 40, 7)),
        np.full((40, 7), np.nan),
        np.full((40, 7), np.inf),
    ],
    ids=["short-horizon", "wrong-action-dimension", "extra-batch-dimension", "nan", "infinity"],
)
def test_invalid_chunk_fails_before_action_dispatch(benchmark, bad_chunk):
    clock = FakeClock()
    env = FakeEnv(clock)
    planner = FakePlanner(clock, make_chunk=lambda _: bad_chunk)
    with pytest.raises(ValueError):
        benchmark.run_episode(env, planner, 0, k=1, control_hz=20, max_steps=1, clock=clock)
    assert env.actions == []


@pytest.mark.parametrize("k", [0, -1, 41])
def test_invalid_execution_horizon_rejected_before_planning(benchmark, k):
    clock = FakeClock()
    env = FakeEnv(clock)
    planner = FakePlanner(clock)
    with pytest.raises(ValueError):
        benchmark.run_episode(env, planner, 0, k=k, control_hz=20, max_steps=1, clock=clock)
    assert planner.observations == []
    assert env.actions == []


def test_report_keeps_failures_and_missing_samples_out_of_completion_estimates(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts" / "latency_test"))
    report = importlib.import_module("libero_feedback_report")
    metadata = {"horizon": 40, "ks": [1, 5, 40], "task_ids": [0], "n_episodes": 2}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    episode_dir = tmp_path / "episodes"
    episode_dir.mkdir()
    for k, episode_index, success, wall_time in (
        (1, 0, True, 10),
        (1, 1, False, 100),
        (40, 0, True, 20),
        (40, 1, True, 30),
    ):
        episode = {
            "k": k,
            "task_id": 0,
            "task_name": "test task",
            "episode_index": episode_index,
            "seed": 42 + episode_index,
            "init_state_id": episode_index,
            "success": success,
            "wall_time_s": wall_time,
            "sim_time_s": 0.1,
            "steps": 2,
            "actions": [
                {"step": 0, "is_boundary": False, "action_delta_l2": 999, "deadline_missed": False},
                {"step": 1, "is_boundary": True, "action_delta_l2": 2, "deadline_missed": True},
            ],
            "plans": [{"pipeline_ms": 10, "e2e_first_action_ms": 50}],
        }
        path = episode_dir / f"task_0_episode_{episode_index}_k_{k}.json"
        path.write_text(json.dumps(episode))
    # Interrupted writes must not count toward experimental coverage.
    (episode_dir / "task_0_episode_0_k_5.json.tmp").write_text("unfinished")

    summary = report.build_report(tmp_path)
    k1, missing, _ = summary["by_k"]
    assert summary["coverage"]["complete"] is False
    assert summary["coverage"]["completed_episodes"] == 4
    assert summary["coverage"]["expected_episodes"] == 6
    assert k1["success_rate"] == pytest.approx(0.5)
    assert k1["completion_wall_s_success_only"]["mean"] == 10
    assert k1["episode_wall_s_all"]["mean"] == 55
    assert k1["deadline_missed"]["fraction"] == pytest.approx(0.5)
    assert k1["boundary_action_delta_l2"]["mean"] == 2
    assert k1["interior_action_delta_l2"]["mean"] is None
    assert k1["paired_vs_k40"]["matched_episodes"] == 2
    assert k1["paired_vs_k40"]["both_successful_episodes"] == 1
    assert k1["paired_vs_k40"]["baseline_only_successes"] == 1
    assert k1["paired_vs_k40"]["wall_delta_s_both_success"]["mean"] == -10
    assert k1["paired_vs_k40"]["wall_ratio_both_success"]["mean"] == pytest.approx(0.5)
    assert missing["success_rate"] is None
    assert missing["completion_wall_s_success_only"]["count"] == 0
    assert missing["completion_wall_s_success_only"]["mean"] is None
    for name in ("summary.json", "summary.csv", "actions.csv", "comparison.png", "comparison.pdf"):
        assert (tmp_path / name).stat().st_size > 0


@pytest.fixture
def fake_libero_resources(benchmark, monkeypatch, tmp_path):
    pytest.importorskip("yaml")
    install = tmp_path / "site-packages" / "libero"
    install.mkdir(parents=True)
    home = tmp_path / "home"
    home.mkdir()
    output = tmp_path / "experiment"
    output.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("LIBERO_CONFIG_PATH", raising=False)
    original_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        benchmark.importlib.util,
        "find_spec",
        lambda name: (
            SimpleNamespace(submodule_search_locations=[str(install)])
            if name == "libero"
            else original_find_spec(name)
        ),
    )
    package = ModuleType("libero")
    package.__path__ = [str(install)]
    module = ModuleType("libero.libero")
    module._assets_path_cache = None
    package.libero = module
    monkeypatch.setitem(sys.modules, "libero", package)
    monkeypatch.setitem(sys.modules, "libero.libero", module)
    hub = ModuleType("huggingface_hub")
    calls = []
    asset_directories = (
        "articulated_objects",
        "stable_scanned_objects",
        "turbosquid_objects",
        "stable_hope_objects",
        "scenes",
    )

    def populate_assets(directory, *, scenes=True):
        for name in asset_directories:
            if scenes or name != "scenes":
                (directory / name).mkdir(parents=True, exist_ok=True)

    def snapshot_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        # HF Hub 1.x rejects LIBERO's obsolete local_dir_use_symlinks argument.
        assert set(kwargs) == {"repo_type", "local_dir", "local_files_only"}
        populate_assets(Path(kwargs["local_dir"]))
        return str(kwargs["local_dir"])

    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return SimpleNamespace(
        install=install,
        home=home,
        output=output,
        module=module,
        calls=calls,
        populate_assets=populate_assets,
    )


def test_resource_setup_is_noninteractive_and_requires_scene_assets(benchmark, fake_libero_resources):
    import yaml

    resources = fake_libero_resources
    assets = resources.home / ".cache/libero/assets"
    resources.populate_assets(assets, scenes=False)
    args = SimpleNamespace(libero_assets=None, local_files_only=True)
    benchmark._prepare_libero_resources(args, resources.output)

    assert len(resources.calls) == 1
    repo_id, kwargs = resources.calls[0]
    assert repo_id == "lerobot/libero-assets"
    assert kwargs["repo_type"] == "dataset"
    assert kwargs["local_files_only"] is True
    assert kwargs["local_dir"] == assets
    assert resources.module._assets_path_cache == str(assets)
    config = yaml.safe_load((resources.output / "libero_config" / "config.yaml").read_text())
    assert config["bddl_files"] == str(resources.install / "libero" / "bddl_files")
    assert config["init_states"] == str(resources.install / "libero" / "init_files")
    assert config["assets"] == str(assets)
    assert not (resources.home / ".libero").exists()


def test_resource_setup_preserves_existing_config_and_uses_explicit_assets(
    benchmark, fake_libero_resources, monkeypatch
):
    resources = fake_libero_resources
    directory = resources.output / "existing-config"
    directory.mkdir()
    config_path = directory / "config.yaml"
    original = "# Custom benchmark paths must survive.\nbenchmark_root: /custom/libero\n"
    config_path.write_text(original)
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(directory))
    assets = resources.output / "custom-assets"
    resources.populate_assets(assets)
    args = SimpleNamespace(libero_assets=str(assets), local_files_only=False)
    benchmark._prepare_libero_resources(args, resources.output)

    assert config_path.read_text() == original
    assert resources.calls == []
    assert resources.module._assets_path_cache == str(assets)
    recorded = json.loads((resources.output / "libero_resources.json").read_text())
    assert recorded == {"config_dir": str(directory), "assets": str(assets)}


def test_explicit_incomplete_assets_fail_without_implicit_download(benchmark, fake_libero_resources):
    resources = fake_libero_resources
    assets = resources.output / "incomplete-assets"
    resources.populate_assets(assets, scenes=False)
    args = SimpleNamespace(libero_assets=str(assets), local_files_only=False)
    with pytest.raises(FileNotFoundError, match="Incomplete --libero-assets"):
        benchmark._prepare_libero_resources(args, resources.output)
    assert resources.calls == []


@pytest.fixture
def raw_observation():
    return {
        "pixels": {"image": np.arange(24, dtype=np.uint8).reshape(2, 4, 3)},
        "robot_state": {"eef": {"pos": np.asarray([0.1, 0.2, 0.3], dtype=np.float64)}},
    }


def test_observation_fingerprint_ignores_mapping_order_and_array_memory_layout(benchmark, raw_observation):
    reordered = {
        "robot_state": deepcopy(raw_observation["robot_state"]),
        "pixels": {"image": np.asfortranarray(raw_observation["pixels"]["image"])},
    }
    assert benchmark.observation_fingerprint(raw_observation) == benchmark.observation_fingerprint(reordered)


@pytest.mark.parametrize("change", ["image-value", "image-shape", "image-dtype", "nested-state"])
def test_observation_fingerprint_detects_material_input_changes(benchmark, raw_observation, change):
    modified = deepcopy(raw_observation)
    if change == "image-value":
        modified["pixels"]["image"][0, 0, 0] += 1
    elif change == "image-shape":
        modified["pixels"]["image"] = modified["pixels"]["image"].reshape(4, 2, 3)
    elif change == "image-dtype":
        modified["pixels"]["image"] = modified["pixels"]["image"].astype(np.float32)
    else:
        modified["robot_state"]["eef"]["pos"][0] += 0.01
    assert benchmark.observation_fingerprint(raw_observation) != benchmark.observation_fingerprint(modified)


@pytest.fixture
def schema2_report(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts" / "latency_test"))
    report = importlib.import_module("libero_feedback_report")
    # Plot artifacts are covered above; these tests exercise validation before reporting.
    monkeypatch.setattr(report, "_plot", lambda *args: None)
    metadata = {"schema_version": 2, "horizon": 40, "ks": [1, 40], "task_ids": [0], "n_episodes": 2}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "episodes").mkdir()

    def write_episode(k, episode_index, initial_hash):
        episode = {
            "k": k,
            "task_id": 0,
            "task_name": "paired task",
            "episode_index": episode_index,
            "seed": 42 + episode_index,
            "init_state_id": episode_index,
            "success": True,
            "wall_time_s": 1.0,
            "sim_time_s": 0.05,
            "steps": 1,
            "actions": [{"step": 0, "is_boundary": False}],
            "plans": [],
        }
        if initial_hash is not None:
            episode["initial_observation_sha256"] = initial_hash
        path = tmp_path / "episodes" / f"task_0_episode_{episode_index}_k_{k}.json"
        path.write_text(json.dumps(episode))

    return report, write_episode


@pytest.mark.parametrize(
    ("second_hash", "message"),
    [
        ("b" * 64, "SHA-256 mismatch across K"),
        (None, "Missing or invalid initial observation SHA-256"),
        ("g" * 64, "Missing or invalid initial observation SHA-256"),
        ("a" * 63, "Missing or invalid initial observation SHA-256"),
    ],
    ids=["different-input", "missing-hash", "non-hex-hash", "short-hash"],
)
def test_schema2_report_rejects_unverified_pairs_before_writing(
    schema2_report, tmp_path, second_hash, message
):
    report, write_episode = schema2_report
    write_episode(1, 0, "a" * 64)
    write_episode(40, 0, second_hash)
    with pytest.raises(ValueError, match=message):
        report.build_report(tmp_path)
    assert not (tmp_path / "summary.json").exists()
    assert not (tmp_path / "summary.csv").exists()


def test_schema2_report_distinguishes_verified_available_pairs_from_complete_coverage(
    schema2_report, tmp_path
):
    report, write_episode = schema2_report
    write_episode(1, 0, "a" * 64)
    write_episode(40, 0, "a" * 64)
    write_episode(1, 1, "b" * 64)
    summary = report.build_report(tmp_path)
    protocol = summary["protocol"]
    assert protocol["initial_observation_pairing_verified"] is True
    assert protocol["initial_observation_pairing_groups"] == 2
    assert protocol["initial_observation_pairing_complete_groups"] == 1
    assert protocol["initial_observation_pairing_partial_groups"] == 1
    assert summary["coverage"]["complete"] is False


def test_invalidated_experiment_cannot_regenerate_or_overwrite_report(schema2_report, tmp_path):
    report, write_episode = schema2_report
    write_episode(1, 0, "a" * 64)
    (tmp_path / "INVALIDATED.md").write_text("Initial observation pairing failed; rerun required.\n")
    original = '{"archived": true}\n'
    (tmp_path / "summary.json").write_text(original)
    with pytest.raises(ValueError, match="Refusing to regenerate an invalidated experiment report"):
        report.build_report(tmp_path)
    assert (tmp_path / "summary.json").read_text() == original
    assert not (tmp_path / "summary.csv").exists()


@pytest.mark.parametrize(
    ("suite", "task_count", "step_limit"), [("libero_10", 10, 520), ("libero_90", 90, 400)]
)
def test_suite_cli_resolves_task_range_and_episode_limit(
    benchmark, monkeypatch, capsys, suite, task_count, step_limit
):
    monkeypatch.setattr(
        sys, "argv", ["benchmark_libero_feedback", "--suite", suite, "--n-episodes", "5", "--dry-run"]
    )
    benchmark.main()
    metadata = json.loads(capsys.readouterr().out)
    assert metadata["suite"] == suite
    assert metadata["suite_task_count"] == task_count
    assert metadata["suite_max_steps"] == step_limit
    assert metadata["task_ids"] == list(range(task_count))
    assert metadata["max_steps"] == step_limit
    assert metadata["n_episodes"] == 5
    assert metadata["ks"] == [1, 5, 10, 20, 40]
    assert len(metadata["task_ids"]) * metadata["n_episodes"] * len(metadata["ks"]) == task_count * 25


@pytest.mark.parametrize(
    "arguments",
    [
        ["--suite", "libero_90", "--task-ids", "90"],
        ["--suite", "libero_10", "--task-ids", "10"],
        ["--suite", "libero_90", "--max-steps", "401"],
        ["--suite", "libero_10", "--max-steps", "521"],
    ],
    ids=[
        "libero90-task-out-of-range",
        "libero10-task-out-of-range",
        "libero90-episode-too-long",
        "libero10-episode-too-long",
    ],
)
def test_suite_cli_rejects_task_ids_and_step_limits_outside_selected_suite(benchmark, monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["benchmark_libero_feedback", *arguments, "--dry-run"])
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 2


def test_libero90_cli_accepts_high_task_ids_and_smoke_limit(benchmark, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_libero_feedback",
            "--suite",
            "libero_90",
            "--task-ids",
            "80",
            "89",
            "--max-steps",
            "45",
            "--dry-run",
        ],
    )
    benchmark.main()
    metadata = json.loads(capsys.readouterr().out)
    assert metadata["task_ids"] == [80, 89]
    assert metadata["max_steps"] == 45
    assert metadata["suite_max_steps"] == 400


def test_libero90_report_counts_five_rollouts_per_task_per_k_as_2250(schema2_report, tmp_path):
    report, write_episode = schema2_report
    metadata = {
        "schema_version": 2,
        "suite": "libero_90",
        "suite_task_count": 90,
        "suite_max_steps": 400,
        "horizon": 40,
        "ks": [1, 5, 10, 20, 40],
        "task_ids": list(range(90)),
        "n_episodes": 5,
        "max_steps": 400,
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    write_episode(1, 0, "a" * 64)
    summary = report.build_report(tmp_path)
    protocol = summary["protocol"]
    assert protocol["suite"] == "libero_90"
    assert protocol["suite_label"] == "LIBERO 90"
    assert protocol["suite_task_count"] == protocol["selected_tasks"] == 90
    assert protocol["episodes_per_task_per_k"] == 5
    assert protocol["expected_rollouts_per_k"] == 450
    assert protocol["full_task_suite"] is True
    assert protocol["max_steps"] == protocol["suite_max_steps"] == 400
    assert protocol["shortened_episode_limit"] is False
    assert summary["coverage"]["expected_episodes"] == 2250
    assert summary["coverage"]["completed_episodes"] == 1
    assert summary["coverage"]["complete"] is False
    assert all(row["expected_episodes"] == 450 for row in summary["by_k"])


def test_report_preserves_long_defaults_for_existing_metadata(schema2_report, tmp_path):
    report, _ = schema2_report
    summary = report.build_report(tmp_path)
    protocol = summary["protocol"]
    assert protocol["suite"] == "libero_10"
    assert protocol["suite_task_count"] == 10
    assert protocol["suite_max_steps"] == protocol["max_steps"] == 520
    assert protocol["shortened_episode_limit"] is False


def test_libero90_report_marks_selected_tasks_and_shortened_limit(schema2_report, tmp_path):
    report, _ = schema2_report
    metadata = {
        "schema_version": 2,
        "suite": "libero_90",
        "horizon": 40,
        "ks": [1, 40],
        "task_ids": [80, 89],
        "n_episodes": 5,
        "max_steps": 45,
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    summary = report.build_report(tmp_path)
    protocol = summary["protocol"]
    assert protocol["selected_tasks"] == 2
    assert protocol["suite_task_count"] == 90
    assert protocol["suite_max_steps"] == 400
    assert protocol["full_task_suite"] is False
    assert protocol["shortened_episode_limit"] is True
    assert summary["coverage"]["expected_episodes"] == 20
