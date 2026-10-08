#!/usr/bin/env python
"""Fixed-H, variable-K synchronous feedback experiment on LIBERO Long or LIBERO-90.

conda run --no-capture-output -n lerobot python scripts/latency_test/benchmark_libero_feedback.py \
    --n-episodes 10 --output-dir outputs/libero_feedback

H is always 40. K controls only how many decoded actions are executed before
replanning. See README.md for the clock, deadline, and simulator semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

HORIZON = 40
CHECKPOINT = "Xihe666/drif_ov_libero_20k_seed42_0821"
RENAME_MAP = {"observation.images.image2": "observation.images.wrist_image"}
SUITES = {"libero_10": (10, 520), "libero_90": (90, 400)}


def run_episode(
    env,
    planner,
    observation,
    *,
    k: int,
    control_hz: float,
    max_steps: int = 520,
    deadline_tolerance_s: float = 0.001,
    clock=None,
) -> dict[str, Any]:
    """Execute one already-reset episode; the injectable clock enables exact tests.

    The first action is due one period after observation delivery. Subsequent
    actions are due one period after the previous actual dispatch. Overruns
    rebase the schedule; no catch-up bursts or skipped policy actions occur.
    Simulation advances only inside env.step and pauses during inference stalls.
    """
    if not 1 <= k <= HORIZON:
        raise ValueError("K must be in [1, 40].")
    if not math.isfinite(control_hz) or control_hz <= 0 or max_steps <= 0:
        raise ValueError("control_hz and max_steps must be positive and finite.")
    if not math.isfinite(deadline_tolerance_s) or deadline_tolerance_s < 0:
        raise ValueError("deadline_tolerance_s must be finite and non-negative.")
    clock = time if clock is None else clock
    period = 1 / control_hz
    started = clock.perf_counter()
    observed_at = started
    deadline = started + period
    previous_dispatch = None
    previous_action = None
    previous_delta = None
    actions, plans = [], []
    success = False
    terminated = truncated = False

    while len(actions) < max_steps and not (terminated or truncated or success):
        source_observed_at = observed_at
        plan_started = clock.perf_counter()
        chunk = np.asarray(planner(observation), dtype=np.float64)
        if chunk.shape != (HORIZON, 7) or not np.isfinite(chunk).all():
            raise ValueError(f"Expected a finite full H=40 action chunk (40, 7), got {chunk.shape}.")
        plan_ready = clock.perf_counter()
        plan_id = len(plans)
        plan = {
            "plan_id": plan_id,
            "step": len(actions),
            "observation_at_s": source_observed_at - started,
            "ready_at_s": plan_ready - started,
            "pipeline_ms": (plan_ready - plan_started) * 1000,
            "e2e_first_action_ms": None,
        }
        plans.append(plan)
        for chunk_index in range(min(k, max_steps - len(actions))):
            action = chunk[chunk_index].copy()
            ready_at = clock.perf_counter()
            remaining = deadline - ready_at
            if remaining > 0:
                clock.sleep(remaining)
            dispatched_at = clock.perf_counter()
            # Timestamp immediately before the actual environment call. Disk
            # writes, rendering videos, and printing are outside this loop.
            observation, _, terminated, truncated, info = env.step(action)
            observed_at = clock.perf_counter()
            success = bool(info.get("is_success", False))
            delta = None if previous_action is None else action[:6] - previous_action[:6]
            boundary = chunk_index == 0 and len(actions) > 0
            age_ms = (dispatched_at - source_observed_at) * 1000
            if chunk_index == 0:
                plan["e2e_first_action_ms"] = age_ms
            actions.append(
                {
                    "step": len(actions),
                    "plan_id": plan_id,
                    "chunk_index": chunk_index,
                    "observation_at_s": source_observed_at - started,
                    "ready_at_s": ready_at - started,
                    "deadline_at_s": deadline - started,
                    "dispatch_at_s": dispatched_at - started,
                    "observation_age_ms": age_ms,
                    "observation_age_steps": chunk_index,
                    "deadline_missed": dispatched_at > deadline + deadline_tolerance_s,
                    "ready_deadline_missed": ready_at > deadline + deadline_tolerance_s,
                    "dispatch_lateness_ms": max(0.0, dispatched_at - deadline) * 1000,
                    "env_step_ms": (observed_at - dispatched_at) * 1000,
                    "is_boundary": boundary,
                    "action_delta_l2": None if delta is None else float(np.linalg.norm(delta)),
                    "action_accel_l2": (
                        None
                        if delta is None or previous_delta is None
                        else float(np.linalg.norm(delta - previous_delta))
                    ),
                    "translation_delta_l2": None if delta is None else float(np.linalg.norm(delta[:3])),
                    "rotation_delta_l2": None if delta is None else float(np.linalg.norm(delta[3:])),
                    "gripper_changed": (
                        None if previous_action is None else bool((action[6] > 0) != (previous_action[6] > 0))
                    ),
                    "boundary_interval_error_ms": (
                        None
                        if previous_dispatch is None
                        else ((dispatched_at - previous_dispatch) - period) * 1000
                    ),
                    "action": action.tolist(),
                }
            )
            previous_action, previous_delta = action, delta
            previous_dispatch = dispatched_at
            deadline = dispatched_at + period
            if terminated or truncated or success:
                break

    return {
        "success": success,
        "steps": len(actions),
        "wall_time_s": observed_at - started,
        "sim_time_s": len(actions) / control_hz,
        "control_hz": control_hz,
        "termination": "success" if success else "terminated" if terminated else "step_limit",
        "plans": plans,
        "actions": actions,
    }


def _batch_observation(value):
    if isinstance(value, dict):
        return {key: _batch_observation(item) for key, item in value.items()}
    return np.expand_dims(value, 0) if isinstance(value, np.ndarray) else value


def observation_fingerprint(observation) -> str:
    """Hash raw images/state, independently of dictionary order and array layout."""
    digest = hashlib.sha256()

    def visit(value):
        if isinstance(value, dict):
            digest.update(b"dict\0")
            for key in sorted(value):
                digest.update(json.dumps(key).encode() + b"\0")
                visit(value[key])
            digest.update(b"end\0")
        elif isinstance(value, np.ndarray):
            digest.update(json.dumps([value.dtype.str, value.shape]).encode() + b"\0")
            digest.update(np.ascontiguousarray(value).tobytes())
        else:
            digest.update(json.dumps(value, allow_nan=False).encode() + b"\0")

    visit(observation)
    return digest.hexdigest()


class ChunkPlanner:
    def __init__(self, policy, preprocessor, postprocessor, env_preprocessor, env_postprocessor):
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.env_preprocessor = env_preprocessor
        self.env_postprocessor = env_postprocessor
        self.task_description = ""

    def reset(self):
        for component in (
            self.policy,
            self.preprocessor,
            self.postprocessor,
            self.env_preprocessor,
            self.env_postprocessor,
        ):
            component.reset()

    def __call__(self, raw_observation):
        import torch

        from lerobot.envs import preprocess_observation

        device = torch.device(self.policy.config.device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        with torch.inference_mode():
            observation = preprocess_observation(_batch_observation(raw_observation))
            observation["task"] = [self.task_description]
            observation = self.env_preprocessor(observation)
            batch = self.preprocessor(observation)
            # Deliberately no prefix conditioning, overlap, action blending, or
            # select_action queue. Decode against this observation's cached state.
            predicted = self.policy.predict_action_chunk(batch)
            if predicted.ndim != 3 or predicted.shape[:2] != (1, HORIZON):
                raise ValueError(f"Checkpoint returned {tuple(predicted.shape)}; H must remain 40.")
            decoded = self.postprocessor(predicted)
            decoded = self.env_postprocessor({"action": decoded})["action"]
            actions = decoded.detach().to(device="cpu", dtype=torch.float32).numpy()[0].copy()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return actions


def _make_environment(suite, task_id: int, max_steps: int, image_size: int, *, suite_name="libero_10"):
    from lerobot.envs.libero import LiberoEnv

    class TimedLiberoEnv(LiberoEnv):
        """Preserve LeRobot observations and success checks without terminal reset."""

        def step(self, action):
            self._ensure_env()
            raw, reward, done, info = self._env.step(action)
            success = bool(self._env.check_success())
            info.update(is_success=success, task=self.task, task_id=self.task_id, done=done)
            return self._format_raw_obs(raw), reward, bool(done or success), False, info

    return TimedLiberoEnv(
        task_suite=suite,
        task_id=task_id,
        task_suite_name=suite_name,
        episode_length=max_steps,
        obs_type="pixels_agent_pos",
        observation_width=image_size,
        observation_height=image_size,
        init_states=True,
        control_mode="relative",
    )


def _atomic_json(path: Path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _episode_path(output_dir: Path, task: int, episode: int, k: int) -> Path:
    return output_dir / "episodes" / f"task_{task:02d}_episode_{episode:03d}_k_{k:02d}.json"


def _manifest(args) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "reset_protocol": "fresh_env_per_episode_with_paired_observation_sha256",
        "checkpoint": args.checkpoint,
        "revision": args.revision,
        "suite": args.suite,
        "suite_task_count": SUITES[args.suite][0],
        "suite_max_steps": SUITES[args.suite][1],
        "horizon": HORIZON,
        "ks": sorted(args.ks),
        "n_episodes": args.n_episodes,
        "task_ids": sorted(args.task_ids),
        "seed": args.seed,
        "max_steps": args.max_steps,
        "mode": "synchronous_paced",
        "control_hz": args.control_hz,
        "deadline_tolerance_ms": args.deadline_tolerance_ms,
        "warmup": args.warmup,
        "image_size": args.image_size,
        "device": args.device,
        "rename_map": RENAME_MAP,
        "libero_assets": args.libero_assets,
        "timing": {
            "observation_timestamp": "raw observation delivered after reset/step, before preprocessing",
            "pipeline": "observation preprocessing + full H prediction + decoding + GPU sync + CPU copy",
            "end_to_end": "source observation delivered to first action dispatch, including pacing",
            "deadline": "previous actual dispatch + 1/control_hz; first deadline = episode start + period",
            "miss": "actual dispatch > deadline + tolerance (includes sleep overshoot)",
            "simulation": "physics pauses during planning and wall-clock sleeps; no asynchronous execution",
            "wall_time": "initial observation delivered through final step return; reset/warmup excluded",
        },
    }


def _prepare_libero_resources(args, output_dir: Path):
    """Avoid LIBERO's first-import prompt and its obsolete HF download argument."""
    import yaml
    from huggingface_hub import snapshot_download

    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("LIBERO is not installed in this Python environment.")
    root = Path(next(iter(spec.submodule_search_locations))) / "libero"
    native_assets = root / "assets"
    assets = (
        Path(args.libero_assets).expanduser().resolve()
        if args.libero_assets
        else native_assets
        if native_assets.is_dir()
        else Path.home() / ".cache/libero/assets"
    )
    config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", Path.home() / ".libero")).expanduser()
    if not (config_dir / "config.yaml").exists():
        # A local, noninteractive configuration leaves an existing user config alone.
        if "LIBERO_CONFIG_PATH" not in os.environ:
            config_dir = output_dir.resolve() / "libero_config"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "benchmark_root": str(root),
                    "bddl_files": str(root / "bddl_files"),
                    "init_states": str(root / "init_files"),
                    "datasets": str(root.parent / "datasets"),
                    "assets": str(assets),
                }
            )
        )
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir.resolve())
    expected = (
        "articulated_objects",
        "stable_scanned_objects",
        "turbosquid_objects",
        "stable_hope_objects",
        "scenes",
    )
    if not all((assets / name).is_dir() for name in expected):
        if args.libero_assets:
            raise FileNotFoundError(f"Incomplete --libero-assets directory: {assets}; expected {expected}.")
        print(f"[feedback] Downloading LIBERO simulation assets to {assets}", flush=True)
        snapshot_download(
            "lerobot/libero-assets",
            repo_type="dataset",
            local_dir=assets,
            local_files_only=args.local_files_only,
        )
    if not all((assets / name).is_dir() for name in expected):
        raise FileNotFoundError(f"LIBERO assets are incomplete: {assets}")
    import libero.libero as libero

    # This installed LIBERO version ignores config.yaml['assets'] in
    # get_assets_path; its cache is the supported function's first lookup.
    if hasattr(libero, "_assets_path_cache"):
        libero._assets_path_cache = str(assets)
    _atomic_json(output_dir / "libero_resources.json", {"config_dir": str(config_dir), "assets": str(assets)})


def _load_planner(args, output_dir):
    import torch
    from huggingface_hub import snapshot_download

    from lerobot.configs import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv as LiberoConfig
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable in this process. Verify nvidia-smi and the conda environment."
        )
    source = Path(args.checkpoint).expanduser()
    runtime_path = output_dir / "runtime.json"
    previous = json.loads(runtime_path.read_text()) if runtime_path.exists() else None
    resolved_revision = previous.get("resolved_revision") if previous else args.revision
    local_source = source.is_dir()
    checkpoint_path = (
        str(source.resolve())
        if local_source
        else snapshot_download(
            args.checkpoint, revision=resolved_revision, local_files_only=args.local_files_only
        )
    )
    config = PreTrainedConfig.from_pretrained(checkpoint_path)
    if config.type != "drif_ov" or config.chunk_size != HORIZON:
        raise ValueError(
            f"Expected a drif_ov checkpoint trained with H=40, got {config.type=} {config.chunk_size=}."
        )
    config.n_action_steps = HORIZON  # prediction_horizon also reads this; K is external.
    config.device = args.device
    config.pretrained_path = Path(checkpoint_path)
    config.pretrained_revision = None
    env_config = LiberoConfig(
        task=args.suite, observation_width=args.image_size, observation_height=args.image_size
    )
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    policy = make_policy(cfg=config, env_cfg=env_config, rename_map=RENAME_MAP).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=checkpoint_path,
        preprocessor_overrides={
            "device_processor": {"device": args.device},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_config, policy_cfg=config)
    revision = None if local_source else Path(checkpoint_path).name
    git_result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    runtime = {
        "checkpoint_path": checkpoint_path,
        "resolved_revision": revision,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "git_commit": git_result.stdout.strip(),
        "policy_config": json.loads(config.to_json()) if hasattr(config, "to_json") else str(config),
    }
    if previous and (previous["gpu"], previous["torch"]) != (runtime["gpu"], runtime["torch"]):
        raise ValueError(
            "GPU/PyTorch changed since this run; use a new output directory for comparable timings."
        )
    _atomic_json(runtime_path, runtime)
    return ChunkPlanner(policy, preprocessor, postprocessor, env_pre, env_post)


def _run(args, metadata):
    from lerobot.utils.random_utils import set_seed

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        if not args.resume:
            raise ValueError(
                "Output directory already contains an experiment; use --resume or a new directory."
            )
        stored = json.loads(metadata_path.read_text())
        stored_suite = stored.get("suite", "libero_10")
        stored.setdefault("suite_task_count", SUITES[stored_suite][0])
        stored.setdefault("suite_max_steps", SUITES[stored_suite][1])
        if stored != metadata:
            raise ValueError("Resume arguments differ from metadata.json. Use the original arguments.")
    else:
        _atomic_json(metadata_path, metadata)
    (output_dir / "episodes").mkdir(exist_ok=True)
    from libero_feedback_report import build_report

    try:
        if all(
            _episode_path(output_dir, task_id, episode, k).exists()
            for task_id in metadata["task_ids"]
            for episode in range(args.n_episodes)
            for k in metadata["ks"]
        ):
            print("[feedback] All requested episodes already exist; regenerating the report.", flush=True)
            return
        set_seed(args.seed)
        _prepare_libero_resources(args, output_dir)
        planner = _load_planner(args, output_dir)
        from lerobot.envs.libero import _get_suite

        suite = _get_suite(args.suite)
        if len(suite.tasks) != metadata["suite_task_count"]:
            raise ValueError("Installed suite size differs from the experiment specification.")
        for task_id in metadata["task_ids"]:
            pending = [
                (episode, k)
                for episode in range(args.n_episodes)
                for k in metadata["ks"]
                if not _episode_path(output_dir, task_id, episode, k).exists()
            ]
            if not pending:
                continue
            env = _make_environment(suite, task_id, args.max_steps, args.image_size, suite_name=args.suite)
            try:
                planner.task_description = env.task_description
                env.init_state_id = 0
                warm_observation, _ = env.reset(seed=args.seed)
                native_hz = float(env._env.env.control_freq)
                control_hz = native_hz if args.control_hz is None else args.control_hz
                if not math.isclose(control_hz, native_hz):
                    raise ValueError(
                        f"Requested {control_hz} Hz differs from LIBERO control_freq={native_hz}; "
                        "this benchmark requires matching wall-clock and simulator periods."
                    )
                planner.reset()
                for _ in range(args.warmup):
                    planner(warm_observation)
            finally:
                env.close()

            for episode in range(args.n_episodes):
                seed = args.seed + task_id * args.n_episodes + episode
                order = list(metadata["ks"])
                random.Random(seed).shuffle(order)
                hashes = {
                    json.loads(path.read_text())["initial_observation_sha256"]
                    for k in order
                    if (path := _episode_path(output_dir, task_id, episode, k)).exists()
                }
                if len(hashes) > 1:
                    raise ValueError(
                        f"Stored paired initial observations differ for task={task_id}, episode={episode}."
                    )
                reference_hash = next(iter(hashes), None)
                for k in order:
                    path = _episode_path(output_dir, task_id, episode, k)
                    if path.exists():
                        continue
                    # Repeated hard resets in the installed LIBERO accumulate
                    # Open/Close/TurnOn samplers, altering fixture placement even
                    # at identical seeds/state IDs. A fresh instance per rollout
                    # prevents this leak. Creation is outside the measured loop.
                    set_seed(seed)
                    env = _make_environment(
                        suite, task_id, args.max_steps, args.image_size, suite_name=args.suite
                    )
                    try:
                        env.init_state_id = episode
                        observation, _ = env.reset(seed=seed)
                        initial_hash = observation_fingerprint(observation)
                        if reference_hash is not None and initial_hash != reference_hash:
                            raise RuntimeError(
                                f"Initial observation differs across K for task={task_id}, episode={episode}, "
                                f"K={k}; refusing an unpaired comparison."
                            )
                        reference_hash = initial_hash
                        planner.reset()
                        set_seed(seed)  # warmup/env random draws cannot alter policy's initial RNG.
                        result = run_episode(
                            env,
                            planner,
                            observation,
                            k=k,
                            control_hz=control_hz,
                            max_steps=args.max_steps,
                            deadline_tolerance_s=args.deadline_tolerance_ms / 1000,
                        )
                        result.update(
                            k=k,
                            task_id=task_id,
                            task_name=env.task,
                            episode_index=episode,
                            seed=seed,
                            init_state_id=episode % len(env._init_states),
                            initial_observation_sha256=initial_hash,
                        )
                        _atomic_json(path, result)
                        print(
                            f"[feedback] task={task_id} episode={episode} K={k} "
                            f"success={result['success']} steps={result['steps']} "
                            f"wall={result['wall_time_s']:.2f}s",
                            flush=True,
                        )
                    finally:
                        env.close()
            build_report(output_dir)
    finally:
        # Interrupted/incomplete runs remain useful, clearly marked PARTIAL.
        build_report(output_dir)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--suite", "--task", choices=tuple(SUITES), default="libero_10")
    parser.add_argument("--revision", help="HF commit/tag; resolved commit is pinned for resume.")
    parser.add_argument("--ks", "--k-values", nargs="+", type=int, default=[1, 5, 10, 20, 40])
    parser.add_argument("--n-episodes", type=int, default=10, help="Episodes per task per K.")
    parser.add_argument("--task-ids", nargs="+", type=int, help="Default: all tasks in the selected suite.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-steps", type=int, help="Default: standard suite limit (Long=520, LIBERO-90=400)."
    )
    parser.add_argument("--warmup", type=int, default=3, help="Untimed full predictions per task.")
    parser.add_argument(
        "--control-hz", type=float, help="Default: read actual LIBERO control_freq (normally 20 Hz)."
    )
    parser.add_argument("--deadline-tolerance-ms", type=float, default=1.0)
    parser.add_argument(
        "--image-size", type=int, default=360, help="Match the existing lerobot-eval camera size."
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="outputs/libero_feedback")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--libero-assets", help="Local LIBERO assets directory; otherwise use installed/default cache."
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print experiment specification without loading anything."
    )
    args = parser.parse_args()
    if args.plot_only:
        from libero_feedback_report import build_report

        build_report(Path(args.output_dir))
        return
    suite_size, suite_max_steps = SUITES[args.suite]
    if args.task_ids is None:
        args.task_ids = list(range(suite_size))
    if args.max_steps is None:
        args.max_steps = suite_max_steps
    if (
        any(k < 1 or k > HORIZON for k in args.ks)
        or len(set(args.ks)) != len(args.ks)
        or any(task < 0 or task >= suite_size for task in args.task_ids)
        or len(set(args.task_ids)) != len(args.task_ids)
        or args.n_episodes < 1
        or not 1 <= args.max_steps <= suite_max_steps
        or args.warmup < 0
        or args.image_size <= 0
        or args.seed < 0
        or not math.isfinite(args.deadline_tolerance_ms)
        or args.deadline_tolerance_ms < 0
        or (args.control_hz is not None and (not math.isfinite(args.control_hz) or args.control_hz <= 0))
    ):
        parser.error(
            f"Invalid K/task IDs, counts, timing, image size, or seed; {args.suite} has task IDs "
            f"0..{suite_size - 1} and max_steps must be 1..{suite_max_steps}."
        )
    metadata = _manifest(args)
    print(json.dumps(metadata, indent=2), flush=True)
    if args.dry_run:
        return
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
    if os.environ.get("LEROBOT_PROFILE_INFERENCE_TIMINGS"):
        parser.error(
            "Unset LEROBOT_PROFILE_INFERENCE_TIMINGS: internal CSV profiling would contaminate this experiment."
        )
    _run(args, metadata)


if __name__ == "__main__":
    main()
