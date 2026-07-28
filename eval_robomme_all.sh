#!/usr/bin/env bash
# Evaluate a policy on the full RoboMME benchmark (16 tasks x 50 test episodes = 800 episodes).
#
# Runs on the host; launches the eval inside the `lerobot-benchmark-robomme` image, one
# `lerobot-eval` invocation per task so a single task failure can't discard the whole run.
# Results are written to the host, and completed tasks are skipped on re-run (resumable).
#
# Usage:
#   ./eval_robomme_all.sh                          # all 16 tasks, 50 episodes each
#   TASKS="PickXtimes BinFill" ./eval_robomme_all.sh
#   BATCH_SIZE=10 ./eval_robomme_all.sh
#   N_EPISODES=10 ./eval_robomme_all.sh            # quick smoke run (first 10 episodes/task)

set -uo pipefail

POLICY="${POLICY:-Xihe666/drifting_robomme}"
IMAGE="${IMAGE:-lerobot-benchmark-robomme}"
DOCKER_CONTEXT_NAME="${DOCKER_CONTEXT_NAME:-default}"
SPLIT="${SPLIT:-test}"
N_EPISODES="${N_EPISODES:-50}"   # distinct episodes evaluated per task
BATCH_SIZE="${BATCH_SIZE:-10}"   # envs stepped in parallel per rollout batch
EPISODE_LENGTH="${EPISODE_LENGTH:-300}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/robomme_eval/$(date +%Y%m%d_%H%M%S)}"

# Fixed 16-task benchmark list (BenchmarkEnvBuilder.get_task_list() order).
DEFAULT_TASKS="PickXtimes StopCube SwingXtimes BinFill \
VideoUnmaskSwap VideoUnmask ButtonUnmaskSwap ButtonUnmask \
VideoRepick VideoPlaceButton VideoPlaceOrder PickHighlight \
InsertPeg MoveCube PatternLock RouteStick"
TASKS="${TASKS:-$DEFAULT_TASKS}"

docker_() { docker --context "$DOCKER_CONTEXT_NAME" "$@"; }

# `task_ids` are *episode indices*: the vec env for task_id t covers episodes t..t+BATCH_SIZE-1,
# and n_batches = ceil(n_episodes / batch_size). Passing n_episodes == BATCH_SIZE therefore runs
# exactly one batch per id, so tiling the ids in BATCH_SIZE-sized strides visits every episode
# exactly once with no repeats and no gaps.
if (( N_EPISODES % BATCH_SIZE != 0 )); then
    echo "ERROR: N_EPISODES ($N_EPISODES) must be divisible by BATCH_SIZE ($BATCH_SIZE) so that" >&2
    echo "       episode indices tile exactly; otherwise episodes are evaluated twice." >&2
    exit 1
fi
TASK_IDS="$(seq -s, 0 "$BATCH_SIZE" $((N_EPISODES - 1)))"

mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/results"

echo "=========================================================="
echo " RoboMME full benchmark eval"
echo "=========================================================="
echo "  policy        : $POLICY"
echo "  image         : $IMAGE"
echo "  split         : $SPLIT"
echo "  tasks         : $(wc -w <<<"$TASKS")"
echo "  episodes/task : $N_EPISODES  (indices 0..$((N_EPISODES - 1)))"
echo "  batch size    : $BATCH_SIZE  -> task_ids=[$TASK_IDS]"
echo "  output        : $OUT_ROOT"
echo "=========================================================="

CONTAINER="robomme_eval_$$"
cleanup() { docker_ rm -f "$CONTAINER" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

# Mount the working tree's source over the image's baked-in copy so the `task_description` fix
# is used, and mount the HF cache so the policy/dataset are not re-downloaded per task.
docker_ run -d --name "$CONTAINER" --gpus all --user root \
    -v "$HOME/.cache/huggingface:/home/user_lerobot/.cache/huggingface" \
    -v "$REPO_ROOT/src/lerobot:/lerobot/src/lerobot:ro" \
    -v "$OUT_ROOT:/eval_out" \
    "$IMAGE" sleep infinity >/dev/null || { echo "ERROR: failed to start container" >&2; exit 1; }

# The published image still ships scipy 1.18, which cannot be imported under the numpy 1.26 pin
# required by mani-skill (it dereferences the numpy-2-only `np.long`). Repair it in-place so this
# script also works against images built before that Dockerfile fix.
echo "[setup] checking scipy/numpy compatibility ..."
if ! docker_ exec "$CONTAINER" bash -lc 'python -c "import scipy.sparse" 2>/dev/null'; then
    echo "[setup] incompatible scipy detected; installing scipy<1.18 ..."
    docker_ exec "$CONTAINER" bash -lc 'cd /lerobot && uv pip install --no-cache -q "scipy<1.18"' \
        || { echo "ERROR: scipy repair failed" >&2; exit 1; }
fi
docker_ exec "$CONTAINER" bash -lc \
    'python -c "import numpy, scipy, gymnasium, mani_skill, robomme; print(f\"[setup] numpy={numpy.__version__} scipy={scipy.__version__} gymnasium={gymnasium.__version__} OK\")"' \
    || { echo "ERROR: environment check failed" >&2; exit 1; }

n_ok=0; n_fail=0; failed_tasks=""
for task in $TASKS; do
    result_json="$OUT_ROOT/results/${task}.json"
    if [[ -s "$result_json" ]]; then
        echo "[skip] $task (already has results)"
        n_ok=$((n_ok + 1))
        continue
    fi

    echo "----------------------------------------------------------"
    echo "[run ] $task  ($N_EPISODES episodes)"
    start_ts=$(date +%s)

    # `lerobot-eval` writes eval_info.json under its own timestamped dir; point it at a
    # deterministic per-task dir so results can be collected reliably.
    docker_ exec "$CONTAINER" bash -lc "cd /lerobot && lerobot-eval \
        --policy.path='$POLICY' \
        --env.type=robomme \
        --env.task='$task' \
        --env.dataset_split='$SPLIT' \
        --env.task_ids='[$TASK_IDS]' \
        --env.episode_length='$EPISODE_LENGTH' \
        --eval.batch_size='$BATCH_SIZE' \
        --eval.n_episodes='$BATCH_SIZE' \
        --output_dir='/eval_out/runs/$task'" \
        >"$OUT_ROOT/logs/${task}.log" 2>&1
    status=$?
    elapsed=$(( $(date +%s) - start_ts ))

    if [[ $status -ne 0 ]]; then
        echo "[FAIL] $task (exit $status, ${elapsed}s) -- see logs/${task}.log"
        tail -n 15 "$OUT_ROOT/logs/${task}.log" | sed 's/^/       | /'
        n_fail=$((n_fail + 1)); failed_tasks="$failed_tasks $task"
        continue
    fi

    if [[ -f "$OUT_ROOT/runs/$task/eval_info.json" ]]; then
        cp "$OUT_ROOT/runs/$task/eval_info.json" "$result_json"
        pc=$(python3 -c "import json;print(json.load(open('$result_json'))['overall']['pc_success'])" 2>/dev/null || echo "?")
        echo "[ok  ] $task  success=${pc}%  (${elapsed}s)"
        n_ok=$((n_ok + 1))
    else
        echo "[FAIL] $task -- eval_info.json not produced"
        n_fail=$((n_fail + 1)); failed_tasks="$failed_tasks $task"
    fi
done

echo "=========================================================="
echo " Completed: $n_ok ok, $n_fail failed"
[[ -n "$failed_tasks" ]] && echo " Failed tasks:$failed_tasks"

# The container runs as root, so everything under $OUT_ROOT lands root-owned on the host.
# Hand it back to the invoking user so results are readable/deletable without sudo.
docker_ exec "$CONTAINER" bash -lc "chown -R $(id -u):$(id -g) /eval_out" >/dev/null 2>&1 || true

# Aggregate every per-task result into one summary (JSON + human-readable table).
python3 - "$OUT_ROOT" <<'PYEOF'
import json, pathlib, sys

out = pathlib.Path(sys.argv[1])
rows, total_ep, total_succ = [], 0, 0.0
for f in sorted((out / "results").glob("*.json")):
    try:
        overall = json.loads(f.read_text())["overall"]
    except Exception as exc:
        print(f"  ! could not parse {f.name}: {exc}")
        continue
    n, pc = overall["n_episodes"], overall["pc_success"]
    rows.append({"task": f.stem, "n_episodes": n, "pc_success": pc,
                 "avg_sum_reward": overall.get("avg_sum_reward")})
    total_ep += n
    total_succ += pc / 100.0 * n          # weight by episode count, not by task

if not rows:
    print("No results to summarize.")
    sys.exit(0)

overall_pc = total_succ / total_ep * 100.0
summary = {"per_task": rows, "n_tasks": len(rows), "total_episodes": total_ep,
           "overall_pc_success": overall_pc}
(out / "summary.json").write_text(json.dumps(summary, indent=2))

lines = ["", f"{'task':<20}{'episodes':>10}{'success %':>12}",
         "-" * 42]
lines += [f"{r['task']:<20}{r['n_episodes']:>10}{r['pc_success']:>12.1f}" for r in rows]
lines += ["-" * 42,
          f"{'OVERALL':<20}{total_ep:>10}{overall_pc:>12.1f}", ""]
table = "\n".join(lines)
(out / "summary.txt").write_text(table)
print(table)
print(f"Summary written to {out/'summary.json'} and {out/'summary.txt'}")
PYEOF

echo "Results: $OUT_ROOT"
[[ -n "$failed_tasks" ]] && exit 1
exit 0
