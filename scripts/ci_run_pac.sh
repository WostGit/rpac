#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-Residual-PAC-Privacy}"
OUT_DIR="${2:-$PWD/pac_ci_out}"
mkdir -p "$OUT_DIR"

if [[ ! -d "$REPO_DIR" ]]; then
  echo "Repository directory '$REPO_DIR' not found" >&2
  exit 2
fi

pushd "$REPO_DIR" >/dev/null

# Install dependencies exactly as documented in README when possible.
# Prefer explicit commands from README; otherwise, fail loudly.
if [[ ! -f README.md ]]; then
  echo "README.md missing in repository root" >&2
  exit 3
fi

python -m pip install --upgrade pip

if rg -q "pip install -r requirements\\.txt" README.md && [[ -f requirements.txt ]]; then
  echo "Installing dependencies via requirements.txt (as documented in README.md)"
  python -m pip install -r requirements.txt
else
  echo "Could not find a documented dependency installation command in README.md" >&2
  echo "Expected to find: 'pip install -r requirements.txt'" >&2
  exit 4
fi

choose_script() {
  if [[ -f Iris_s.py ]]; then
    echo "Iris_s.py"
  elif [[ -f MNIST_s.py ]]; then
    echo "MNIST_s.py"
  else
    return 1
  fi
}

SCRIPT="$(choose_script)" || {
  echo "Neither Iris_s.py nor MNIST_s.py exists; cannot run low-resource PAC pipeline" >&2
  exit 5
}

echo "Selected experiment script: $SCRIPT"

# Build conservative minimal-resource CLI arguments based on available flags.
HELP_FILE="$OUT_DIR/help.txt"
python "$SCRIPT" --help > "$HELP_FILE" 2>&1 || true

ARGS=()
add_arg_if_supported() {
  local pattern="$1"
  shift
  if rg -q -- "$pattern" "$HELP_FILE"; then
    ARGS+=("$@")
  fi
}

# Strictly minimal resources while preserving end-to-end PAC pipeline.
add_arg_if_supported "--dataset-shard" --dataset-shard 1
add_arg_if_supported "--shard" --shard 1
add_arg_if_supported "--subset" --subset 32
add_arg_if_supported "--sample-size" --sample-size 32
add_arg_if_supported "--batch-size" --batch-size 4
add_arg_if_supported "--epochs" --epochs 1
add_arg_if_supported "--repetitions" --repetitions 1
add_arg_if_supported "--trials" --trials 1
add_arg_if_supported "--runs" --runs 1
add_arg_if_supported "--stochastic-reps" --stochastic-reps 1
add_arg_if_supported "--num-workers" --num-workers 0
add_arg_if_supported "--hidden-dim" --hidden-dim 4
add_arg_if_supported "--hidden-size" --hidden-size 4
add_arg_if_supported "--width" --width 4
add_arg_if_supported "--depth" --depth 1
add_arg_if_supported "--channels" --channels 4
add_arg_if_supported "--model-size" --model-size tiny
add_arg_if_supported "--device" --device cpu
add_arg_if_supported "--no-cuda" --no-cuda

LOG_FILE="$OUT_DIR/pac_run.log"
TIME_FILE="$OUT_DIR/time.txt"
METRICS_FILE="$OUT_DIR/pac_metrics.txt"

set +e
/usr/bin/timeout --preserve-status 20m /usr/bin/time -v \
  python "$SCRIPT" "${ARGS[@]}" \
  >"$LOG_FILE" 2>"$TIME_FILE"
RC=$?
set -e

if [[ $RC -eq 124 ]]; then
  echo "Run exceeded CI time limit (20 minutes)" >&2
  exit 6
fi
if [[ $RC -ne 0 ]]; then
  echo "PAC pipeline execution failed with exit code $RC" >&2
  tail -n 80 "$LOG_FILE" >&2 || true
  tail -n 80 "$TIME_FILE" >&2 || true
  exit 7
fi

# Ensure pipeline includes key PAC components and is numerically stable.
if ! rg -Eiq "residual|leakage|bound" "$LOG_FILE"; then
  echo "Output does not indicate complete PAC pipeline (residual/leakage/bound missing)" >&2
  exit 8
fi
if rg -Eiq "nan|inf|overflow|underflow|diverg|numerical" "$LOG_FILE"; then
  echo "Numerical instability detected in PAC output" >&2
  exit 9
fi

BOUND_LINE="$(rg -Eio '(privacy[^\n]*bound[^\n]*|bound[^\n]*privacy[^\n]*|bound[^\n]*=[^\n]*)' "$LOG_FILE" | tail -n1 || true)"
if [[ -z "$BOUND_LINE" ]]; then
  BOUND_LINE="$(rg -Eio 'bound[^\n]*' "$LOG_FILE" | tail -n1 || true)"
fi
if [[ -z "$BOUND_LINE" ]]; then
  echo "PAC-style privacy bound was not found in output" >&2
  exit 10
fi

RUNTIME_SECONDS="$(awk -F': ' '/Elapsed \(wall clock\) time/{print $2}' "$TIME_FILE" | tail -n1)"
PEAK_KB="$(awk -F': ' '/Maximum resident set size/{print $2}' "$TIME_FILE" | tail -n1)"
if [[ -z "$RUNTIME_SECONDS" || -z "$PEAK_KB" ]]; then
  echo "Failed to parse runtime and peak memory from /usr/bin/time output" >&2
  exit 11
fi

{
  echo "script=$SCRIPT"
  echo "args=${ARGS[*]}"
  echo "privacy_bound=$BOUND_LINE"
  echo "runtime_wall_clock=$RUNTIME_SECONDS"
  echo "peak_memory_kb=$PEAK_KB"
} > "$METRICS_FILE"

cp "$LOG_FILE" "$TIME_FILE" "$METRICS_FILE" "$OUT_DIR/" 2>/dev/null || true

popd >/dev/null
