#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np


def _set_if_present(mod, name, value):
    if hasattr(mod, name):
        setattr(mod, name, value)


def _install_iris_shard(mod, per_class: int):
    if not hasattr(mod, "load_iris"):
        return

    original = mod.load_iris

    def _tiny_load_iris(*args, **kwargs):
        bunch = original(*args, **kwargs)
        y = np.asarray(bunch.target)
        keep = []
        for c in np.unique(y):
            keep.extend(np.where(y == c)[0][:per_class])
        keep = np.asarray(keep)

        bunch.data = np.asarray(bunch.data)[keep]
        bunch.target = y[keep]
        if hasattr(bunch, "target_names"):
            bunch.target_names = np.asarray(bunch.target_names)
        if hasattr(bunch, "frame") and bunch.frame is not None:
            bunch.frame = bunch.frame.iloc[keep].reset_index(drop=True)
        return bunch

    mod.load_iris = _tiny_load_iris


def _finite_float(value: str) -> float:
    f = float(value)
    if not math.isfinite(f):
        raise ValueError(f"Non-finite numeric value: {value}")
    return f


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--script", default="Iris_s.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-per-class", type=int, default=5)
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    script_path = repo_dir / args.script
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    spec = importlib.util.spec_from_file_location("residual_pac_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    # Minimal-resource overrides while preserving full PAC pipeline.
    _set_if_present(module, "BETA_LIST", [0.10])
    _set_if_present(module, "ALT_EPOCHS", 2)
    _set_if_present(module, "FEW_EPOCHS", 1)
    _set_if_present(module, "N_HC_SAMPLES", 2)
    _set_if_present(module, "WARM_EPOCHS", 1)
    _set_if_present(module, "K_MEAS", 2)
    _set_if_present(module, "SMALL_SIGMAS", [0.0])

    if args.script.lower().startswith("iris"):
        _install_iris_shard(module, max(1, args.shard_per_class))

    try:
        module.main()
    except Exception as exc:  # noqa: BLE001
        print(f"[CI-ERROR] PAC pipeline execution failed: {exc}", file=sys.stderr)
        return 2

    out_csv_name = getattr(module, "OUT_CSV", None)
    csv_path = (repo_dir / out_csv_name) if out_csv_name else None
    if not csv_path or not csv_path.exists():
        matches = sorted(repo_dir.glob("*summary*.csv"))
        if not matches:
            print("[CI-ERROR] Could not find PAC summary CSV output.", file=sys.stderr)
            return 3
        csv_path = matches[-1]

    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        print("[CI-ERROR] PAC summary CSV is empty.", file=sys.stderr)
        return 4

    for row in rows:
        for key, val in row.items():
            if val is None or val == "":
                continue
            try:
                _finite_float(val)
            except Exception:
                continue

    baseline = next((r for r in rows if r.get("tag") == "baseline"), rows[0])
    if "Hc" not in baseline:
        print("[CI-ERROR] Hc/privacy-bound column missing from output.", file=sys.stderr)
        return 5

    privacy_bound = _finite_float(baseline["Hc"])
    target = _finite_float(baseline.get("target_Hc", baseline["Hc"]))
    gap = abs(privacy_bound - target)

    summary = {
        "script": args.script,
        "csv": str(csv_path),
        "rows": len(rows),
        "privacy_bound": privacy_bound,
        "target_bound": target,
        "bound_gap": gap,
        "selected_tag": baseline.get("tag", ""),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "summary.csv").write_text(csv_path.read_text())

    print(f"CI_PRIVACY_BOUND={privacy_bound:.8f}")
    print(f"CI_TARGET_BOUND={target:.8f}")
    print(f"CI_BOUND_GAP={gap:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
