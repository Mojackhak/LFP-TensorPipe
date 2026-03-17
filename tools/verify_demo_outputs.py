"""Verify demo outputs against expected metrics with tolerance."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _compare(
    expected, actual, path: str, rtol: float, atol: float, errors: list[str]
) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            errors.append(f"{path}: expected dict, got {type(actual).__name__}")
            return
        for key, expected_value in expected.items():
            if key not in actual:
                errors.append(f"{path}.{key}: missing key")
                continue
            _compare(expected_value, actual[key], f"{path}.{key}", rtol, atol, errors)
        return

    if isinstance(expected, list):
        if not isinstance(actual, list):
            errors.append(f"{path}: expected list, got {type(actual).__name__}")
            return
        if len(expected) != len(actual):
            errors.append(
                f"{path}: length mismatch (expected {len(expected)}, got {len(actual)})"
            )
            return
        for idx, (exp_item, act_item) in enumerate(zip(expected, actual)):
            _compare(exp_item, act_item, f"{path}[{idx}]", rtol, atol, errors)
        return

    if isinstance(expected, (int, float)):
        if not isinstance(actual, (int, float)):
            errors.append(f"{path}: expected number, got {type(actual).__name__}")
            return
        if not math.isclose(float(actual), float(expected), rel_tol=rtol, abs_tol=atol):
            errors.append(f"{path}: {actual} != {expected} (rtol={rtol}, atol={atol})")
        return

    if expected != actual:
        errors.append(f"{path}: {actual!r} != {expected!r}")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify demo outputs against expected metrics."
    )
    parser.add_argument("--out", required=True, help="Demo output directory.")
    parser.add_argument(
        "--expected",
        default="demo/expected/metrics.json",
        help="Path to expected metrics JSON.",
    )
    parser.add_argument("--rtol", type=float, default=1e-9, help="Relative tolerance.")
    parser.add_argument("--atol", type=float, default=1e-9, help="Absolute tolerance.")
    args = parser.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    expected_input = Path(args.expected).expanduser()
    if not expected_input.is_absolute():
        expected_input = REPO_ROOT / expected_input
    expected_path = expected_input.resolve()

    metrics_path = out_dir / "metrics.json"
    manifest_path = out_dir / "run_manifest.json"
    bandpower_csv_path = out_dir / "bandpower.csv"

    if not metrics_path.exists():
        raise SystemExit(f"Missing metrics.json at {metrics_path}")
    if not manifest_path.exists():
        raise SystemExit(f"Missing run_manifest.json at {manifest_path}")
    if not bandpower_csv_path.exists():
        raise SystemExit(f"Missing bandpower.csv at {bandpower_csv_path}")

    metrics = _load_json(metrics_path)
    expected = _load_json(expected_path)

    errors: list[str] = []
    _compare(expected, metrics, "metrics", args.rtol, args.atol, errors)

    manifest = _load_json(manifest_path)
    for key in ("core", "workflows", "demo"):
        if key not in manifest:
            errors.append(f"manifest missing key: {key}")

    if errors:
        print("Demo output verification failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Demo outputs verified successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
