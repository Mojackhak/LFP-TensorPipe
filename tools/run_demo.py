"""Run a lightweight, deterministic workflow demo.

The demo:
- imports the core library (lfptensorpipe) and its src module
- loads workflow band definitions from packaged tensor defaults
- computes analytic bandpower for synthetic sine components
- writes a manifest plus reproducible metrics artifacts
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

FREQUENCIES_HZ = [10.0, 20.0, 40.0]
CHANNEL_AMPLITUDES = {
    "ch0": {10.0: 1.0, 20.0: 0.5, 40.0: 0.2},
    "ch1": {10.0: 0.8, 20.0: 0.3, 40.0: 0.1},
    "ch2": {10.0: 1.2, 20.0: 0.6, 40.0: 0.0},
    "ch3": {10.0: 0.5, 20.0: 0.2, 40.0: 0.05},
}


def _round_obj(obj, decimals: int = 12):
    if isinstance(obj, float):
        return round(obj, decimals)
    if isinstance(obj, dict):
        return {k: _round_obj(v, decimals) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_obj(v, decimals) for v in obj]
    return obj


def _git_commit(repo_root: Path) -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return output.strip()
    except Exception:
        return "unknown"


def _load_bands() -> dict[str, tuple[float, float]]:
    try:
        from lfptensorpipe.app.shared.config_store import _default_template_payload
    except Exception as exc:
        raise RuntimeError(f"Failed to load config-store helper: {exc}")

    payload = _default_template_payload("tensor.yml")
    raw_bands = payload.get("bands_defaults", []) if isinstance(payload, dict) else []
    if not isinstance(raw_bands, list) or not raw_bands:
        raise RuntimeError("tensor.yml does not define non-empty `bands_defaults`.")

    bands: dict[str, tuple[float, float]] = {}
    for item in raw_bands:
        if not isinstance(item, dict):
            raise RuntimeError("Invalid band definition in tensor defaults.")
        name = str(item.get("name", "")).strip()
        if not name:
            raise RuntimeError("Band definition is missing `name`.")
        bands[name] = (float(item["start"]), float(item["end"]))
    return bands


def _band_for_freq(freq: float, bands: dict[str, tuple[float, float]]) -> str | None:
    for name, (low, high) in bands.items():
        if name == "gamma":
            if low <= freq <= high:
                return name
        else:
            if low <= freq < high:
                return name
    return None


def _compute_bandpower(
    bands: dict[str, tuple[float, float]],
    channel_amplitudes: dict[str, dict[float, float]],
) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, float]]:
    bandpower_per_channel: dict[str, dict[str, float]] = {
        ch: {b: 0.0 for b in bands} for ch in channel_amplitudes
    }

    for channel, amps in channel_amplitudes.items():
        for freq, amp in amps.items():
            band = _band_for_freq(freq, bands)
            if band is None:
                continue
            power = (amp * amp) / 2.0
            bandpower_per_channel[channel][band] += power

    total_power_per_channel = {
        ch: sum(bandpower_per_channel[ch].values()) for ch in channel_amplitudes
    }

    bandpower_mean = {b: 0.0 for b in bands}
    for band in bands:
        bandpower_mean[band] = sum(
            bandpower_per_channel[ch][band] for ch in channel_amplitudes
        ) / len(channel_amplitudes)

    return bandpower_per_channel, total_power_per_channel, bandpower_mean


def _normalize_bands(bands: dict[str, tuple[float, float]]) -> dict[str, list[float]]:
    return {name: [float(low), float(high)] for name, (low, high) in bands.items()}


def _write_bandpower_csv(
    out_path: Path,
    bands: dict[str, tuple[float, float]],
    bandpower_per_channel: dict[str, dict[str, float]],
) -> None:
    header = ["channel", *bands.keys()]
    lines = [",".join(header)]
    for channel, band_values in bandpower_per_channel.items():
        row = [channel] + [f"{band_values[b]:.12f}" for b in bands]
        lines.append(",".join(row))
    out_path.write_text("\n".join(lines) + "\n")


def _resolve_core_info() -> dict[str, str]:
    import lfptensorpipe as core  # noqa: F401

    try:
        version = getattr(core, "__version__")
    except Exception:
        try:
            version = metadata.version("lfp-tensorpipe")
        except Exception:
            try:
                version = metadata.version("lfptensorpipe")
            except Exception:
                version = "unknown"

    module_path = str(getattr(core, "__file__", "unknown"))
    info = {
        "version": version,
        "module_path": module_path,
        "package_root": (
            str(Path(module_path).resolve().parent)
            if module_path != "unknown"
            else "unknown"
        ),
    }

    return info


def run_demo(out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]

    bands = _load_bands()
    bands_norm = _normalize_bands(bands)
    bandpower_per_channel, total_power_per_channel, bandpower_mean = _compute_bandpower(
        bands, CHANNEL_AMPLITUDES
    )

    metrics = {
        "schema_version": 1,
        "bands_hz": bands_norm,
        "bandpower_per_channel": _round_obj(bandpower_per_channel),
        "total_power_per_channel": _round_obj(total_power_per_channel),
        "bandpower_mean": _round_obj(bandpower_mean),
    }

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    bandpower_csv_path = out_dir / "bandpower.csv"
    _write_bandpower_csv(bandpower_csv_path, bands, bandpower_per_channel)

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "core": _resolve_core_info(),
        "workflows": {
            "repo_root": str(repo_root),
            "commit": _git_commit(repo_root),
            "config_defaults_used": ["tensor.yml"],
        },
        "demo": {
            "frequencies_hz": FREQUENCIES_HZ,
            "channel_amplitudes": CHANNEL_AMPLITUDES,
            "bands_hz": bands_norm,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }

    manifest_path = out_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    return {
        "metrics": metrics_path,
        "bandpower_csv": bandpower_csv_path,
        "manifest": manifest_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the lfp-tensorpipe demo."
    )
    parser.add_argument(
        "--out", required=True, help="Output directory for demo artifacts."
    )
    args = parser.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    outputs = run_demo(out_dir)

    print("Demo outputs:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
