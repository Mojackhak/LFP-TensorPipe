from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from lfptensorpipe.app.dataset.source_parser import parse_record_source
from lfptensorpipe import desktop_smoke
from lfptensorpipe.io.pkl_io import save_pkl


def test_legacy_mne_smoke_path_generates_temp_fif_when_repo_fixture_missing(
    tmp_path: Path,
) -> None:
    records_root = tmp_path / "records"
    generated_path = desktop_smoke._legacy_mne_smoke_path(
        records_root,
        temp_dir=tmp_path / "generated",
    )

    assert generated_path.exists()
    assert generated_path.name == "generated_demo.fif"

    raw, report, is_fif_input = parse_record_source(
        import_type="Legacy (MNE supported)",
        paths={"file_path": str(generated_path)},
        options=None,
    )

    assert is_fif_input is True
    assert report["vendor"] == "Legacy (MNE supported)"
    assert raw.ch_names == ["demo-01", "demo-02"]


def test_legacy_mne_smoke_path_prefers_checked_in_fixture(tmp_path: Path) -> None:
    records_root = tmp_path / "records"
    checked_in_path = records_root / "mne" / "gait.fif"
    desktop_smoke._write_smoke_mne_fixture(checked_in_path)

    resolved_path = desktop_smoke._legacy_mne_smoke_path(
        records_root,
        temp_dir=tmp_path / "generated",
    )

    assert resolved_path == checked_in_path


def test_should_emit_smoke_output_disables_frozen_windows() -> None:
    assert (
        desktop_smoke._should_emit_smoke_output(platform="win32", frozen=True) is False
    )
    assert (
        desktop_smoke._should_emit_smoke_output(platform="win32", frozen=False) is True
    )


def test_smoke_print_is_noop_when_output_disabled(capsys) -> None:
    desktop_smoke._smoke_print("hidden", emit_output=False)
    desktop_smoke._smoke_print("shown", emit_output=True)

    assert capsys.readouterr().out == "shown\n"


def test_compare_pkl_trees_ignores_macos_sidecar_pickles(tmp_path: Path) -> None:
    reference_root = tmp_path / "reference"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    save_pkl({"value": 1.0}, reference_root / "metric.pkl")
    save_pkl({"value": 1.0}, candidate_root / "metric.pkl")
    (reference_root / "._metric.pkl").write_bytes(b"appledouble-sidecar")

    desktop_smoke._compare_pkl_trees(
        reference_root=reference_root,
        candidate_root=candidate_root,
        rtol=1e-5,
        atol=1e-8,
    )


def test_compare_pkl_trees_applies_windows_specparam_tolerance(tmp_path: Path) -> None:
    reference_root = tmp_path / "reference"
    candidate_root = tmp_path / "candidate"
    relative_path = Path("cycle-l") / "aperiodic" / "mean-trace.pkl"
    reference_frame = pd.DataFrame({"Value": [1.0, 2.0, 3.0]})
    candidate_frame = pd.DataFrame({"Value": [1.005, 2.005, 3.005]})
    save_pkl(reference_frame, reference_root / relative_path)
    save_pkl(candidate_frame, candidate_root / relative_path)

    desktop_smoke._compare_pkl_trees(
        reference_root=reference_root,
        candidate_root=candidate_root,
        rtol=1e-5,
        atol=1e-8,
        platform="win32",
    )


def test_compare_pkl_trees_keeps_default_tolerance_for_non_specparam_payloads(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "reference"
    candidate_root = tmp_path / "candidate"
    relative_path = Path("cycle-l") / "coherence" / "mean-trace.pkl"
    reference_frame = pd.DataFrame({"Value": [1.0, 2.0, 3.0]})
    candidate_frame = pd.DataFrame({"Value": [1.005, 2.005, 3.005]})
    save_pkl(reference_frame, reference_root / relative_path)
    save_pkl(candidate_frame, candidate_root / relative_path)

    with pytest.raises(RuntimeError, match="Numeric mismatch"):
        desktop_smoke._compare_pkl_trees(
            reference_root=reference_root,
            candidate_root=candidate_root,
            rtol=1e-5,
            atol=1e-8,
            platform="win32",
        )
