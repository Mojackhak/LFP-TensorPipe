"""Unit tests for Contact Viewer MATLAB worker entrypoint resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from lfptensorpipe.app import localize_viewer_worker


class _FakeMatlabEngine:
    def __init__(self, *, entrypoint_exists: bool) -> None:
        self.entrypoint_exists = entrypoint_exists
        self.addpath_calls: list[str] = []
        self.feval_calls: list[tuple[str, str, str, int]] = []
        self.eval_calls: list[tuple[str, int]] = []
        self.genpath_arg: str | None = None

    def genpath(self, path: str) -> str:
        self.genpath_arg = path
        return f"GENPATH::{path}"

    def addpath(self, path: str, nargout: int = 0) -> None:
        _ = nargout
        self.addpath_calls.append(path)

    def which(self, symbol: str) -> str:
        if (
            symbol == localize_viewer_worker.CONTACT_VIEWER_ENTRYPOINT
            and self.entrypoint_exists
        ):
            return "/tmp/contact_viewer.m"
        return ""

    def feval(self, symbol: str, csv_path: str, atlas: str, nargout: int = 0) -> None:
        self.feval_calls.append((symbol, csv_path, atlas, nargout))

    def eval(self, script: str, nargout: int = 0) -> None:
        self.eval_calls.append((script, nargout))


def test_run_worker_calls_unique_contact_viewer_entrypoint(
    tmp_path: Path,
) -> None:
    fake_engine = _FakeMatlabEngine(entrypoint_exists=True)

    csv_path = tmp_path / "contact_midpoint_coords.csv"
    csv_path.write_text("MNI_x,MNI_y,MNI_z\n1,2,3\n", encoding="utf-8")
    leaddbs_dir = tmp_path / "leaddbs"
    leaddbs_dir.mkdir()
    matlab_root = tmp_path / "matlab"
    matlab_root.mkdir()

    localize_viewer_worker.run_worker(
        csv_path=csv_path,
        atlas="DISTAL",
        leaddbs_dir=leaddbs_dir,
        matlab_root=matlab_root,
        ensure_matlab_engine_fn=lambda _: None,
        start_matlab_fn=lambda: fake_engine,
    )

    assert fake_engine.genpath_arg == str(leaddbs_dir)
    assert any(call.startswith("GENPATH::") for call in fake_engine.addpath_calls)
    assert any(
        call.endswith("lfptensorpipe/anat/leaddbs")
        for call in fake_engine.addpath_calls
    )
    assert fake_engine.feval_calls == [
        (localize_viewer_worker.CONTACT_VIEWER_ENTRYPOINT, str(csv_path), "DISTAL", 0)
    ]
    assert fake_engine.eval_calls == [
        (localize_viewer_worker.CONTACT_VIEWER_WAIT_SCRIPT, 0)
    ]


def test_run_worker_raises_when_entrypoint_missing(
    tmp_path: Path,
) -> None:
    fake_engine = _FakeMatlabEngine(entrypoint_exists=False)

    with pytest.raises(RuntimeError, match="contact_viewer"):
        localize_viewer_worker.run_worker(
            csv_path=tmp_path / "contact_midpoint_coords.csv",
            atlas="DISTAL",
            leaddbs_dir=tmp_path / "leaddbs",
            matlab_root=tmp_path / "matlab",
            ensure_matlab_engine_fn=lambda _: None,
            start_matlab_fn=lambda: fake_engine,
        )


def test_parse_args_requires_all_contact_viewer_options() -> None:
    args = localize_viewer_worker.parse_args(
        [
            "--csv-path",
            "/tmp/c.csv",
            "--atlas",
            "DISTAL",
            "--leaddbs-dir",
            "/tmp/leaddbs",
            "--matlab-root",
            "/tmp/matlab",
        ]
    )
    assert args.csv_path == "/tmp/c.csv"
    assert args.atlas == "DISTAL"
    assert args.leaddbs_dir == "/tmp/leaddbs"
    assert args.matlab_root == "/tmp/matlab"


def test_wait_script_uses_startup_and_empty_cycle_guards() -> None:
    script = localize_viewer_worker.CONTACT_VIEWER_WAIT_SCRIPT
    assert "startup_tic=tic;" in script
    assert "seen_fig=false;" in script
    assert "empty_cycles=0;" in script
    assert "toc(startup_tic)>=" in script
    assert "empty_cycles=empty_cycles+1;" in script


def test_main_resolves_paths_and_forwards_to_run_worker(
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}

    def _fake_run_worker(**kwargs: object) -> None:
        calls.update(kwargs)

    csv_path = tmp_path / "contact_midpoint_coords.csv"
    leaddbs_dir = tmp_path / "leaddbs"
    matlab_root = tmp_path / "matlab"
    exit_code = localize_viewer_worker.main(
        [
            "--csv-path",
            str(csv_path),
            "--atlas",
            "DISTAL",
            "--leaddbs-dir",
            str(leaddbs_dir),
            "--matlab-root",
            str(matlab_root),
        ],
        run_worker_fn=_fake_run_worker,
    )

    assert exit_code == 0
    assert calls["csv_path"] == csv_path.expanduser().resolve()
    assert calls["atlas"] == "DISTAL"
    assert calls["leaddbs_dir"] == leaddbs_dir.expanduser().resolve()
    assert calls["matlab_root"] == matlab_root.expanduser().resolve()
