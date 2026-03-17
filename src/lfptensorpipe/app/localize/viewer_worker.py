"""Independent worker process for MATLAB Contact Viewer."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

from lfptensorpipe.matlab import ensure_matlab_engine

CONTACT_VIEWER_ENTRYPOINT = "contact_viewer"
CONTACT_VIEWER_STARTUP_TIMEOUT_SEC = 15.0
CONTACT_VIEWER_EMPTY_CONFIRM_CYCLES = 4
CONTACT_VIEWER_WAIT_POLL_SEC = 0.2
CONTACT_VIEWER_WAIT_SCRIPT = (
    "drawnow; "
    "startup_tic=tic; "
    "seen_fig=false; "
    "empty_cycles=0; "
    "while true; "
    "figs=findall(0,'Type','figure'); "
    "if isempty(figs); "
    f"if ~seen_fig && toc(startup_tic)>={CONTACT_VIEWER_STARTUP_TIMEOUT_SEC}; break; end; "
    "if seen_fig; "
    "empty_cycles=empty_cycles+1; "
    f"if empty_cycles>={CONTACT_VIEWER_EMPTY_CONFIRM_CYCLES}; break; end; "
    "end; "
    "else; "
    "seen_fig=true; "
    "empty_cycles=0; "
    "end; "
    f"pause({CONTACT_VIEWER_WAIT_POLL_SEC}); "
    "drawnow; "
    "end"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="lfptensorpipe-contact-viewer-worker")
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--leaddbs-dir", required=True)
    parser.add_argument("--matlab-root", required=False)
    parser.add_argument(
        "--matlab-engine-path", dest="matlab_engine_path", required=False
    )
    return parser.parse_args(argv)


def run_worker(
    *,
    csv_path: Path,
    atlas: str,
    leaddbs_dir: Path,
    matlab_root: Path,
    ensure_matlab_engine_fn: Callable[[Path], Any] = ensure_matlab_engine,
    start_matlab_fn: Callable[[], Any] | None = None,
    viewer_function_dir: Path | None = None,
) -> None:
    ensure_matlab_engine_fn(matlab_root)
    if start_matlab_fn is None:
        import matlab.engine

        start_matlab_fn = matlab.engine.start_matlab

    viewer_fn_dir = viewer_function_dir or _resolve_viewer_function_dir(
        Path(__file__).resolve()
    )

    eng = start_matlab_fn()
    eng.addpath(eng.genpath(str(leaddbs_dir)), nargout=0)
    if viewer_fn_dir.is_dir():
        eng.addpath(str(viewer_fn_dir), nargout=0)

    resolved = str(eng.which(CONTACT_VIEWER_ENTRYPOINT))
    if not resolved:
        raise RuntimeError(
            f"MATLAB function `{CONTACT_VIEWER_ENTRYPOINT}` not found on path. "
            "Ensure `src/lfptensorpipe/anat/leaddbs` is available."
        )
    eng.feval(CONTACT_VIEWER_ENTRYPOINT, str(csv_path), atlas, nargout=0)
    # Keep the detached worker/engine alive until user closes all MATLAB viewer windows.
    eng.eval(CONTACT_VIEWER_WAIT_SCRIPT, nargout=0)


def _resolve_viewer_function_dir(module_file: Path) -> Path:
    """Resolve the bundled or source `lfptensorpipe/anat/leaddbs` directory."""
    package_root = module_file.resolve().parents[2]
    package_relative = package_root / "anat" / "leaddbs"
    if package_relative.is_dir():
        return package_relative
    for parent in module_file.parents:
        candidate = parent / "src" / "lfptensorpipe" / "anat" / "leaddbs"
        if candidate.is_dir():
            return candidate
    return package_relative


def main(
    argv: list[str] | None = None,
    *,
    run_worker_fn: Callable[..., None] = run_worker,
) -> int:
    args = parse_args(argv)
    matlab_root_raw = args.matlab_root or args.matlab_engine_path
    if not matlab_root_raw:
        raise SystemExit("--matlab-root is required")
    run_worker_fn(
        csv_path=Path(args.csv_path).expanduser().resolve(),
        atlas=args.atlas,
        leaddbs_dir=Path(args.leaddbs_dir).expanduser().resolve(),
        matlab_root=Path(matlab_root_raw).expanduser().resolve(),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint guard
    raise SystemExit(main())
