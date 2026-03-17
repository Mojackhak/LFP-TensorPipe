"""Sign the macOS app bundle and refresh the dmg from the signed app."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.release.build_pyinstaller import (  # noqa: E402
    build_dmg,
    default_macos_artifact_paths,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sign_macos_app",
        description="Sign the macOS app bundle with a Developer ID identity.",
    )
    parser.add_argument(
        "--app-path",
        default=None,
        help=(
            "Path to the .app bundle. Defaults to the version-resolved app "
            "bundle under dist/desktop/macos."
        ),
    )
    parser.add_argument(
        "--dmg-path",
        default=None,
        help=(
            "Path to the dmg output. Defaults to the version-resolved dmg "
            "under dist/desktop/macos."
        ),
    )
    parser.add_argument(
        "--codesign-identity",
        default=os.environ.get("LFPTP_CODESIGN_IDENTITY"),
        help="Developer ID Application identity. Defaults to LFPTP_CODESIGN_IDENTITY.",
    )
    parser.add_argument(
        "--skip-dmg-refresh",
        action="store_true",
        help="Sign and verify the app, but do not rebuild the dmg.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args(argv)


def _run(cmd: list[str], *, dry_run: bool = False) -> None:
    if dry_run:
        print("DRY RUN:", " ".join(cmd))
        return
    subprocess.run(cmd, check=True)


def _resolve_default_path(value: str | None, *, relative: str) -> Path:
    if value is not None:
        return Path(value).expanduser().resolve()
    return REPO_ROOT / relative


def _resolve_default_macos_paths(
    app_value: str | None,
    dmg_value: str | None,
) -> tuple[Path, Path]:
    default_app_path, default_dmg_path = default_macos_artifact_paths()
    app_path = (
        _resolve_default_path(app_value, relative="")
        if app_value is not None
        else default_app_path
    )
    dmg_path = (
        _resolve_default_path(dmg_value, relative="")
        if dmg_value is not None
        else default_dmg_path
    )
    return app_path, dmg_path


def _require_codesign_identity(value: str | None) -> str:
    if value is None or not value.strip():
        raise SystemExit(
            "A Developer ID identity is required. Set LFPTP_CODESIGN_IDENTITY "
            "or pass --codesign-identity."
        )
    return value.strip()


def sign_app(*, app_path: Path, codesign_identity: str, dry_run: bool = False) -> None:
    _run(
        [
            "codesign",
            "--force",
            "--deep",
            "--options",
            "runtime",
            "--timestamp",
            "--sign",
            codesign_identity,
            str(app_path),
        ],
        dry_run=dry_run,
    )
    _run(
        [
            "codesign",
            "--verify",
            "--deep",
            "--strict",
            "--verbose=2",
            str(app_path),
        ],
        dry_run=dry_run,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if sys.platform != "darwin":
        raise SystemExit("macOS signing helpers can only run on macOS.")

    app_path, dmg_path = _resolve_default_macos_paths(
        args.app_path,
        args.dmg_path,
    )
    codesign_identity = _require_codesign_identity(args.codesign_identity)

    if not args.dry_run and not app_path.exists():
        raise SystemExit(f"App bundle not found: {app_path}")

    sign_app(
        app_path=app_path,
        codesign_identity=codesign_identity,
        dry_run=args.dry_run,
    )

    if args.skip_dmg_refresh:
        return 0

    if dmg_path.exists() and not args.dry_run:
        dmg_path.unlink()
    if not args.dry_run:
        dmg_path.parent.mkdir(parents=True, exist_ok=True)

    build_dmg(app_path=app_path, dmg_path=dmg_path, dry_run=args.dry_run)
    if not args.dry_run and not dmg_path.exists():
        raise SystemExit(f"Failed to rebuild dmg at: {dmg_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint guard
    raise SystemExit(main())
