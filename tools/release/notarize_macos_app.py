"""Submit the signed macOS dmg for notarization and staple the result."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.release.build_pyinstaller import default_macos_artifact_paths  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="notarize_macos_app",
        description="Submit the signed macOS dmg with notarytool and staple the result.",
    )
    parser.add_argument(
        "--app-path",
        default=None,
        help=(
            "Path to the signed .app bundle. Defaults to the "
            "version-resolved app bundle under dist/desktop/macos."
        ),
    )
    parser.add_argument(
        "--dmg-path",
        default=None,
        help=(
            "Path to the signed dmg. Defaults to the version-resolved dmg "
            "under dist/desktop/macos."
        ),
    )
    parser.add_argument(
        "--notary-profile",
        default=os.environ.get("LFPTP_NOTARY_PROFILE"),
        help="notarytool keychain profile. Defaults to LFPTP_NOTARY_PROFILE.",
    )
    parser.add_argument(
        "--timeout",
        default="30m",
        help="notarytool wait timeout. Defaults to 30m.",
    )
    parser.add_argument(
        "--skip-staple",
        action="store_true",
        help="Submit and wait, but skip stapling and post-submit assessment.",
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


def _require_notary_profile(value: str | None) -> str:
    if value is None or not value.strip():
        raise SystemExit(
            "A notarytool keychain profile is required. Set LFPTP_NOTARY_PROFILE "
            "or pass --notary-profile."
        )
    return value.strip()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if sys.platform != "darwin":
        raise SystemExit("macOS notarization helpers can only run on macOS.")

    app_path, dmg_path = _resolve_default_macos_paths(
        args.app_path,
        args.dmg_path,
    )
    notary_profile = _require_notary_profile(args.notary_profile)

    if not args.dry_run:
        if not app_path.exists():
            raise SystemExit(f"Signed app bundle not found: {app_path}")
        if not dmg_path.exists():
            raise SystemExit(f"Signed dmg not found: {dmg_path}")

    _run(
        [
            "xcrun",
            "notarytool",
            "submit",
            str(dmg_path),
            "--keychain-profile",
            notary_profile,
            "--wait",
            "--timeout",
            args.timeout,
        ],
        dry_run=args.dry_run,
    )

    if args.skip_staple:
        return 0

    _run(["xcrun", "stapler", "staple", str(app_path)], dry_run=args.dry_run)
    _run(["xcrun", "stapler", "validate", str(app_path)], dry_run=args.dry_run)
    _run(["xcrun", "stapler", "staple", str(dmg_path)], dry_run=args.dry_run)
    _run(["xcrun", "stapler", "validate", str(dmg_path)], dry_run=args.dry_run)
    _run(
        ["spctl", "--assess", "--type", "exec", "--verbose=4", str(app_path)],
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint guard
    raise SystemExit(main())
