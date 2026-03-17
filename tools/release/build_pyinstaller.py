"""Build PyInstaller desktop app artifacts from the private repository."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

APP_NAME = "LFP-TensorPipe"
MACOS_TARGET = "macos"
WINDOWS_TARGET = "windows"
WINDOWS_EXECUTABLE_NAME = f"{APP_NAME}.exe"
_DESKTOP_VERSION_PATTERN = re.compile(
    r"^v(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)"
    r"-(?P<distance>\d+)-g(?P<sha>[0-9a-f]+)(?P<dirty>-dirty)?$"
)


@dataclass(frozen=True)
class DesktopVersionInfo:
    artifact_version: str
    bundle_short_version: str
    bundle_build_version: str
    windows_file_version: tuple[int, int, int, int]

    @property
    def windows_file_version_text(self) -> str:
        return ".".join(str(part) for part in self.windows_file_version)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _icon_root(repo: Path) -> Path:
    return repo / "src" / "lfptensorpipe" / "gui" / "assets" / "icons"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build_pyinstaller",
        description=(
            "Build native PyInstaller desktop artifacts from the private "
            "source repository."
        ),
    )
    parser.add_argument(
        "--target-platform",
        choices=(MACOS_TARGET, WINDOWS_TARGET),
        default=MACOS_TARGET,
        help="Target platform to build. Must run on the matching native host.",
    )
    parser.add_argument(
        "--dist-dir",
        default=None,
        help="Final output directory for the app bundle and dmg.",
    )
    parser.add_argument(
        "--pyinstaller-work-dir",
        default=None,
        help="Scratch directory for PyInstaller work and temporary dist output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved plan without running external commands.",
    )
    parser.add_argument(
        "--skip-dmg",
        action="store_true",
        help="Build the .app bundle only and skip dmg creation (macOS only).",
    )
    parser.add_argument(
        "--dmg-only",
        action="store_true",
        help=(
            "Skip PyInstaller and rebuild only the dmg from an existing app "
            "bundle (macOS only)."
        ),
    )
    return parser.parse_args(argv)


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    dry_run: bool = False,
    env: dict[str, str] | None = None,
) -> None:
    if dry_run:
        if env:
            overrides = " ".join(f"{key}={value}" for key, value in sorted(env.items()))
            print(f"DRY RUN: env {overrides} {' '.join(cmd)}")
        else:
            print("DRY RUN:", " ".join(cmd))
        return
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=str(repo_root()),
        text=True,
    ).strip()


def _fallback_desktop_version() -> DesktopVersionInfo:
    try:
        sha = _git_output("rev-parse", "--short", "HEAD")
    except subprocess.CalledProcessError:
        sha = "unknown"
    return DesktopVersionInfo(
        artifact_version=f"0.0.1-dev0-g{sha}",
        bundle_short_version="0.0.1",
        bundle_build_version="0.0.1.0",
        windows_file_version=(0, 0, 1, 0),
    )


def _next_patch_version(major: int, minor: int, patch: int) -> tuple[int, int, int]:
    return major, minor, patch + 1


def _utc_date_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d")


@lru_cache(maxsize=1)
def resolve_desktop_version() -> DesktopVersionInfo:
    try:
        describe = _git_output(
            "describe",
            "--tags",
            "--dirty",
            "--long",
            "--match",
            "v[0-9]*",
        )
    except subprocess.CalledProcessError:
        return _fallback_desktop_version()

    match = _DESKTOP_VERSION_PATTERN.fullmatch(describe)
    if match is None:
        return _fallback_desktop_version()

    major = int(match.group("major"))
    minor = int(match.group("minor"))
    patch = int(match.group("patch"))
    distance = int(match.group("distance"))
    sha = match.group("sha")
    dirty = match.group("dirty") is not None
    short_version = f"{major}.{minor}.{patch}"

    if distance == 0 and not dirty:
        artifact_version = short_version
        bundle_short_version = short_version
        build_index = 0
    else:
        next_major, next_minor, next_patch = _next_patch_version(
            major,
            minor,
            patch,
        )
        bundle_short_version = f"{next_major}.{next_minor}.{next_patch}"
        suffix = f"dev{distance}-g{sha}"
        if dirty:
            suffix += f"-d{_utc_date_stamp()}"
        artifact_version = f"{bundle_short_version}-{suffix}"
        major, minor, patch = next_major, next_minor, next_patch
        build_index = distance

    return DesktopVersionInfo(
        artifact_version=artifact_version,
        bundle_short_version=bundle_short_version,
        bundle_build_version=(
            bundle_short_version
            if build_index == 0 and artifact_version == bundle_short_version
            else f"{bundle_short_version}.{build_index}"
        ),
        windows_file_version=(major, minor, patch, build_index),
    )


def default_macos_artifact_paths(
    *,
    dist_dir: Path | None = None,
    version_info: DesktopVersionInfo | None = None,
) -> tuple[Path, Path]:
    resolved_dist_dir = (
        dist_dir
        if dist_dir is not None
        else repo_root() / "dist" / "desktop" / MACOS_TARGET
    )
    version = version_info or resolve_desktop_version()
    artifact_prefix = f"{APP_NAME}-{version.artifact_version}"
    return (
        resolved_dist_dir / f"{artifact_prefix}.app",
        resolved_dist_dir / f"{artifact_prefix}.dmg",
    )


def default_windows_artifact_paths(
    *,
    dist_dir: Path | None = None,
    version_info: DesktopVersionInfo | None = None,
) -> tuple[Path, Path]:
    resolved_dist_dir = (
        dist_dir
        if dist_dir is not None
        else repo_root() / "dist" / "desktop" / WINDOWS_TARGET
    )
    version = version_info or resolve_desktop_version()
    artifact_prefix = f"{APP_NAME}-{version.artifact_version}"
    return (
        resolved_dist_dir / artifact_prefix,
        resolved_dist_dir / f"{artifact_prefix}-windows-x86_64.zip",
    )


def _pyinstaller_version_env(version_info: DesktopVersionInfo) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "LFPTP_DESKTOP_ARTIFACT_VERSION": version_info.artifact_version,
            "LFPTP_MACOS_BUNDLE_SHORT_VERSION": version_info.bundle_short_version,
            "LFPTP_MACOS_BUNDLE_BUILD_VERSION": version_info.bundle_build_version,
            "LFPTP_WINDOWS_FILE_VERSION": version_info.windows_file_version_text,
            "LFPTP_WINDOWS_PRODUCT_VERSION": version_info.artifact_version,
        }
    )
    return env


def _icon_source_candidates(icon_root: Path) -> tuple[Path, ...]:
    return (
        icon_root / "png" / "app_icon.png",
        icon_root / "app_icon.png",
        icon_root / "app_icon.svg",
        icon_root / "png" / "app_icon.svg",
    )


def _required_generated_icon_paths(
    icon_root: Path, *, target_platform: str
) -> tuple[Path, ...]:
    required = [icon_root / "png" / "icon_256.png"]
    if target_platform == MACOS_TARGET:
        required.append(icon_root / "macos" / "lfptensorpipe.icns")
    elif target_platform == WINDOWS_TARGET:
        required.append(icon_root / "windows" / "lfptensorpipe.ico")
    else:
        raise ValueError(f"Unsupported target platform: {target_platform}")
    return tuple(required)


def _refresh_icon_assets(*, target_platform: str, dry_run: bool = False) -> None:
    repo = repo_root()
    icon_root = _icon_root(repo)
    source_candidates = _icon_source_candidates(icon_root)

    if any(path.exists() for path in source_candidates):
        _run(
            [sys.executable, "-m", "lfptensorpipe.gui.icon_pipeline"],
            cwd=repo,
            dry_run=dry_run,
        )
        return

    required_generated = _required_generated_icon_paths(
        icon_root,
        target_platform=target_platform,
    )
    if all(path.exists() for path in required_generated):
        prefix = "DRY RUN: " if dry_run else ""
        print(
            f"{prefix}skip icon refresh; using existing generated assets in "
            f"{icon_root}"
        )
        return

    missing_outputs = ", ".join(
        str(path) for path in required_generated if not path.exists()
    )
    raise FileNotFoundError(
        "Icon source master is missing and generated icon assets are incomplete. "
        f"Tried source candidates: {', '.join(str(path) for path in source_candidates)}. "
        f"Missing generated outputs: {missing_outputs}"
    )


def _copy_directory(source: Path, destination: Path) -> None:
    if destination.exists():
        last_error: PermissionError | None = None
        for attempt in range(3):
            try:
                shutil.rmtree(destination)
                last_error = None
                break
            except PermissionError as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(1.0)
        if last_error is not None:
            exe_hint = destination / WINDOWS_EXECUTABLE_NAME
            raise RuntimeError(
                "Failed to replace existing build output at "
                f"{destination} because it is locked by another process. "
                "Close the packaged app and rerun the build."
                + (f" Locked executable: {exe_hint}" if exe_hint.exists() else "")
            ) from last_error
    shutil.copytree(source, destination)


def _format_size(num_bytes: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)}{unit}"
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def _directory_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for file_name in files:
            file_path = Path(root) / file_name
            try:
                total += file_path.stat().st_size
            except OSError:
                continue
    return total


def _print_macos_artifact_sizes(*, app_path: Path, dmg_path: Path) -> None:
    if app_path.exists():
        print(
            f"Built app bundle: {app_path} ({_format_size(_directory_size(app_path))})"
        )
    if dmg_path.exists():
        print(f"Built dmg: {dmg_path} ({_format_size(dmg_path.stat().st_size)})")


def _print_windows_artifact_sizes(*, app_dir: Path, zip_path: Path) -> None:
    if app_dir.exists():
        print(
            f"Built app directory: {app_dir} "
            f"({_format_size(_directory_size(app_dir))})"
        )
    if zip_path.exists():
        print(f"Built zip: {zip_path} ({_format_size(zip_path.stat().st_size)})")


def build_zip(
    *, source_dir: Path, destination_zip: Path, dry_run: bool = False
) -> None:
    if dry_run:
        print(f"DRY RUN: zip {source_dir} -> {destination_zip}")
        return
    if destination_zip.exists():
        destination_zip.unlink()
    destination_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        destination_zip,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_dir():
                continue
            archive.write(
                path,
                arcname=(
                    Path(source_dir.name) / path.relative_to(source_dir)
                ).as_posix(),
            )


def build_dmg(*, app_path: Path, dmg_path: Path, dry_run: bool = False) -> None:
    if dmg_path.exists() and not dry_run:
        dmg_path.unlink()
    with tempfile.TemporaryDirectory(prefix="lfptp_dmg_") as tmp_dir_raw:
        tmp_dir = Path(tmp_dir_raw)
        staged_app = tmp_dir / app_path.name
        if dry_run:
            print(f"DRY RUN: stage app bundle at {staged_app}")
        else:
            shutil.copytree(app_path, staged_app)
        _run(
            [
                "hdiutil",
                "create",
                "-volname",
                APP_NAME,
                "-srcfolder",
                str(tmp_dir),
                "-ov",
                "-format",
                "UDZO",
                str(dmg_path),
            ],
            cwd=repo_root(),
            dry_run=dry_run,
        )


def build_macos(
    *,
    dry_run: bool = False,
    dist_dir: Path,
    work_dir: Path,
    skip_dmg: bool = False,
    dmg_only: bool = False,
) -> None:
    repo = repo_root()
    spec_path = repo / "packaging" / "pyinstaller" / "LFP-TensorPipe.spec"
    pyinstaller_dist = work_dir / "dist"
    pyinstaller_work = work_dir / "work"
    version_info = resolve_desktop_version()
    pyinstaller_env = _pyinstaller_version_env(version_info)
    final_app, final_dmg = default_macos_artifact_paths(
        dist_dir=dist_dir,
        version_info=version_info,
    )

    if not dry_run:
        dist_dir.mkdir(parents=True, exist_ok=True)
        if not dmg_only:
            if work_dir.exists():
                shutil.rmtree(work_dir)
            work_dir.mkdir(parents=True, exist_ok=True)

    if not dmg_only:
        _refresh_icon_assets(target_platform=MACOS_TARGET, dry_run=dry_run)
        _run(
            [
                sys.executable,
                "-m",
                "PyInstaller",
                "--clean",
                "--noconfirm",
                "--distpath",
                str(pyinstaller_dist),
                "--workpath",
                str(pyinstaller_work),
                str(spec_path),
            ],
            cwd=repo,
            dry_run=dry_run,
            env=pyinstaller_env,
        )

        built_app = pyinstaller_dist / f"{APP_NAME}.app"
        if dry_run:
            print(f"DRY RUN: copy {built_app} -> {final_app}")
        else:
            if not built_app.exists():
                raise FileNotFoundError(
                    f"PyInstaller app bundle not found: {built_app}"
                )
            _copy_directory(built_app, final_app)
    elif not dry_run and not final_app.exists():
        raise FileNotFoundError(
            f"Existing app bundle not found for dmg-only mode: {final_app}"
        )

    if skip_dmg:
        if not dry_run:
            _print_macos_artifact_sizes(app_path=final_app, dmg_path=final_dmg)
        return

    build_dmg(app_path=final_app, dmg_path=final_dmg, dry_run=dry_run)
    if not dry_run:
        _print_macos_artifact_sizes(app_path=final_app, dmg_path=final_dmg)


def build_windows(
    *,
    dry_run: bool = False,
    dist_dir: Path,
    work_dir: Path,
) -> None:
    repo = repo_root()
    spec_path = repo / "packaging" / "pyinstaller" / "LFP-TensorPipe.spec"
    pyinstaller_dist = work_dir / "dist"
    pyinstaller_work = work_dir / "work"
    version_info = resolve_desktop_version()
    pyinstaller_env = _pyinstaller_version_env(version_info)
    final_app_dir, final_zip = default_windows_artifact_paths(
        dist_dir=dist_dir,
        version_info=version_info,
    )

    if not dry_run:
        dist_dir.mkdir(parents=True, exist_ok=True)
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    _refresh_icon_assets(target_platform=WINDOWS_TARGET, dry_run=dry_run)
    _run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--clean",
            "--noconfirm",
            "--distpath",
            str(pyinstaller_dist),
            "--workpath",
            str(pyinstaller_work),
            str(spec_path),
        ],
        cwd=repo,
        dry_run=dry_run,
        env=pyinstaller_env,
    )

    built_app_dir = pyinstaller_dist / APP_NAME
    if dry_run:
        print(f"DRY RUN: copy {built_app_dir} -> {final_app_dir}")
    else:
        if not built_app_dir.exists():
            raise FileNotFoundError(
                f"PyInstaller app directory not found: {built_app_dir}"
            )
        _copy_directory(built_app_dir, final_app_dir)

    build_zip(source_dir=final_app_dir, destination_zip=final_zip, dry_run=dry_run)
    if not dry_run:
        _print_windows_artifact_sizes(app_dir=final_app_dir, zip_path=final_zip)


def _is_windows_platform(platform_name: str) -> bool:
    return platform_name.startswith("win")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = repo_root()

    if args.skip_dmg and args.dmg_only:
        raise SystemExit("--skip-dmg and --dmg-only cannot be used together.")

    if args.target_platform == MACOS_TARGET:
        if sys.platform != "darwin":
            raise SystemExit("The macOS PyInstaller target must run on a macOS host.")
        dist_dir = (
            Path(args.dist_dir).expanduser().resolve()
            if args.dist_dir
            else repo / "dist" / "desktop" / MACOS_TARGET
        )
        work_dir = (
            Path(args.pyinstaller_work_dir).expanduser().resolve()
            if args.pyinstaller_work_dir
            else repo / "build" / "pyinstaller" / MACOS_TARGET
        )
        build_macos(
            dry_run=args.dry_run,
            dist_dir=dist_dir,
            work_dir=work_dir,
            skip_dmg=args.skip_dmg,
            dmg_only=args.dmg_only,
        )
        return 0

    if args.skip_dmg or args.dmg_only:
        raise SystemExit(
            "--skip-dmg and --dmg-only are available only for the macOS target."
        )
    if not _is_windows_platform(sys.platform):
        raise SystemExit("The Windows PyInstaller target must run on a Windows host.")

    dist_dir = (
        Path(args.dist_dir).expanduser().resolve()
        if args.dist_dir
        else repo / "dist" / "desktop" / WINDOWS_TARGET
    )
    work_dir = (
        Path(args.pyinstaller_work_dir).expanduser().resolve()
        if args.pyinstaller_work_dir
        else repo / "build" / "pyinstaller" / WINDOWS_TARGET
    )
    build_windows(
        dry_run=args.dry_run,
        dist_dir=dist_dir,
        work_dir=work_dir,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint guard
    raise SystemExit(main())
