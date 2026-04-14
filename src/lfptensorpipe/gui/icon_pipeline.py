"""Icon build and runtime resolution helpers."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Callable

PNG_SIZES = (16, 24, 32, 48, 64, 128, 256, 512, 1024)
ICONSET_MAP = {
    "icon_16x16.png": 16,
    "icon_16x16@2x.png": 32,
    "icon_32x32.png": 32,
    "icon_32x32@2x.png": 64,
    "icon_128x128.png": 128,
    "icon_128x128@2x.png": 256,
    "icon_256x256.png": 256,
    "icon_256x256@2x.png": 512,
    "icon_512x512.png": 512,
    "icon_512x512@2x.png": 1024,
}


def default_icon_root() -> Path:
    """Return canonical icon asset directory."""
    return Path(__file__).resolve().parent / "assets" / "icons"


def _resolve_source_asset(
    icon_root: Path,
    source_image: Path | None,
    source_svg: Path | None,
) -> Path:
    if source_image is not None:
        candidate = source_image
        if not candidate.exists():
            raise FileNotFoundError(f"Source image not found: {candidate}")
        return candidate
    if source_svg is not None:
        candidate = source_svg
        if not candidate.exists():
            raise FileNotFoundError(f"Source SVG not found: {candidate}")
        return candidate

    # Prefer raster source to preserve authored text rendering/styling.
    candidates = (
        icon_root / "png" / "app_icon.png",
        icon_root / "app_icon.png",
        icon_root / "app_icon.svg",
        icon_root / "png" / "app_icon.svg",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Icon source not found. Tried: " + ", ".join(str(path) for path in candidates)
    )


def _canonicalize_source(icon_root: Path, source_asset: Path) -> Path:
    suffix = source_asset.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg"}:
        canonical = icon_root / "png" / "app_icon.png"
    else:
        canonical = icon_root / "app_icon.svg"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    if canonical != source_asset:
        shutil.copy2(source_asset, canonical)
    return canonical


def _build_png_set(
    icon_root: Path,
    source_asset: Path,
    *,
    svg_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    raster_loader: Callable[[Path], Any] | None = None,
) -> dict[int, Path]:
    png_dir = icon_root / "png"
    png_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[int, Path] = {}
    is_raster_source = source_asset.suffix.lower() in {".png", ".jpg", ".jpeg"}
    base_image = None
    if is_raster_source:
        from PIL import Image

        loader = raster_loader or Image.open
        base_image = loader(source_asset).convert("RGBA")

    for size in PNG_SIZES:
        out_path = png_dir / f"icon_{size}.png"
        if is_raster_source:
            assert base_image is not None
            from PIL import Image

            resized = base_image.resize((size, size), Image.Resampling.LANCZOS)
            resized.save(out_path, format="PNG")
        else:
            svg_runner(
                [
                    "sips",
                    "-s",
                    "format",
                    "png",
                    "-z",
                    str(size),
                    str(size),
                    str(source_asset),
                    "--out",
                    str(out_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        outputs[size] = out_path
    return outputs


def _build_windows_ico(
    icon_root: Path,
    png_outputs: dict[int, Path],
    *,
    image_cls: type | None = None,
) -> Path:
    windows_dir = icon_root / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)
    ico_path = windows_dir / "lfptensorpipe.ico"
    if image_cls is None:
        from PySide6.QtGui import QImage as image_cls

    source = image_cls(str(png_outputs[256]))
    if source.isNull():
        raise RuntimeError(f"Failed to load PNG for ICO generation: {png_outputs[256]}")
    if not source.save(str(ico_path), "ICO"):
        raise RuntimeError(f"Failed to write ICO file: {ico_path}")
    return ico_path


def _build_macos_icns(
    icon_root: Path,
    png_outputs: dict[int, Path],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> Path:
    macos_dir = icon_root / "macos"
    macos_dir.mkdir(parents=True, exist_ok=True)
    icns_path = macos_dir / "lfptensorpipe.icns"

    with tempfile.TemporaryDirectory(prefix="lfptensorpipe_iconset_") as tmp:
        iconset_dir = Path(tmp) / "lfptensorpipe.iconset"
        iconset_dir.mkdir(parents=True, exist_ok=True)
        for filename, size in ICONSET_MAP.items():
            shutil.copy2(png_outputs[size], iconset_dir / filename)
        runner(
            [
                "iconutil",
                "-c",
                "icns",
                str(iconset_dir),
                "-o",
                str(icns_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    return icns_path


def build_icon_assets(
    *,
    icon_root: Path | None = None,
    source_image: Path | None = None,
    source_svg: Path | None = None,
    resolve_source_asset_fn: Callable[
        [Path, Path | None, Path | None], Path
    ] = _resolve_source_asset,
    canonicalize_source_fn: Callable[[Path, Path], Path] = _canonicalize_source,
    build_png_set_fn: Callable[..., dict[int, Path]] = _build_png_set,
    build_windows_ico_fn: Callable[..., Path] = _build_windows_ico,
    build_macos_icns_fn: Callable[..., Path] = _build_macos_icns,
) -> dict[str, Path]:
    """Generate PNG/ICO/ICNS icon artifacts from source image asset."""
    root = icon_root or default_icon_root()
    root.mkdir(parents=True, exist_ok=True)
    source_asset = resolve_source_asset_fn(root, source_image, source_svg)
    canonical_source = canonicalize_source_fn(root, source_asset)

    png_outputs = build_png_set_fn(root, canonical_source)
    ico_path = build_windows_ico_fn(root, png_outputs)
    icns_path = build_macos_icns_fn(root, png_outputs)
    return {
        "source_asset": canonical_source,
        "png_dir": root / "png",
        "windows_ico": ico_path,
        "macos_icns": icns_path,
    }


def preferred_runtime_icon_path(icon_root: Path | None = None) -> Path | None:
    """Return best available runtime icon candidate for Qt startup."""
    root = icon_root or default_icon_root()
    candidates = (
        root / "png" / "icon_256.png",
        root / "png" / "app_icon.png",
        root / "app_icon.svg",
        root / "png" / "app_icon.svg",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="lfptensorpipe-icon-build")
    parser.add_argument("--icon-root", default=None)
    parser.add_argument("--source-image", default=None)
    parser.add_argument("--source-svg", default=None)
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    build_icon_assets_fn: Callable[..., dict[str, Path]] = build_icon_assets,
) -> int:
    args = parse_args(argv)
    icon_root = Path(args.icon_root).expanduser().resolve() if args.icon_root else None
    source_image = (
        Path(args.source_image).expanduser().resolve() if args.source_image else None
    )
    source_svg = (
        Path(args.source_svg).expanduser().resolve() if args.source_svg else None
    )
    outputs = build_icon_assets_fn(
        icon_root=icon_root,
        source_image=source_image,
        source_svg=source_svg,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint guard
    raise SystemExit(main())
