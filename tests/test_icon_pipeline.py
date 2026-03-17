"""Tests for icon build and runtime helpers."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

import pytest

from lfptensorpipe.gui import icon_pipeline
from lfptensorpipe.gui.icon_pipeline import (
    _build_macos_icns,
    _build_png_set,
    _build_windows_ico,
    _canonicalize_source,
    _resolve_source_asset,
    build_icon_assets,
    default_icon_root,
    main,
    parse_args,
    preferred_runtime_icon_path,
)


def _write_png(path: Path, size: int = 32) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (size, size), color=(255, 0, 0, 255)).save(path)


def test_default_icon_root_is_assets_icons_path() -> None:
    root = default_icon_root()
    assert root.name == "icons"
    assert root.parent.name == "assets"


def test_preferred_runtime_icon_path_prefers_generated_png(tmp_path: Path) -> None:
    icon_root = tmp_path / "icons"
    png_dir = icon_root / "png"
    png_dir.mkdir(parents=True)
    target = png_dir / "icon_256.png"
    target.write_bytes(b"png")

    resolved = preferred_runtime_icon_path(icon_root)
    assert resolved == target


def test_preferred_runtime_icon_path_returns_none_when_missing(tmp_path: Path) -> None:
    resolved = preferred_runtime_icon_path(tmp_path / "icons_missing")
    assert resolved is None


def test_resolve_source_asset_prefers_png_master_by_default(tmp_path: Path) -> None:
    icon_root = tmp_path / "icons"
    png_dir = icon_root / "png"
    png_dir.mkdir(parents=True)
    png_source = png_dir / "app_icon.png"
    svg_source = icon_root / "app_icon.svg"
    png_source.write_bytes(b"png")
    svg_source.write_text("<svg/>", encoding="utf-8")

    resolved = _resolve_source_asset(icon_root, source_image=None, source_svg=None)
    assert resolved == png_source


def test_resolve_source_asset_uses_explicit_source_image(tmp_path: Path) -> None:
    source = tmp_path / "custom.png"
    source.write_bytes(b"png")

    resolved = _resolve_source_asset(
        tmp_path / "unused",
        source_image=source,
        source_svg=None,
    )
    assert resolved == source


def test_resolve_source_asset_uses_explicit_source_svg(tmp_path: Path) -> None:
    source = tmp_path / "custom.svg"
    source.write_text("<svg/>", encoding="utf-8")

    resolved = _resolve_source_asset(
        tmp_path / "unused",
        source_image=None,
        source_svg=source,
    )
    assert resolved == source


@pytest.mark.parametrize(
    ("arg_name", "suffix", "error_match"),
    [
        ("source_image", ".png", "Source image not found"),
        ("source_svg", ".svg", "Source SVG not found"),
    ],
)
def test_resolve_source_asset_raises_for_missing_explicit_source(
    tmp_path: Path,
    arg_name: str,
    suffix: str,
    error_match: str,
) -> None:
    missing = tmp_path / f"missing{suffix}"
    kwargs: dict[str, Path | None] = {"source_image": None, "source_svg": None}
    kwargs[arg_name] = missing

    with pytest.raises(FileNotFoundError, match=error_match):
        _resolve_source_asset(tmp_path / "icons", **kwargs)


def test_resolve_source_asset_fallback_prefers_root_png_over_svg(
    tmp_path: Path,
) -> None:
    icon_root = tmp_path / "icons"
    icon_root.mkdir(parents=True)
    root_png = icon_root / "app_icon.png"
    root_svg = icon_root / "app_icon.svg"
    root_png.write_bytes(b"png")
    root_svg.write_text("<svg/>", encoding="utf-8")

    resolved = _resolve_source_asset(icon_root, source_image=None, source_svg=None)
    assert resolved == root_png


def test_resolve_source_asset_raises_when_no_default_source_exists(
    tmp_path: Path,
) -> None:
    icon_root = tmp_path / "icons"
    icon_root.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="Icon source not found"):
        _resolve_source_asset(icon_root, source_image=None, source_svg=None)


def test_canonicalize_source_copies_png_to_master(tmp_path: Path) -> None:
    icon_root = tmp_path / "icons"
    source = tmp_path / "input" / "from_ui.jpg"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"jpg")

    canonical = _canonicalize_source(icon_root, source)
    assert canonical == icon_root / "png" / "app_icon.png"
    assert canonical.read_bytes() == b"jpg"


def test_canonicalize_source_copies_svg_to_master(tmp_path: Path) -> None:
    icon_root = tmp_path / "icons"
    source = tmp_path / "input" / "icon.svg"
    source.parent.mkdir(parents=True)
    source.write_text("<svg/>", encoding="utf-8")

    canonical = _canonicalize_source(icon_root, source)
    assert canonical == icon_root / "app_icon.svg"
    assert canonical.read_text(encoding="utf-8") == "<svg/>"


def test_build_png_set_from_raster_generates_all_sizes(tmp_path: Path) -> None:
    icon_root = tmp_path / "icons"
    source = tmp_path / "app_icon.png"
    _write_png(source, size=64)

    outputs = _build_png_set(icon_root, source)
    assert set(outputs) == set(icon_pipeline.PNG_SIZES)
    for size, output_path in outputs.items():
        assert output_path == icon_root / "png" / f"icon_{size}.png"
        assert output_path.exists()


def test_build_png_set_from_svg_invokes_sips_for_each_size(
    tmp_path: Path,
) -> None:
    icon_root = tmp_path / "icons"
    source = tmp_path / "app_icon.svg"
    source.write_text("<svg/>", encoding="utf-8")
    calls: list[list[str]] = []

    def _fake_run(
        cmd: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert check is True
        assert capture_output is True
        assert text is True
        out_path = Path(cmd[cmd.index("--out") + 1])
        out_path.write_bytes(b"png")
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    outputs = _build_png_set(icon_root, source, svg_runner=_fake_run)

    assert set(outputs) == set(icon_pipeline.PNG_SIZES)
    assert len(calls) == len(icon_pipeline.PNG_SIZES)
    assert all(call[0] == "sips" for call in calls)


def test_build_windows_ico_generates_ico_from_png(tmp_path: Path) -> None:
    icon_root = tmp_path / "icons"
    png256 = tmp_path / "icon_256.png"
    _write_png(png256, size=256)

    ico_path = _build_windows_ico(icon_root, {256: png256})
    assert ico_path.exists()
    assert ico_path.suffix == ".ico"


def test_build_windows_ico_raises_when_png_unreadable(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Failed to load PNG"):
        _build_windows_ico(tmp_path / "icons", {256: tmp_path / "missing.png"})


def test_build_windows_ico_raises_when_qimage_save_fails(
    tmp_path: Path,
) -> None:
    class _FailSaveImage:
        def __init__(self, path: str) -> None:
            _ = path

        def isNull(self) -> bool:
            return False

        def save(self, path: str, fmt: str) -> bool:
            _ = (path, fmt)
            return False

    _write_png(tmp_path / "icon_256.png", size=256)

    with pytest.raises(RuntimeError, match="Failed to write ICO file"):
        _build_windows_ico(
            tmp_path / "icons",
            {256: tmp_path / "icon_256.png"},
            image_cls=_FailSaveImage,
        )


def test_build_macos_icns_runs_iconutil(tmp_path: Path) -> None:
    icon_root = tmp_path / "icons"
    png_outputs: dict[int, Path] = {}
    for size in set(icon_pipeline.ICONSET_MAP.values()):
        png = tmp_path / f"icon_{size}.png"
        _write_png(png, size=size)
        png_outputs[size] = png

    calls: list[list[str]] = []

    def _fake_run(
        cmd: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert check is True
        assert capture_output is True
        assert text is True
        out_path = Path(cmd[cmd.index("-o") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"icns")
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    icns_path = _build_macos_icns(icon_root, png_outputs, runner=_fake_run)

    assert icns_path.exists()
    assert icns_path.suffix == ".icns"
    assert len(calls) == 1
    assert calls[0][0] == "iconutil"


def test_build_icon_assets_orchestrates_all_sub_steps(
    tmp_path: Path,
) -> None:
    icon_root = tmp_path / "icons"
    source = tmp_path / "source.png"
    source.write_bytes(b"png")
    canonical = icon_root / "png" / "app_icon.png"
    png_outputs = {256: icon_root / "png" / "icon_256.png"}
    ico = icon_root / "windows" / "lfptensorpipe.ico"
    icns = icon_root / "macos" / "lfptensorpipe.icns"

    outputs = build_icon_assets(
        icon_root=icon_root,
        resolve_source_asset_fn=lambda *_: source,
        canonicalize_source_fn=lambda *_: canonical,
        build_png_set_fn=lambda *_args, **_kwargs: png_outputs,
        build_windows_ico_fn=lambda *_args, **_kwargs: ico,
        build_macos_icns_fn=lambda *_args, **_kwargs: icns,
    )
    assert outputs == {
        "source_asset": canonical,
        "png_dir": icon_root / "png",
        "windows_ico": ico,
        "macos_icns": icns,
    }


def test_parse_args_reads_cli_options() -> None:
    args = parse_args(
        [
            "--icon-root",
            "/tmp/icons",
            "--source-image",
            "/tmp/a.png",
            "--source-svg",
            "/tmp/a.svg",
        ]
    )
    assert args.icon_root == "/tmp/icons"
    assert args.source_image == "/tmp/a.png"
    assert args.source_svg == "/tmp/a.svg"


def test_main_prints_generated_output_map(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    outputs: dict[str, Any] = {
        "source_asset": tmp_path / "source.png",
        "png_dir": tmp_path / "png",
        "windows_ico": tmp_path / "windows.ico",
        "macos_icns": tmp_path / "mac.icns",
    }
    exit_code = main(
        ["--icon-root", str(tmp_path)],
        build_icon_assets_fn=lambda **_: outputs,
    )
    printed = capsys.readouterr().out

    assert exit_code == 0
    for key, value in outputs.items():
        assert f"{key}: {value}" in printed
