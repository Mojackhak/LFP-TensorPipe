"""Tests for the managed MATLAB bridge bootstrap runtime."""

from __future__ import annotations

from pathlib import Path
import sys

from lfptensorpipe.app.config_store import AppConfigStore
from lfptensorpipe.matlab import (
    ensure_matlab_engine,
    infer_matlab_root,
    matlab_arch_name,
    matlab_engine_binary_dir,
    prepare_managed_matlab_bridge,
    resolve_matlab_root,
)


def _fake_matlab_root(tmp_path: Path) -> Path:
    matlab_root = tmp_path / "MATLAB_R2024b.app"
    (matlab_root / "bin").mkdir(parents=True)
    (matlab_root / "bin" / "matlab").write_text("", encoding="utf-8")
    dist_engine_dir = (
        matlab_root / "extern" / "engines" / "python" / "dist" / "matlab" / "engine"
    )
    dist_engine_dir.mkdir(parents=True)
    (
        matlab_root
        / "extern"
        / "engines"
        / "python"
        / "dist"
        / "matlab"
        / "__init__.py"
    ).write_text(
        "from pkgutil import extend_path\n__path__ = extend_path(__path__, '__name__')\n",
        encoding="utf-8",
    )
    (dist_engine_dir / "__init__.py").write_text(
        "ENGINE_IMPORTED = True\n",
        encoding="utf-8",
    )
    (dist_engine_dir / matlab_arch_name()).mkdir(parents=True)
    (
        dist_engine_dir / matlab_arch_name() / "matlabengineforpython_abi3.abi3.so"
    ).write_text(
        "",
        encoding="utf-8",
    )
    (matlab_root / "extern" / "bin" / matlab_arch_name()).mkdir(parents=True)
    (matlab_root / "bin" / matlab_arch_name()).mkdir(parents=True)
    return matlab_root


def test_infer_matlab_root_accepts_root_and_legacy_engine_path(
    tmp_path: Path,
) -> None:
    matlab_root = _fake_matlab_root(tmp_path)
    legacy_engine_path = matlab_root / "extern" / "engines" / "python"

    assert infer_matlab_root(matlab_root) == matlab_root.resolve()
    assert infer_matlab_root(legacy_engine_path) == matlab_root.resolve()
    assert resolve_matlab_root(legacy_engine_path) == matlab_root.resolve()


def test_prepare_managed_matlab_bridge_copies_package_and_writes_arch_file(
    tmp_path: Path,
) -> None:
    matlab_root = _fake_matlab_root(tmp_path)
    store = AppConfigStore(repo_root=tmp_path / "app")
    site_dir = prepare_managed_matlab_bridge(matlab_root, config_store=store)

    assert (site_dir / "matlab" / "__init__.py").is_file()
    assert (site_dir / "matlab" / "engine" / "__init__.py").is_file()
    arch_path = site_dir / "matlab" / "engine" / "_arch.txt"
    assert arch_path.is_file()
    arch_text = arch_path.read_text(encoding="utf-8")
    assert str(matlab_engine_binary_dir(matlab_root.resolve())) in arch_text
    assert str(matlab_root / "extern" / "bin" / matlab_arch_name()) in arch_text


def test_ensure_matlab_engine_uses_managed_bridge_cache(
    tmp_path: Path,
) -> None:
    matlab_root = _fake_matlab_root(tmp_path)
    store = AppConfigStore(repo_root=tmp_path / "app")
    calls: list[str] = []

    def _fake_import() -> None:
        calls.append(sys.path[0])

    site_dir = ensure_matlab_engine(
        matlab_root,
        import_matlab_engine_fn=_fake_import,
        config_store=store,
    )

    assert calls == [str(site_dir)]
    assert sys.path[0] == str(site_dir)
