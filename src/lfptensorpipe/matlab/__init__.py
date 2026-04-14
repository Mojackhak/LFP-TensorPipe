"""MATLAB runtime bootstrap helpers for LFP-TensorPipe."""

from __future__ import annotations

from hashlib import sha1
import os
from pathlib import Path
import platform
import shutil
import sys
from typing import Any, Callable, Dict, Tuple

from lfptensorpipe.app.config_store import AppConfigStore

_ACTIVE_BRIDGE_KEY: str | None = None
_ACTIVE_SYS_PATHS: tuple[str, ...] = ()


def find_paths_config() -> Path:
    """Return the user-storage `paths.yml` location."""
    store = AppConfigStore()
    store.ensure_core_files()
    return store.path_for("paths.yml")


def load_paths_config() -> Tuple[Dict, Path]:
    """Load `paths.yml` and return `(config_dict, config_path)`."""
    cfg_path = find_paths_config()
    store = AppConfigStore()
    payload = store.read_yaml("paths.yml", default={})
    cfg = payload if isinstance(payload, dict) else {}
    return cfg, cfg_path


def require_path(cfg: Dict, key: str, cfg_path: Path) -> Path:
    """Fetch a required path from config dict and validate presence."""
    value = str(cfg.get(key, "")).strip()
    if not value:
        raise RuntimeError(f'Missing "{key}" in {cfg_path}')
    return Path(value).expanduser()


def matlab_arch_name() -> str:
    """Return MATLAB's platform-specific architecture token."""
    system = platform.system()
    if system == "Windows":
        return "win64"
    if system == "Linux":
        return "glnxa64"
    if system == "Darwin":
        return "maca64" if platform.machine() == "arm64" else "maci64"
    raise RuntimeError(f"Unsupported platform for MATLAB Engine: {system}")


def infer_matlab_root(candidate: Path) -> Path | None:
    """Infer a MATLAB installation root from a root or legacy engine path."""
    try:
        resolved = Path(candidate).expanduser().resolve()
    except OSError:
        return None
    search_roots = [resolved, *resolved.parents]
    for root in search_roots:
        if not root.is_dir():
            continue
        engine_source = root / "extern" / "engines" / "python"
        executable = matlab_executable_path(root)
        if engine_source.is_dir() and executable.is_file():
            return root
    return None


def resolve_matlab_root(candidate: Path) -> Path:
    """Resolve and validate a MATLAB installation root."""
    root = infer_matlab_root(candidate)
    if root is None:
        raise RuntimeError(
            "Invalid MATLAB installation path: "
            f"{Path(candidate).expanduser()}. Expected a MATLAB root that "
            "contains `extern/engines/python` and `bin/matlab`."
        )
    return root


def matlab_engine_source_dir(matlab_root: Path) -> Path:
    """Return MATLAB's bundled Python-engine source directory."""
    return matlab_root / "extern" / "engines" / "python"


def matlab_engine_dist_package_dir(matlab_root: Path) -> Path:
    """Return the package directory copied into the managed bridge cache."""
    return matlab_engine_source_dir(matlab_root) / "dist" / "matlab"


def matlab_engine_binary_dir(matlab_root: Path) -> Path:
    """Return MATLAB's original engine binary directory for the current arch."""
    return matlab_engine_dist_package_dir(matlab_root) / "engine" / matlab_arch_name()


def matlab_executable_path(matlab_root: Path) -> Path:
    """Return the MATLAB executable path for the current platform."""
    executable = "matlab.exe" if os.name == "nt" else "matlab"
    return matlab_root / "bin" / executable


def matlab_bin_dir(matlab_root: Path) -> Path:
    """Return MATLAB's `bin/<arch>` directory."""
    return matlab_root / "bin" / matlab_arch_name()


def matlab_extern_bin_dir(matlab_root: Path) -> Path:
    """Return MATLAB's `extern/bin/<arch>` directory."""
    return matlab_root / "extern" / "bin" / matlab_arch_name()


def _bridge_root_suffix(matlab_root: Path) -> str:
    stem = matlab_root.stem or matlab_root.name
    normalized = stem.replace("MATLAB_", "").replace(" ", "_")
    digest = sha1(str(matlab_root).encode("utf-8")).hexdigest()[:8]
    return f"{normalized}-{digest}"


def matlab_bridge_cache_dir(
    matlab_root: Path,
    *,
    config_store: AppConfigStore | None = None,
) -> Path:
    """Return the app-managed cache root for one MATLAB installation."""
    store = config_store or AppConfigStore()
    return store.cache_dir / "matlab_bridge" / _bridge_root_suffix(matlab_root)


def matlab_bridge_site_dir(
    matlab_root: Path,
    *,
    config_store: AppConfigStore | None = None,
) -> Path:
    """Return the app-managed site-packages directory for one MATLAB root."""
    return (
        matlab_bridge_cache_dir(matlab_root, config_store=config_store)
        / "site-packages"
    )


def matlab_bridge_package_dir(
    matlab_root: Path,
    *,
    config_store: AppConfigStore | None = None,
) -> Path:
    """Return the cached `matlab` package directory."""
    return matlab_bridge_site_dir(matlab_root, config_store=config_store) / "matlab"


def matlab_bridge_engine_dir(
    matlab_root: Path,
    *,
    config_store: AppConfigStore | None = None,
) -> Path:
    """Return the cached `matlab/engine/<arch>` binary directory."""
    return (
        matlab_bridge_package_dir(matlab_root, config_store=config_store)
        / "engine"
        / matlab_arch_name()
    )


def _arch_file_text(
    matlab_root: Path,
    *,
    config_store: AppConfigStore | None = None,
) -> str:
    return (
        "\n".join(
            (
                matlab_arch_name(),
                str(matlab_bin_dir(matlab_root)),
                str(matlab_engine_binary_dir(matlab_root)),
                str(matlab_extern_bin_dir(matlab_root)),
            )
        )
        + "\n"
    )


def prepare_managed_matlab_bridge(
    matlab_root: Path,
    *,
    config_store: AppConfigStore | None = None,
    refresh: bool = False,
    copytree_fn: Callable[..., Any] = shutil.copytree,
) -> Path:
    """Create or refresh the managed MATLAB bridge cache for one MATLAB root."""
    resolved_root = resolve_matlab_root(matlab_root)
    source_package = matlab_engine_dist_package_dir(resolved_root)
    if not source_package.is_dir():
        raise RuntimeError(
            "MATLAB installation is missing the Python engine package directory: "
            f"{source_package}"
        )

    site_dir = matlab_bridge_site_dir(resolved_root, config_store=config_store)
    package_dir = site_dir / "matlab"
    engine_dir = package_dir / "engine"
    if refresh and package_dir.exists():
        shutil.rmtree(package_dir)
    if not package_dir.exists():
        site_dir.mkdir(parents=True, exist_ok=True)
        copytree_fn(source_package, package_dir, dirs_exist_ok=True)

    engine_dir.mkdir(parents=True, exist_ok=True)
    arch_path = engine_dir / "_arch.txt"
    arch_path.write_text(
        _arch_file_text(resolved_root, config_store=config_store),
        encoding="utf-8",
    )
    return site_dir


def _remove_tracked_sys_paths() -> None:
    global _ACTIVE_SYS_PATHS
    if not _ACTIVE_SYS_PATHS:
        return
    sys.path[:] = [item for item in sys.path if item not in _ACTIVE_SYS_PATHS]
    _ACTIVE_SYS_PATHS = ()


def _purge_loaded_matlab_modules() -> None:
    for name in list(sys.modules):
        if name == "matlab" or name.startswith("matlab."):
            sys.modules.pop(name, None)


def _track_active_sys_paths(
    matlab_root: Path,
    site_dir: Path,
    *,
    config_store: AppConfigStore | None = None,
) -> None:
    global _ACTIVE_SYS_PATHS
    package_dir = matlab_bridge_package_dir(matlab_root, config_store=config_store)
    tracked = (
        str(site_dir),
        str(package_dir),
        str(matlab_engine_binary_dir(matlab_root)),
        str(matlab_extern_bin_dir(matlab_root)),
    )
    _ACTIVE_SYS_PATHS = tracked


def _import_matlab_engine() -> None:
    import matlab.engine  # noqa: F401


def ensure_matlab_engine(
    matlab_root: Path,
    *,
    import_matlab_engine_fn: Callable[[], None] = _import_matlab_engine,
    config_store: AppConfigStore | None = None,
    refresh: bool = False,
) -> Path:
    """Ensure `matlab.engine` is importable from the managed MATLAB bridge cache."""
    global _ACTIVE_BRIDGE_KEY

    resolved_root = resolve_matlab_root(matlab_root)
    site_dir = prepare_managed_matlab_bridge(
        resolved_root,
        config_store=config_store,
        refresh=refresh,
    )
    bridge_key = f"{resolved_root}::{site_dir}"

    if _ACTIVE_BRIDGE_KEY != bridge_key:
        _remove_tracked_sys_paths()
        _purge_loaded_matlab_modules()
        _ACTIVE_BRIDGE_KEY = bridge_key

    site_dir_str = str(site_dir)
    if site_dir_str in sys.path:
        sys.path.remove(site_dir_str)
    sys.path.insert(0, site_dir_str)
    _track_active_sys_paths(resolved_root, site_dir, config_store=config_store)

    try:
        import_matlab_engine_fn()
    except Exception:
        _purge_loaded_matlab_modules()
        try:
            import_matlab_engine_fn()
        except Exception as retry_exc:
            raise RuntimeError(
                "Failed to import MATLAB Engine from the selected MATLAB "
                f"installation: {resolved_root}"
            ) from retry_exc
        return site_dir
    return site_dir


def is_managed_matlab_bridge_ready(
    matlab_root: Path,
    *,
    config_store: AppConfigStore | None = None,
) -> bool:
    """Return whether the managed cache already contains the required files."""
    try:
        resolved_root = resolve_matlab_root(matlab_root)
    except RuntimeError:
        return False
    package_dir = matlab_bridge_package_dir(resolved_root, config_store=config_store)
    return (
        (package_dir / "__init__.py").is_file()
        and (package_dir / "engine" / "__init__.py").is_file()
        and (
            matlab_engine_binary_dir(resolved_root)
            / "matlabengineforpython_abi3.abi3.so"
        ).is_file()
        and (package_dir / "engine" / "_arch.txt").is_file()
    )


__all__ = [
    "ensure_matlab_engine",
    "find_paths_config",
    "infer_matlab_root",
    "is_managed_matlab_bridge_ready",
    "load_paths_config",
    "matlab_arch_name",
    "matlab_bin_dir",
    "matlab_bridge_cache_dir",
    "matlab_bridge_engine_dir",
    "matlab_bridge_package_dir",
    "matlab_bridge_site_dir",
    "matlab_engine_binary_dir",
    "matlab_engine_dist_package_dir",
    "matlab_engine_source_dir",
    "matlab_executable_path",
    "matlab_extern_bin_dir",
    "prepare_managed_matlab_bridge",
    "require_path",
    "resolve_matlab_root",
]
