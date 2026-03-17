#!/usr/bin/env python3
"""Fail when app/gui structure guardrails are exceeded."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

HARD_LIMITS = {
    ROOT / "src/lfptensorpipe/gui/shell/main_window_logic.py": 500,
}

THIN_ENTRY_LIMIT = 80
SERVICE_LIMIT = 600
HELPER_WARN_LIMIT = 350


def line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def iter_service_files() -> list[Path]:
    return sorted((ROOT / "src/lfptensorpipe/app").glob("**/service.py"))


def iter_thin_entry_files() -> list[Path]:
    app_root = ROOT / "src/lfptensorpipe/app"
    files = [ROOT / "src/lfptensorpipe/gui/main_window.py"]
    patterns = [
        "*_service.py",
        "*_store.py",
        "*_index.py",
        "*_resolver.py",
        "localize_viewer_worker.py",
    ]
    for pattern in patterns:
        files.extend(sorted(app_root.glob(pattern)))
    return sorted(dict.fromkeys(files))


def iter_helper_files() -> list[Path]:
    roots = [
        ROOT / "src/lfptensorpipe/gui",
        ROOT / "src/lfptensorpipe/app",
    ]
    files: list[Path] = []
    for base in roots:
        for path in base.glob("**/*.py"):
            if path.name in {"main_window.py", "service.py", "__init__.py"}:
                continue
            files.append(path)
    return sorted(files)


def main() -> int:
    failures: list[str] = []
    warnings: list[str] = []

    for path, limit in HARD_LIMITS.items():
        count = line_count(path)
        if count > limit:
            failures.append(f"{path.relative_to(ROOT)}: {count} > {limit}")

    for path in iter_thin_entry_files():
        count = line_count(path)
        if count > THIN_ENTRY_LIMIT:
            failures.append(f"{path.relative_to(ROOT)}: {count} > {THIN_ENTRY_LIMIT}")

    for path in iter_service_files():
        count = line_count(path)
        if count > SERVICE_LIMIT:
            failures.append(f"{path.relative_to(ROOT)}: {count} > {SERVICE_LIMIT}")

    for path in iter_helper_files():
        count = line_count(path)
        if count > HELPER_WARN_LIMIT:
            warnings.append(
                f"{path.relative_to(ROOT)}: {count} > advisory {HELPER_WARN_LIMIT}"
            )

    if warnings:
        print("Advisory helper-module size warnings:")
        for item in warnings:
            print(f"  - {item}")

    if failures:
        print("Size guard failures:")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("Size guards passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
