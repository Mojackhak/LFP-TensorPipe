#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "module alias via sys.modules",
        re.compile(r"\bsys\.modules\[\s*__name__\s*\]\s*="),
    ),
    (
        "dynamic globals copy",
        re.compile(r"\bglobals\(\)\.update\("),
    ),
    (
        "joblib callback monkeypatch",
        re.compile(r"\bjoblib\.parallel\.BatchCompletionCallBack\s*="),
    ),
    (
        "imported stage-module constant mutation",
        re.compile(r"\b_stage_[a-z_]+_panel\.[A-Z_]+\s*="),
    ),
)


@dataclass(frozen=True)
class Finding:
    path: Path
    line_number: int
    label: str
    line: str


def _load_allowlist(path: Path | None) -> set[str]:
    if path is None:
        return set()
    if not path.exists():
        print(f"Allowlist file not found: {path}", file=sys.stderr)
        raise SystemExit(2)

    entries: set[str] = set()
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        entry = raw.strip()
        if not entry or entry.startswith("#"):
            continue
        entries.add(entry)
    return entries


def _scan_file(path: Path, repo_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(),
        start=1,
    ):
        for label, pattern in PATTERNS:
            if pattern.search(line):
                findings.append(
                    Finding(
                        path=path.relative_to(repo_root),
                        line_number=line_number,
                        label=label,
                        line=line.strip(),
                    )
                )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail when production code introduces known monkeypatch-style patterns."
    )
    parser.add_argument(
        "--root",
        default="src/lfptensorpipe",
        help="Root package directory to scan (default: src/lfptensorpipe).",
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=Path("tools/monkeypatch_allowlist.txt"),
        help="Relative-path allowlist for temporarily deferred files.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"Monkeypatch guard skipped: {root} does not exist.")
        return 0

    repo_root = Path.cwd().resolve()
    allowlist = _load_allowlist(args.allowlist)
    findings: list[Finding] = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(repo_root).as_posix()
        if relative in allowlist:
            continue
        findings.extend(_scan_file(path, repo_root))

    if not findings:
        print("Monkeypatch guard passed.")
        return 0

    print("Monkeypatch guard failed:\n", file=sys.stderr)
    for finding in findings:
        print(
            f"- {finding.path}:{finding.line_number}: {finding.label}\n"
            f"  {finding.line}",
            file=sys.stderr,
        )

    print(
        "\nAction: replace runtime mutation with static imports, explicit adapters, "
        "or dependency injection. If a file is an approved temporary exception, "
        "add its relative path to tools/monkeypatch_allowlist.txt with a comment.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
