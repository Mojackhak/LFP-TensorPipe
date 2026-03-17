#!/usr/bin/env python3
"""Fail when core scientific modules import app or GUI layers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
CORE_DIRS = (
    ROOT / "src/lfptensorpipe/lfp",
    ROOT / "src/lfptensorpipe/preproc",
    ROOT / "src/lfptensorpipe/stats",
    ROOT / "src/lfptensorpipe/tabular",
    ROOT / "src/lfptensorpipe/utils",
    ROOT / "src/lfptensorpipe/io",
    ROOT / "src/lfptensorpipe/viz",
)

FORBIDDEN_PATTERNS = (
    re.compile(r"\bfrom\s+lfptensorpipe\.(app|gui)\b"),
    re.compile(r"\bimport\s+lfptensorpipe\.(app|gui)\b"),
)


@dataclass(frozen=True)
class Finding:
    path: Path
    line_number: int
    line: str


def _scan_file(path: Path) -> list[Finding]:
    findings: list[Finding] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(),
        start=1,
    ):
        if any(pattern.search(line) for pattern in FORBIDDEN_PATTERNS):
            findings.append(Finding(path=path, line_number=line_number, line=line.strip()))
    return findings


def main() -> int:
    findings: list[Finding] = []
    for base in CORE_DIRS:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            findings.extend(_scan_file(path))

    if not findings:
        print("Core purity passed.")
        return 0

    print("Core purity failed:\n", file=sys.stderr)
    for finding in findings:
        relative = finding.path.relative_to(ROOT)
        print(f"- {relative}:{finding.line_number}: {finding.line}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
