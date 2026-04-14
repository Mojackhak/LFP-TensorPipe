from __future__ import annotations

import argparse
from pathlib import Path
import sys

_FILE = Path(__file__).resolve()
_REPO_ROOT = _FILE.parents[4]
_SRC_ROOT = _REPO_ROOT / "src"

for _path in (_REPO_ROOT, _SRC_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from paper.pd.examples.med.workflow import EXAMPLE_SUBJECT, export_subject_burst_example


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the representative Sub-014 med burst example pickle."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Optional PD project root. Defaults to paper.pd.specs.DEFAULT_PROJECT_ROOT.",
    )
    parser.add_argument(
        "--subject",
        default=EXAMPLE_SUBJECT,
        help="Subject identifier to export. Defaults to Sub-014.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    out_path = export_subject_burst_example(
        project_root=args.project_root,
        subject=str(args.subject),
    )
    print(out_path)


if __name__ == "__main__":
    main()

