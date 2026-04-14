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

from paper.pd.examples.med.workflow import EXAMPLE_SUBJECT, run_subject_burst_example_viz


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render representative Sub-014 med burst example PDFs."
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
        help="Subject identifier used in the exported example table. Defaults to Sub-014.",
    )
    parser.add_argument(
        "--source-path",
        type=Path,
        default=None,
        help="Optional exported example pickle path. Defaults to the standard Sub-014 output.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    outputs = run_subject_burst_example_viz(
        project_root=args.project_root,
        subject=str(args.subject),
        source_path=args.source_path,
    )
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
