"""Console interface for one exported page configuration."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from lfptensorpipe.app.shared.page_config import PAGE_SCHEMAS, page_config_node


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="lfptp run")
    parser.add_argument(
        "--subject-path",
        required=True,
        type=Path,
        help="Existing <project>/derivatives/lfptensorpipe/<subject> directory.",
    )
    parser.add_argument(
        "--record", required=True, help="Existing imported record name."
    )
    parser.add_argument(
        "--config", required=True, type=Path, help="One page-exported .json file."
    )
    parser.add_argument(
        "--trial", help="Exact trial slug; required for Align and Features."
    )
    return parser.parse_args(argv)


def load_request(args: argparse.Namespace):
    from lfptensorpipe.app.dataset.validation import (
        validate_record_name,
        validate_subject_name,
    )
    from lfptensorpipe.app.path_resolver import RecordContext

    subject_path = args.subject_path.expanduser().resolve()
    if (
        not subject_path.is_dir()
        or subject_path.parent.name != "lfptensorpipe"
        or subject_path.parent.parent.name != "derivatives"
    ):
        raise ValueError(
            "--subject-path must be an existing <project>/derivatives/lfptensorpipe/<subject> directory."
        )
    for value, validator in (
        (subject_path.name, validate_subject_name),
        (args.record, validate_record_name),
    ):
        ok, message = validator(value)
        if not ok:
            raise ValueError(message)
    context = RecordContext(
        subject_path.parents[2], subject_path.name, args.record.strip()
    )
    record_path = subject_path / context.record
    if (
        not record_path.is_dir()
        or not (record_path / "preproc" / "raw" / "raw.fif").is_file()
    ):
        raise ValueError(f"Imported record not found: {record_path}")
    if record_path.resolve().parent != subject_path:
        raise ValueError(
            "Record path must remain inside the selected subject directory."
        )
    config_path = args.config.expanduser().resolve()
    if config_path.suffix.lower() != ".json":
        raise ValueError("--config must be a page-exported .json file.")
    with config_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    schema = payload.get("schema") if isinstance(payload, dict) else None
    page = next(
        (key for key, (name, _) in PAGE_SCHEMAS.items() if name == schema), None
    )
    if page is None:
        raise ValueError(f"Unsupported page config schema: {schema!r}.")
    node = page_config_node(payload, page)
    if page in {"alignment", "features"}:
        if not args.trial:
            raise ValueError(f"--trial is required for {page}.")
        from lfptensorpipe.app.alignment.param_normalizers import _normalize_slug

        if args.trial != _normalize_slug(args.trial):
            raise ValueError(
                "--trial must be an exact canonical trial slug (lowercase letters, digits, and hyphens)."
            )
        trial_path = record_path / "alignment" / args.trial
        if trial_path.resolve().parent != (record_path / "alignment").resolve():
            raise ValueError("Trial path must remain inside the selected record.")
    elif args.trial is not None:
        raise ValueError(f"--trial is not accepted for {page}.")
    return context, page, node, payload["version"]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        context, page, node, version = load_request(args)
    except (OSError, ValueError) as exc:
        print(f"Invalid input: {exc}", file=sys.stderr)
        return 2
    os.environ["MPLBACKEND"] = "Agg"
    from lfptensorpipe.app.cli import ConfigError, RunFailed, run_page

    try:
        message = run_page(
            context, page=page, node=node, version=version, trial=args.trial
        )
    except ConfigError as exc:
        print(f"Invalid configuration: {exc}", file=sys.stderr)
        return 2
    except (OSError, RunFailed) as exc:
        print(f"{page} failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(f"{page} interrupted.", file=sys.stderr)
        return 130
    from lfptensorpipe.app.path_resolver import PathResolver

    scope = f"{context.subject}/{context.record}" + (
        f"/{args.trial}" if args.trial else ""
    )
    print(
        f"{page} completed: {scope}\n{message}\nResults: {PathResolver(context).lfp_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
