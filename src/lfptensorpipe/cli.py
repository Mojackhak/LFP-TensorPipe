"""Console interface for one exported page configuration."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from lfptensorpipe.app.shared.page_config import PAGE_SCHEMAS, page_config_node


def target_arguments(parser):
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
        "--trial", help="Exact trial slug; required for Align and Features."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="lfptp run")
    target_arguments(parser)
    parser.add_argument(
        "--config", required=True, type=Path, help="Page or variant plan JSON."
    )
    parser.add_argument(
        "--output-root", type=Path, help="Output directory for a named variant plan."
    )
    return parser.parse_args(argv)


def record_context(args: argparse.Namespace):
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
    if not record_path.is_dir():
        raise ValueError(f"Imported record not found: {record_path}")
    if record_path.resolve().parent != subject_path:
        raise ValueError(
            "Record path must remain inside the selected subject directory."
        )
    return context


def read_page(config_path):
    with config_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("The JSON configuration must be an object.")
    schema = payload.get("schema")
    if schema is None and {"base", "variants"} <= payload.keys():
        return "plan", payload, None
    if schema == "lfptensorpipe.preprocess-config":
        page = "preprocess"
        node = payload.get(page)
        if not isinstance(node, dict):
            raise ValueError("Expected a preprocess object.")
    else:
        page = next(
            (key for key, (name, _) in PAGE_SCHEMAS.items() if name == schema), None
        )
        if page is None:
            raise ValueError(f"Unsupported page config schema: {schema!r}.")
        node = page_config_node(payload, page)
    resolve_references(page, node, config_path.parent)
    return page, node, payload.get("version")


def resolve_references(page, node, parent):
    if page == "preprocess":
        owner, key = node.get("annotations"), "csv"
    elif page == "localize":
        owner, key = node, "labels_csv"
    elif page == "alignment":
        owner, key = node, "epoch_selection"
    else:
        return
    if (
        isinstance(owner, dict)
        and isinstance(owner.get(key), str)
        and not (key == "epoch_selection" and owner[key] == "accepted")
    ):
        owner[key] = str((parent / Path(owner[key]).expanduser()).resolve())


def check_trial(context, page, trial):
    from lfptensorpipe.app.path_resolver import PathResolver

    record_path = PathResolver(context).lfp_root
    if page in {"alignment", "features", "epoch-selection", "plan"}:
        if not trial:
            raise ValueError(f"--trial is required for {page}.")
        from lfptensorpipe.app.alignment.param_normalizers import _normalize_slug

        if trial != _normalize_slug(trial):
            raise ValueError(
                "--trial must be an exact canonical trial slug (lowercase letters, digits, and hyphens)."
            )
        trial_path = record_path / "alignment" / trial
        if trial_path.resolve().parent != (record_path / "alignment").resolve():
            raise ValueError("Trial path must remain inside the selected record.")
    elif trial is not None:
        raise ValueError(f"--trial is not accepted for {page}.")


def load_request(args: argparse.Namespace):
    from lfptensorpipe.app.preproc.paths import rawdata_input_fif_path
    from lfptensorpipe.app.path_resolver import PathResolver

    context = record_context(args)
    path = args.config.expanduser().resolve()
    if path.suffix.lower() != ".json":
        raise ValueError("--config must be a JSON file.")
    page, node, version = read_page(path)
    if (page == "plan") != (getattr(args, "output_root", None) is not None):
        raise ValueError("Specify --output-root for a variant plan only.")
    if not (PathResolver(context).preproc_root / "raw" / "raw.fif").is_file() and (
        page not in {"localize", "preprocess"}
        or not rawdata_input_fif_path(context).is_file()
    ):
        raise ValueError(f"Imported record not found: {context.record}")
    check_trial(context, page, args.trial)
    return context, page, node, version


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
        if page == "plan":
            from lfptensorpipe.app.cli_variants import run_plan

            results = run_plan(
                context,
                args.trial,
                node,
                args.config.resolve().parent,
                args.output_root.expanduser().resolve(),
            )
            print("\n".join(str(path) for path in results))
            return 0
        message = run_page(
            context, page=page, node=node, version=version, trial=args.trial
        )
    except (ConfigError, ValueError) as exc:
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


def analysis_command(command, argv):
    parser = argparse.ArgumentParser(prog=f"lfptp {command}")
    parser.add_argument("--output", required=True, type=Path)
    target_arguments(parser)
    parser.add_argument(
        "--page",
        required=True,
        choices=("preprocess", "epoch-selection", "alignment", "features"),
    )
    args = parser.parse_args(argv)
    from .app.cli import RunFailed

    try:
        from .app.cli_export import export_page
        from .io.atomic_file import write_json_atomically

        context = record_context(args)
        check_trial(context, args.page, args.trial)
        if args.output.exists():
            raise ValueError("Choose a new export file.")
        payload = export_page(context, args.page, args.trial)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomically(args.output, payload)
    except ValueError as exc:
        print(f"Invalid input: {exc}", file=sys.stderr)
        return 2
    except (OSError, RunFailed) as exc:
        print(f"{command} failed: {exc}", file=sys.stderr)
        return 1
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
