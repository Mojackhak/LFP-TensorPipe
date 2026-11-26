"""Export accepted decisions independently of current GUI drafts."""

from .cli import RunFailed
from .cli_epochs import accepted_selection
from .path_resolver import PathResolver
from .shared.page_config import PAGE_SCHEMAS
from .shared.software_info import software_identity


def export_page(context, page, trial=None):
    if page == "preprocess":
        from .cli_preprocess import export_preprocess

        return {
            "schema": "lfptensorpipe.preprocess-config",
            "software": software_identity(),
            page: export_preprocess(context),
        }
    if page in ("epoch-selection", "alignment"):
        from .alignment.generation import latest_alignment_step_entry
        from .shared.runlog_store import run_log_read_snapshot

        with run_log_read_snapshot():
            selection = accepted_selection(context, trial)
            if page == "epoch-selection":
                return {**selection, "software": software_identity()}
            entry = latest_alignment_step_entry(
                PathResolver(context), trial_slug=trial, step="run_align_epochs"
            )[1]["params"]
        config = entry.get("trial_config", entry)
        node = {k: config[k] for k in ("method", "method_params")}
        node["epoch_selection"] = selection
    elif page == "features":
        from .features.execution import execution_snapshot
        from .features.indicator import (
            extract_features_indicator_state,
            features_derivatives_log_path,
        )
        from .shared.runlog_store import read_run_log

        resolver = PathResolver(context)
        if extract_features_indicator_state(resolver, trial_slug=trial) != "green":
            raise RunFailed("No current extracted Features to export.")
        params = read_run_log(
            features_derivatives_log_path(resolver, trial_slug=trial)
        )["params"]
        node = {
            "active_metric": params["metrics"][0],
            "axes_by_metric": params["axes_by_metric"],
            "execution": execution_snapshot(
                context, trial, params["metrics"], None, accepted=params
            ),
        }
    else:
        raise ValueError(f"Unsupported export: {page}")
    schema, versions = PAGE_SCHEMAS[page]
    return {
        "schema": schema,
        "version": versions[-1],
        "software": software_identity(),
        page: node,
    }
