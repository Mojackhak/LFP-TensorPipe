"""Run or export the desktop preprocessing decisions for one record."""

from pathlib import Path

from .cli import ConfigError, RunFailed, _checked_result
from .path_resolver import PathResolver
from .preproc import service as svc
from .preproc.lineage import (
    capture_preproc_input_generation,
    preproc_step_is_skipped,
    preproc_step_lineage_is_current,
)
from .preproc.paths import (
    preproc_step_raw_path,
    preproc_step_log_path,
    preproc_filter_preview_raw_path,
    write_preproc_step_routing,
)
from .preproc.review_edits import apply_edits, current_state
from .shared.generation_lineage import accepted_result_generation_id
from .shared.runlog_store import read_run_log

STEPS = ("signal_repair", "filter", "ecg_artifact_removal", "annotations")


def step_entry(resolver, step):
    return read_run_log(preproc_step_log_path(resolver, step, create=False)) or {}


def annotation_rows(path):
    ok, rows, message = svc.load_annotations_csv_rows(Path(path))
    if not ok:
        raise ConfigError(message)
    return rows


def _normalized_annotation_rows(rows):
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ConfigError("Annotations rows must be a list of objects.")
    normalized, invalid = svc._normalize_annotation_rows(rows)
    if invalid:
        raise ConfigError(f"Invalid annotation rows: {invalid}")
    return normalized


def recipe_steps(recipe):
    """Resolve paths, optional routing and numerical defaults at the recipe boundary."""
    from lfptensorpipe.preproc.signal_repair import normalize_signal_repair_params

    if not isinstance(recipe, dict) or set(recipe) != set(STEPS):
        raise ConfigError("Specify all four optional preprocessing steps.")
    result = {}
    allowed = {
        "signal_repair": {"settings", "decisions"},
        "filter": {"low_freq", "high_freq", "notches", "advance", "review"},
        "ecg_artifact_removal": {
            "method",
            "picks",
            "method_kwargs",
            "mark_filter_edges",
        },
        "annotations": {"rows", "csv", "mark_filter_edges"},
    }
    for step in STEPS:
        spec = recipe[step]
        if (
            not isinstance(spec, dict)
            or spec.get("action") not in ("apply", "skip")
            or set(spec)
            - allowed[step]
            - ({"action"} if step == "filter" else {"action", "review_edits"})
        ):
            raise ConfigError(f"Invalid fields or action for {step}.")
        if spec["action"] == "skip":
            result[step] = {"action": "skip"}
            continue
        spec = dict(spec)
        if step == "signal_repair":
            spec["settings"] = normalize_signal_repair_params(spec.get("settings", {}))
        elif step == "filter":
            ok, advance, message = svc.normalize_filter_advance_params(
                spec.get("advance")
            )
            if not ok:
                raise ConfigError(message)
            from lfptensorpipe.preproc.notch import effective_notch_model
            from .preproc.steps.filter import normalize_filter_runtime_params

            ok, basic, message = normalize_filter_runtime_params(
                notches=spec.get("notches"),
                l_freq=spec.get("low_freq"),
                h_freq=spec.get("high_freq"),
                notch_model=effective_notch_model(advance["notch_model"]),
            )
            if not ok:
                raise ConfigError(message)
            spec.update(
                advance=advance,
                low_freq=basic["l_freq"],
                high_freq=basic["h_freq"],
                notches=basic["notches"],
            )
            if "review" not in spec or not (
                spec["review"] == "detected" or isinstance(spec["review"], (list, dict))
            ):
                raise ConfigError(
                    "Filter requires detected or an exported review change list."
                )
        elif step == "ecg_artifact_removal":
            method = spec.get("method", "svd")
            ok, kwargs, message = svc.normalize_ecg_method_params(
                method, spec.get("method_kwargs")
            )
            if not ok:
                raise ConfigError(message)
            spec.update(
                method=method,
                method_kwargs=kwargs,
                mark_filter_edges=spec.get("mark_filter_edges", False),
            )
            if (
                "picks" in spec
                and spec["picks"] is not None
                and (
                    not isinstance(spec["picks"], list)
                    or not all(isinstance(ch, str) for ch in spec["picks"])
                )
            ):
                raise ConfigError("ECG picks must contain channel names.")
        elif step == "annotations":
            spec.setdefault("mark_filter_edges", False)
            if ("csv" in spec) == ("rows" in spec):
                raise ConfigError("Annotations requires either rows or a CSV.")
            if "csv" in spec:
                spec["rows"] = annotation_rows(spec.pop("csv"))
            spec["rows"] = _normalized_annotation_rows(spec["rows"])
        result[step] = spec
    return result


def _review_inputs(steps, step):
    result = {}
    for key in STEPS:
        spec = dict(steps[key])
        if key == step:
            spec.pop("review", None)
            spec.pop("review_edits", None)
        result[key] = spec
        if key == step:
            break
    return result


def _historical_review(resolver, step, recipe):
    import mne

    with mne.io.read_raw_fif(
        preproc_step_raw_path(resolver, step, create=False),
        preload=False,
        verbose="ERROR",
    ) as raw:
        state = current_state(raw)
    complete = {k: recipe.get(k, {"action": "skip"}) for k in STEPS}
    return {
        "accepted_state": state,
        "raw_generation": accepted_result_generation_id(step_entry(resolver, "raw")),
        "steps": _review_inputs(recipe_steps(complete), step),
    }


def _validate_historical_review(review, steps, step, resolver):
    if isinstance(review, dict):
        if (
            set(review) != {"accepted_state", "raw_generation", "steps"}
            or not isinstance(review["accepted_state"], dict)
            or set(review["accepted_state"]) != {"annotations", "bads"}
        ):
            raise ConfigError(f"Invalid saved review for {step}.")
        prefix = review["steps"]
        if isinstance(prefix, dict) and isinstance(prefix.get("annotations"), dict):
            annotations = prefix["annotations"]
            if "rows" in annotations:
                prefix = {
                    **prefix,
                    "annotations": {
                        **annotations,
                        "rows": _normalized_annotation_rows(annotations["rows"]),
                    },
                }
        if review["raw_generation"] != accepted_result_generation_id(
            step_entry(resolver, "raw")
        ) or prefix != _review_inputs(steps, step):
            raise ConfigError(
                f"The historical {step} review belongs to a different Raw or preprocessing prefix. Supply decisions for this input."
            )


def _applied_spec(resolver, step, prior):
    from lfptensorpipe.preproc.repair_review import repair_decisions
    from .preproc.steps.filter import default_filter_advance_params

    entry = step_entry(resolver, step)
    params = entry["params"]
    spec = {"action": "apply"}
    if step == "signal_repair":
        spec.update(
            settings=params["settings"],
            decisions=repair_decisions(params["intervals"]),
        )
    elif step == "filter":
        spec.update(
            low_freq=params.get("low_freq"),
            high_freq=params.get("high_freq"),
            notches=params["notches"],
            advance={
                k: params[k] for k in default_filter_advance_params() if k in params
            },
            review=params.get("review_edits", "detected"),
        )
    elif step == "ecg_artifact_removal":
        spec.update(
            {
                k: params[k]
                for k in ("method", "picks", "method_kwargs", "mark_filter_edges")
                if k in params
            }
        )
    else:
        spec.update(
            rows=annotation_rows(resolver.preproc_step_dir(step) / "annotations.csv"),
            mark_filter_edges=params.get("mark_filter_edges", False),
        )
    result = {**prior, step: spec}
    if step == "filter" and "review_edits" not in params:
        spec["review"] = _historical_review(resolver, step, result)
    if step != "filter":
        if "review_edits" in params:
            spec["review_edits"] = params["review_edits"]
        elif "plot" in str(entry.get("message", "")).lower():
            spec["review_edits"] = _historical_review(resolver, step, result)
    return spec


def export_preprocess(context):
    resolver = PathResolver(context)
    if not preproc_step_lineage_is_current(resolver, "finish"):
        raise RunFailed("A current Preprocess Finish is required to export its recipe.")
    result = {}
    for step in STEPS:
        if preproc_step_is_skipped(
            resolver, step
        ) or not preproc_step_lineage_is_current(resolver, step):
            result[step] = {"action": "skip"}
        else:
            result[step] = _applied_spec(resolver, step, result)
    return result


def _matches_applied(resolver, step, steps):
    if not preproc_step_lineage_is_current(resolver, step) or (
        step == "filter" and svc.preproc_filter_review_required(resolver)
    ):
        return False
    saved = _applied_spec(resolver, step, steps)
    saved = recipe_steps(
        {s: saved if s == step else {"action": "skip"} for s in STEPS}
    )[step]
    desired = dict(steps[step])
    if step == "signal_repair" and "decisions" not in desired:
        if not all(r["accepted"] for r in saved["decisions"]):
            return False
        saved.pop("decisions")
    if step == "filter" and desired["review"] == "detected":
        desired["review"] = []
    if step == "ecg_artifact_removal" and desired.get("picks") is None:
        from .tensor.selectors import load_tensor_channel_inventory

        source, _ = capture_preproc_input_generation(resolver, step)
        desired["picks"] = list(
            load_tensor_channel_inventory(
                preproc_step_raw_path(resolver, source)
            ).all_channels
        )
    if step == "annotations" and desired["rows"] != saved["rows"]:
        import mne
        from .preproc.steps.annotations import _clip_annotation_rows_to_raw_support

        # The accepted CSV contains effective rows after clipping to this recording.
        with mne.io.read_raw_fif(
            preproc_step_raw_path(resolver, step), preload=False, verbose="ERROR"
        ) as raw:
            desired["rows"], _, _ = _clip_annotation_rows_to_raw_support(
                desired["rows"], raw
            )
    for value in (desired, saved):
        if not value.get("review_edits"):
            value.pop("review_edits", None)
    return desired == saved


def run_preprocess(context, recipe, *, reused=(), report=None):
    import mne

    steps = recipe_steps(recipe)
    resolver = PathResolver(context)
    if not preproc_step_lineage_is_current(resolver, "raw"):
        if preproc_step_raw_path(resolver, "raw", create=False).exists():
            raise RunFailed(
                "The existing Raw is unaccepted. Review or restore it before running preprocessing."
            )
        _checked_result(svc.bootstrap_raw_step_from_rawdata(context))
    for step, spec in steps.items():
        for field in ("review", "review_edits"):
            if isinstance(spec.get(field), dict):
                _validate_historical_review(spec[field], steps, step, resolver)
    for step, spec in steps.items():
        if step in reused:
            continue
        if report:
            report(step, "running")
        if spec["action"] == "skip":
            if not preproc_step_is_skipped(resolver, step):
                if step_entry(resolver, step).get("completed") is True or (
                    step == "filter" and svc.preproc_filter_review_required(resolver)
                ):
                    _checked_result(svc.skip_preproc_step(context, step))
                else:
                    write_preproc_step_routing(
                        resolver=resolver, step=step, skipped=True
                    )
            status = "skipped"
        elif _matches_applied(resolver, step, steps):
            if preproc_step_is_skipped(resolver, step):
                _checked_result(svc.skip_preproc_step(context, step))
            status = "reused"
        else:
            if step == "signal_repair":
                _checked_result(
                    svc.apply_signal_repair_step(
                        context,
                        params=spec["settings"],
                        decisions=spec.get("decisions"),
                        review_edits=spec.get("review_edits"),
                    )
                )
            elif step == "filter":
                _checked_result(
                    svc.apply_filter_step(
                        context,
                        advance_params=spec.get("advance"),
                        notches=spec.get("notches"),
                        l_freq=spec.get("low_freq"),
                        h_freq=spec.get("high_freq"),
                    )
                )
                with mne.io.read_raw_fif(
                    preproc_filter_preview_raw_path(resolver),
                    preload=False,
                    verbose="ERROR",
                ) as preview:
                    if spec["review"] != "detected":
                        apply_edits(preview, spec["review"])
                    _checked_result(
                        svc.finalize_filter_review(
                            context,
                            reviewed_annotations=preview.annotations.copy(),
                            reviewed_bads=list(preview.info["bads"]),
                        )
                    )
            elif step == "ecg_artifact_removal":
                _checked_result(
                    svc.apply_ecg_step(
                        context,
                        **{
                            k: spec[k]
                            for k in (
                                "method",
                                "picks",
                                "method_kwargs",
                                "mark_filter_edges",
                                "review_edits",
                            )
                            if k in spec
                        },
                    )
                )
            else:
                _checked_result(
                    svc.apply_annotations_step(
                        context,
                        rows=spec["rows"],
                        review_edits=spec.get("review_edits"),
                        mark_filter_edges=spec.get("mark_filter_edges", False),
                    )
                )
            status = "computed"
        if report:
            report(step, status)
    if report:
        report("finish", "running")
    current = preproc_step_lineage_is_current(resolver, "finish")
    message = (
        "Using the current Preprocess Finish."
        if current
        else _checked_result(svc.apply_finish_step(context))
    )
    if report:
        report("finish", "reused" if current else "computed")
    return message
