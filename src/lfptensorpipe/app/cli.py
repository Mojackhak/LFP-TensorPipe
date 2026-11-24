"""Execute page configurations through the application services."""

from __future__ import annotations

import sys
from collections.abc import Iterable
from copy import deepcopy
from typing import Any

from .path_resolver import PathResolver, RecordContext
from .shared.runlog_store import read_ui_state, write_ui_state


class ConfigError(ValueError):
    """The supplied page configuration is not executable for this target."""


class RunFailed(RuntimeError):
    """The requested page did not complete successfully."""


def _require_fields(node: Any, keys: Iterable[str], label: str) -> None:
    if not isinstance(node, dict):
        raise ConfigError(f"{label} must be an object.")
    missing = sorted(set(keys) - set(node))
    if missing:
        raise ConfigError(f"{label} is missing: {', '.join(missing)}.")


def _patch_state(
    context: RecordContext, keys: tuple[str, ...], patch: dict[str, Any]
) -> None:
    path = PathResolver(context).record_ui_state_path()
    payload = read_ui_state(path) or {}
    node = payload
    for key in keys:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ConfigError(
                f"Existing record state {'.'.join(keys)} must be an object."
            )
    node.update(patch)
    write_ui_state(path, payload)


def _checked_result(result: tuple) -> str:
    ok, message = result[:2]
    if not ok:
        raise RunFailed(message)
    if "warning" in message.lower():
        print(message, file=sys.stderr)
    return message


def prepare_tensor_config(
    context: RecordContext, node: dict[str, Any], version: int
) -> dict[str, Any]:
    from .preproc.paths import preproc_step_raw_path
    from .tensor import service as tensor_service
    from .tensor.config import (
        convert_legacy_tensor_metric,
        normalize_tensor_config_metric_params,
    )
    from .tensor.cpu_budget import normalize_tensor_cpu_percent
    from .tensor.orchestration_plan_validation import prepare_metric_plan_inputs
    from .tensor.params import TENSOR_CHANNEL_SELECTOR_KEYS, TENSOR_METRICS_BY_KEY
    from .tensor.selectors import load_tensor_channel_inventory

    _require_fields(
        node,
        ("selected_metrics", "metric_params", "mask_edge_effects", "cpu_percent"),
        "tensor",
    )
    metrics = node["selected_metrics"]
    if (
        not isinstance(metrics, list)
        or not metrics
        or any(
            not isinstance(key, str)
            or key not in TENSOR_METRICS_BY_KEY
            or not TENSOR_METRICS_BY_KEY[key].supported
            for key in metrics
        )
    ):
        raise ConfigError(
            "tensor.selected_metrics must contain supported metric names."
        )
    metrics = list(dict.fromkeys(metrics))
    if not isinstance(node["mask_edge_effects"], bool):
        raise ConfigError("tensor.mask_edge_effects must be true or false.")
    _require_fields(node["metric_params"], metrics, "tensor.metric_params")
    inventory = load_tensor_channel_inventory(
        preproc_step_raw_path(PathResolver(context), "finish")
    )
    normalized_params = {}
    try:
        cpu = normalize_tensor_cpu_percent(node["cpu_percent"])
        for metric in metrics:
            raw = node["metric_params"][metric]
            if not isinstance(raw, dict):
                raise ValueError(f"tensor.metric_params.{metric} must be an object.")
            raw = (
                convert_legacy_tensor_metric(metric, raw) if version == 3 else dict(raw)
            )
            selector = (
                "selected_channels"
                if metric in TENSOR_CHANNEL_SELECTOR_KEYS
                else "selected_pairs"
            )
            required = [selector, "method"]
            if metric in tensor_service.TENSOR_COMMON_BASIC_KEYS:
                required.extend(("low_freq_hz", "high_freq_hz", "freq_step_hz"))
            if metric in {"psi", "burst"}:
                required.append("bands")
            if metric == "psi":
                required.append("freq_step_hz")
            _require_fields(raw, required, f"tensor.metric_params.{metric}")
            normalized, _ = normalize_tensor_config_metric_params(
                metric,
                raw,
                available_channels=inventory.usable_channels,
                legacy_notch_fields=version == 3,
                strict=True,
            )
            prepare_metric_plan_inputs(
                tensor_service,
                context,
                metric_key=metric,
                metric_label=TENSOR_METRICS_BY_KEY[metric].display_name,
                metric_params=normalized,
            )
            if (
                metric == "burst"
                and normalized.get("thresholds") is None
                and normalized.get("baseline_keep")
            ):
                from .tensor.annotation_source import (
                    load_burst_baseline_annotation_labels,
                )

                available = set(load_burst_baseline_annotation_labels(context))
                if set(normalized["baseline_keep"]) - available:
                    raise ValueError(
                        "Burst baseline_keep contains unavailable annotations."
                    )
            normalized_params[metric] = normalized
    except (ValueError, TypeError) as exc:
        raise ConfigError(str(exc)) from exc
    snapshot = deepcopy(node)
    snapshot["metric_params"].update(normalized_params)
    snapshot.update(selected_metrics=metrics, cpu_percent=cpu)
    return snapshot


def _run_tensor(context: RecordContext, node: dict[str, Any], version: int) -> str:
    from .tensor.lineage import capture_tensor_input_generation

    if capture_tensor_input_generation(PathResolver(context)) is None:
        raise RunFailed(
            "Preprocess Finish must be current before Build Tensor can run."
        )
    snapshot = prepare_tensor_config(context, node, version)
    return execute_tensor_metrics(context, snapshot, snapshot["selected_metrics"])


def execute_tensor_metrics(context, snapshot, metrics):
    from .tensor.worker import run_tensor_subprocess

    _patch_state(context, ("tensor",), snapshot)
    if not metrics:
        return "Using accepted Tensor metrics."
    message = _checked_result(
        run_tensor_subprocess(
            context,
            selected_metrics=metrics,
            metric_params_map={key: snapshot["metric_params"][key] for key in metrics},
            mask_edge_effects=snapshot["mask_edge_effects"],
            cpu_percent=snapshot["cpu_percent"],
        )
    )
    return f"Metrics: {', '.join(metrics)}\n{message}"


def _trial_config(
    context: RecordContext, trial: str, *, allow_missing: bool
) -> dict | None:
    from .alignment.trial_config import _load_trial_config_from_log

    resolver = PathResolver(context)
    path = resolver.alignment_paradigm_dir(trial)
    if not path.exists():
        if allow_missing:
            return None
        raise ConfigError(f"Trial not found: {trial}")
    config = _load_trial_config_from_log(resolver, slug=trial)
    if not path.is_dir() or config is None:
        raise ConfigError(
            f"Trial directory exists without a valid trial configuration: {path}"
        )
    return config


def _select_trial_state(
    context: RecordContext,
    trial_config: dict[str, Any],
    picks: list[int] | None = None,
    *,
    submitted: bool = False,
) -> None:
    path = PathResolver(context).record_ui_state_path()
    state = read_ui_state(path) or {}
    alignment = state.setdefault("alignment", {})
    if not isinstance(alignment, dict):
        raise ConfigError("Existing record state alignment must be an object.")
    slug = trial_config["trial_slug"]
    fields = ("method", "method_params", "method_params_by_method")
    drafts = alignment.setdefault("trial_params_by_slug", {})
    if not isinstance(drafts, dict):
        raise ConfigError("Existing alignment trial drafts must be an object.")
    previous = alignment.get("trial_slug")
    if previous == slug:
        saved = alignment
    else:
        if isinstance(previous, str) and isinstance(
            alignment.get("method_params"), dict
        ):
            drafts[previous] = {key: alignment.get(key) for key in fields}
        saved = drafts.pop(slug, {})
    patch = {key: deepcopy(trial_config.get(key)) for key in fields}
    if isinstance(saved, dict) and saved.get("method"):
        if submitted:
            cached = saved.get("method_params_by_method")
            if isinstance(cached, dict):
                patch["method_params_by_method"].update(cached)
            if saved["method"] != patch["method"] and isinstance(
                saved.get("method_params"), dict
            ):
                patch["method_params_by_method"][saved["method"]] = saved[
                    "method_params"
                ]
        else:
            patch.update({key: saved[key] for key in fields if key in saved})
    if isinstance(patch["method_params_by_method"], dict):
        patch["method_params_by_method"] = {
            k: v
            for k, v in patch["method_params_by_method"].items()
            if k != patch["method"]
        }
    patch["trial_slug"] = slug
    if picks is not None:
        patch["picked_epoch_indices"] = list(picks)
    alignment.update(patch)
    write_ui_state(path, state)


def _run_alignment(
    context: RecordContext, node: dict[str, Any], trial: str, config_store: Any
) -> str:
    from .alignment import service as svc

    if not svc._completed_tensor_metrics(PathResolver(context)):
        raise RunFailed("No current Tensor metrics are available for Align Epochs.")
    existing = _trial_config(context, trial, allow_missing=True)
    reference = None
    if "epoch_selection" in node:
        from .cli_epochs import selection_reference

        reference = selection_reference(node["epoch_selection"], context, trial)
    replay_mode = node.get("epoch_match", "exact")
    if replay_mode not in ("exact", "common") or (
        "epoch_match" in node and reference is None
    ):
        raise ConfigError(
            "epoch_match requires an epoch selection and exact or common matching."
        )
    _require_fields(node, ("method", "method_params"), "alignment")
    method, params = node["method"], node["method_params"]
    if not isinstance(method, str) or method not in svc.ALIGNMENT_METHODS_BY_KEY:
        raise ConfigError(f"Unknown alignment method: {method!r}")
    paired = (
        method == "linear_warper"
        and isinstance(params, dict)
        and "cue_response" in params
    )
    required = (
        ("drop_bad", "drop_fields", "sample_rate")
        if paired
        else svc.default_alignment_method_params(method)
    )
    _require_fields(params, required, "alignment.method_params")
    explicit = paired or (method == "linear_warper" and "event_groups" in params)
    if explicit:
        from .alignment.annotation_source import (
            explicit_alignment_annotation_error,
            load_alignment_annotation_events,
        )

        events = load_alignment_annotation_events(context)
        labels = list(events)
    else:
        labels = svc.load_alignment_annotation_labels(context)
    for field in ("drop_bad", "linear_warp"):
        if field in params and not isinstance(params[field], bool):
            raise ConfigError(f"alignment.method_params.{field} must be true or false.")
    fields = params["drop_fields"]
    if (
        not isinstance(fields, list)
        or not fields
        or any(not isinstance(item, str) or not item.strip() for item in fields)
    ):
        raise ConfigError(
            "alignment.method_params.drop_fields must contain nonempty strings."
        )
    if explicit:
        requested_labels = []
    elif method == "linear_warper":
        anchors = params["anchors_percent"]
        if not isinstance(anchors, dict) or not anchors:
            raise ConfigError(
                "alignment.method_params.anchors_percent must contain event anchors."
            )
        requested_labels = list(anchors.values())
    else:
        requested_labels = params["annotations"]
    if (
        reference
        and replay_mode == "common"
        and method != "linear_warper"
        and isinstance(requested_labels, list)
    ):
        reviewed_labels = {r["epoch"] for r in reference["decisions"]}
        # A previously reviewed label may disappear from Finish in a preprocessing variant.
        retained = [
            label
            for label in requested_labels
            if label in labels or label not in reviewed_labels
        ]
        if retained != requested_labels:
            params = {**params, "annotations": retained}
            requested_labels = retained
    if not explicit and (
        not isinstance(requested_labels, list)
        or not requested_labels
        or any(
            not isinstance(label, str) or label.strip() not in labels
            for label in requested_labels
        )
    ):
        raise ConfigError(
            "Alignment configuration must select available annotation labels."
        )
    ok, normalized, message = svc.validate_alignment_method_params(
        method, params, annotation_labels=labels
    )
    if not ok:
        raise ConfigError(message)
    if explicit:
        message = explicit_alignment_annotation_error(normalized, events)
        if message:
            raise ConfigError(message)
    if existing is None:
        _checked_result(
            svc.create_alignment_paradigm(config_store, name=trial, context=context)
        )
    _checked_result(
        svc.update_alignment_paradigm(
            config_store,
            slug=trial,
            method=method,
            method_params=normalized,
            context=context,
        )
    )
    config = _trial_config(context, trial, allow_missing=False)
    _select_trial_state(context, config, submitted=True)
    ok, message, rows = svc.run_align_epochs(
        context, config_store=config_store, paradigm_slug=trial
    )
    _checked_result((ok, message))
    picks = [row["epoch_index"] for row in rows]
    if reference is not None:
        from .cli_epochs import replay_picks

        picks = replay_picks(context, trial, reference, replay_mode)
    if not svc.persist_alignment_epoch_picks(
        context, paradigm_slug=trial, picked_epoch_indices=picks
    ):
        raise RunFailed("Could not save the epoch selection after Align Run.")
    _select_trial_state(context, config, picks)
    finish_message = _checked_result(
        svc.finish_alignment_epochs(
            context, paradigm_slug=trial, picked_epoch_indices=picks
        )
    )
    scope = (
        "all generated epochs"
        if reference is None
        else f"{replay_mode} source-event replay"
    )
    return f"Epochs: {len(picks)} ({scope})\n{message}\n{finish_message}"


def prepare_features_config(
    context: RecordContext,
    node: dict[str, Any],
    metrics: list[str],
    *,
    trial: str | None = None,
) -> dict[str, Any]:
    from .features.derive_axes import (
        feature_band_support_error,
        normalize_feature_axis_rows,
    )
    from .features.derive_defaults import _metric_uses_auto_bands
    from .features.phase_time import (
        phase_limit,
        trial_seconds_duration,
        validate_phase_bounds,
    )
    from .shared.runlog_store import read_run_log
    from .tensor.paths import tensor_metric_log_path

    _require_fields(node, ("axes_by_metric",), "features")
    _require_fields(node["axes_by_metric"], metrics, "features.axes_by_metric")
    axes = {}
    duration = None
    for metric in metrics:
        source = node["axes_by_metric"][metric]
        auto = _metric_uses_auto_bands(metric)
        _require_fields(
            source,
            ("times",) if auto else ("bands", "times"),
            f"features.axes_by_metric.{metric}",
        )
        bands = (
            [] if auto else normalize_feature_axis_rows(source["bands"], min_start=0.0)
        )
        unit = source.get("time_unit", "percent")
        try:
            if unit == "s" and duration is None and trial:
                duration = trial_seconds_duration(PathResolver(context), trial)
            limit = phase_limit(unit, duration)
        except ValueError as exc:
            raise ConfigError(f"{metric}: {exc}") from exc
        times = normalize_feature_axis_rows(
            source["times"], min_start=0.0, max_end=limit, allow_duplicate_names=True
        )
        for key in (("times",) if auto else ("times", "bands")):
            normalized = times if key == "times" else bands
            raw = source[key]
            if (
                not isinstance(raw, list)
                or not normalized
                or len(raw) != len(normalized)
            ):
                if key == "times" and unit == "s" and isinstance(raw, list):
                    try:
                        validate_phase_bounds(raw, unit, duration)
                    except ValueError as exc:
                        raise ConfigError(f"{metric}: {exc}") from exc
                raise ConfigError(
                    f"features.axes_by_metric.{metric}.{key} contains missing, invalid, or duplicate rows."
                )
        if not auto:
            log = (
                read_run_log(tensor_metric_log_path(PathResolver(context), metric))
                or {}
            )
            params = log.get("params", {})
            if "low_freq" in params and "high_freq" in params:
                message = feature_band_support_error(
                    metric,
                    bands,
                    (float(params["low_freq"]), float(params["high_freq"])),
                )
                if message:
                    raise ConfigError(message)
        axes[metric] = {
            "bands": bands,
            "times": times,
            **({"time_unit": unit} if "time_unit" in source else {}),
        }
    return {"active_metric": node.get("active_metric", ""), "axes_by_metric": axes}


def _run_features(
    context: RecordContext, node: dict[str, Any], trial: str, config_store: Any
) -> str:
    from .alignment.epoch_view import load_alignment_epoch_picks
    from .alignment.generation import (
        accepted_alignment_metrics,
        alignment_generation_rerun_message,
    )
    from .features.service import run_extract_features

    config = _trial_config(context, trial, allow_missing=False)
    resolver = PathResolver(context)
    message = alignment_generation_rerun_message(
        resolver, trial_slug=trial, stage="finish"
    )
    if message:
        raise RunFailed(message)
    metrics = accepted_alignment_metrics(resolver, trial_slug=trial, stage="finish")
    if not metrics:
        raise RunFailed("The selected trial must have a current Align Finish.")
    snapshot = prepare_features_config(context, node, metrics, trial=trial)
    message = _checked_result(
        run_extract_features(
            context,
            paradigm_slug=trial,
            config_store=config_store,
            axes_by_metric=snapshot["axes_by_metric"],
            **({"execution": node["execution"]} if "execution" in node else {}),
        )
    )
    _patch_state(context, ("features", "trial_params_by_slug", trial), snapshot)
    _select_trial_state(
        context, config, load_alignment_epoch_picks(context, paradigm_slug=trial)
    )
    return message


def _run_localize(
    context: RecordContext, node: dict[str, Any], config_store: Any
) -> str:
    from .localize.paths import LEAD_DBS, USER_LABELS

    source = node.get("source", LEAD_DBS)
    if source == USER_LABELS:
        import csv
        from pathlib import Path
        from .localize.user_labels import read_labels
        from .localize.service import run_localize_apply
        from .preproc.paths import rawdata_input_fif_path
        from .tensor.selectors import load_tensor_channel_inventory

        raw_path = rawdata_input_fif_path(context)
        if not raw_path.is_file():
            raw_path = PathResolver(context).preproc_root / "raw/raw.fif"
        channels = tuple(load_tensor_channel_inventory(raw_path).all_channels)
        if (
            not isinstance(node.get("labels_csv"), str)
            or not node["labels_csv"].strip()
        ):
            raise ConfigError("User Labels requires a labels_csv path.")
        try:
            labels = read_labels(Path(node["labels_csv"]), channels)
        except (OSError, ValueError, csv.Error) as exc:
            raise ConfigError(f"Invalid User Labels CSV: {exc}") from exc
        if not labels.regions or not labels.assessed:
            raise ConfigError(
                "User Labels requires at least one region and assessed channel."
            )
        message = _checked_result(
            run_localize_apply(
                project_root=context.project_root,
                subject=context.subject,
                record=context.record,
                space="",
                atlas="",
                selected_regions=[],
                source=source,
                channels=channels,
                user_labels=labels,
            )
        )
        _patch_state(context, ("localize",), {"selected_source": source})
        return message
    if source != LEAD_DBS:
        raise ConfigError(f"Unknown localization source: {source!r}")
    from lfptensorpipe.anat.lead_config import discover_regions

    from .localize import service as svc
    from .localize.config import (
        build_lead_signature,
        normalize_localize_lead_signature,
        normalize_localize_match_payload,
        validate_match_contacts,
    )
    from .localize.lead_inputs import cached_lead_signature
    from .localize.paths import (
        discover_atlases,
        infer_subject_space,
        localize_log_path,
        reconstruction_mat_path,
        source_results,
    )
    from .runlog_store import read_run_log
    from .preproc.paths import rawdata_input_fif_path
    from .tensor.selectors import load_tensor_channel_inventory

    _require_fields(
        node, ("atlas", "selected_regions", "match", "lead_signature"), "localize"
    )
    paths = svc.load_localize_paths(config_store)
    try:
        space, message = infer_subject_space(context.project_root, context.subject)
        if space is None:
            raise RunFailed(message)
        atlas = node["atlas"]
        if atlas not in discover_atlases(paths.leaddbs_dir, space):
            raise ConfigError(f"Unavailable Localize atlas: {atlas!r}")
        regions = node["selected_regions"]
        available_regions = set(
            discover_regions(
                paths.leaddbs_dir / "templates" / "space" / space / "atlases" / atlas
            )
        )
        if (
            not isinstance(regions, list)
            or not regions
            or any(
                not isinstance(name, str) or name not in available_regions
                for name in regions
            )
        ):
            raise ConfigError(
                "Localize selected_regions must contain available region names."
            )
        args = (context.project_root, context.subject, context.record)
        saved = source_results(read_run_log(localize_log_path(*args))).get(LEAD_DBS, {})
        reco = reconstruction_mat_path(context.project_root, context.subject)
        current_signature = cached_lead_signature(saved, reco, paths)
        reconstruction = None
        if current_signature is None:
            ok, message, reconstruction = svc.load_reconstruction_contacts(
                context.project_root, context.subject, paths
            )
            _checked_result((ok, message))
            current_signature = build_lead_signature(reconstruction["leads"])
        raw_path = rawdata_input_fif_path(context)
        if not raw_path.is_file():
            raw_path = PathResolver(context).preproc_root / "raw" / "raw.fif"
        channels = load_tensor_channel_inventory(raw_path).all_channels
        try:
            signature = normalize_localize_lead_signature(node["lead_signature"])
            if signature != current_signature:
                raise ValueError(
                    "Localize lead_signature does not match the current reconstruction."
                )
            match = normalize_localize_match_payload(
                node["match"], expected_channels=channels, strict=True
            )
            validate_match_contacts(match, signature)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        match.update(
            subject=context.subject,
            record=context.record,
            space=space,
            atlas=atlas,
            lead_signature=signature,
        )
        _patch_state(
            context,
            ("localize",),
            {
                "atlas": atlas,
                "selected_regions": regions,
                "match": match,
                "selected_source": LEAD_DBS,
            },
        )
        return _checked_result(
            svc.run_localize_apply(
                project_root=context.project_root,
                subject=context.subject,
                record=context.record,
                space=space,
                atlas=atlas,
                selected_regions=regions,
                paths=paths,
                channels=channels,
                load_reconstruction_contacts_fn=(
                    (lambda *_: (True, "", reconstruction))
                    if reconstruction is not None
                    else None
                ),
            )
        )
    finally:
        if not svc.shutdown_matlab_runtime():
            raise RunFailed("Localize could not shut down its owned MATLAB runtime.")


def run_page(
    context: RecordContext,
    *,
    page: str,
    node: dict,
    version: int,
    trial: str | None,
    config_store=None,
) -> str:
    """Run one page; all optional service imports are action-local."""
    if page == "preprocess":
        from .cli_preprocess import run_preprocess

        return run_preprocess(context, node)
    if page == "tensor":
        return _run_tensor(context, node, version)
    if config_store is None:
        from .config_store import AppConfigStore

        config_store = AppConfigStore()
    if page == "alignment":
        return _run_alignment(context, node, trial, config_store)
    if page == "features":
        return _run_features(context, node, trial, config_store)
    if page == "localize":
        return _run_localize(context, node, config_store)
    raise ConfigError(f"Unsupported page: {page}")
