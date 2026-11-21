"""Accepted-generation lineage for Build Tensor metric artifacts."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
import json
import os
from pathlib import Path
from typing import Any, Concatenate, Mapping, ParamSpec, TypeVar, cast
from uuid import uuid4

import yaml

from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.preproc.finish_fingerprint import (
    annotation_dependency_digest,
    finish_current_annotation_view,
    finish_legacy_bridge,
    finish_tensor_input_receipts,
)
from lfptensorpipe.app.preproc.lineage import preproc_step_lineage_is_current
from lfptensorpipe.app.runlog_store import read_run_log
from lfptensorpipe.app.shared.atomic_outputs import OUTPUT_TRANSACTION_RUN_ID_ENV
from lfptensorpipe.app.shared.generation_lineage import (
    accepted_result_generation_id,
    input_generation_receipts_match,
    preproc_generation_ref,
)
from lfptensorpipe.app.shared.runlog_store import cache_in_run_log_read_snapshot
from lfptensorpipe.utils.freqs import (
    decimal_notch_intervals,
    frequency_bin_count,
    regular_frequency_grid,
)

from .paths import (
    tensor_metric_config_path,
    tensor_metric_log_path,
    tensor_metric_tensor_path,
    tensor_output_metric_keys,
)

TENSOR_INPUT_LINEAGE_ENV = "LFPTENSORPIPE_TENSOR_INPUT_LINEAGE"
TENSOR_CURRENT_FINISH_REQUIRED_MESSAGE = (
    "Preprocess finish must be current before running a Tensor metric."
)

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _read_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = read_run_log(path)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _finish_log_path(resolver: PathResolver) -> Path:
    return resolver.preproc_step_dir("finish", create=False) / "lfptensorpipe_log.json"


def capture_tensor_input_generation(
    resolver: PathResolver,
) -> dict[str, str | None] | None:
    """Capture the current Finish receipt for one Build run.

    The receipt names the signal, record structure, and BAD/EDGE mask rather
    than the Finish event, so an annotation-only Finish rewrite keeps it.
    """
    if not preproc_step_lineage_is_current(resolver, "finish"):
        return None
    receipts = finish_tensor_input_receipts(_read_payload(_finish_log_path(resolver)))
    return {preproc_generation_ref("finish"): receipts[0] if receipts else None}


def tensor_input_generation_matches(
    resolver: PathResolver,
    expected: Mapping[str, str | None],
) -> bool:
    """Return whether the captured Build input is still current."""
    current = capture_tensor_input_generation(resolver)
    return current is not None and dict(current) == dict(expected)


def serialize_tensor_input_lineage(
    context: RecordContext,
    input_generations: Mapping[str, str | None],
) -> str:
    """Serialize one parent-captured Build input snapshot for worker processes."""
    return json.dumps(
        {
            "context": {
                "project_root": str(context.project_root),
                "subject": context.subject,
                "record": context.record,
            },
            "input_generations": dict(input_generations),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


@contextmanager
def tensor_runner_input_lineage(
    context: RecordContext,
    *,
    require_run_id: bool = False,
) -> Iterator[bool]:
    """Install one direct runner's Finish receipt before its first input read."""
    prior_payload = os.environ.get(TENSOR_INPUT_LINEAGE_ENV)
    installed_lineage = not (prior_payload is not None and prior_payload.strip())
    if installed_lineage:
        input_generations = capture_tensor_input_generation(PathResolver(context))
        if input_generations is None:
            yield False
            return
        os.environ[TENSOR_INPUT_LINEAGE_ENV] = serialize_tensor_input_lineage(
            context,
            input_generations,
        )

    prior_run_id = os.environ.get(OUTPUT_TRANSACTION_RUN_ID_ENV)
    installed_run_id = require_run_id and not str(prior_run_id or "").strip()
    if installed_run_id:
        os.environ[OUTPUT_TRANSACTION_RUN_ID_ENV] = uuid4().hex
    try:
        yield True
    finally:
        if installed_run_id:
            if prior_run_id is None:
                os.environ.pop(OUTPUT_TRANSACTION_RUN_ID_ENV, None)
            else:
                os.environ[OUTPUT_TRANSACTION_RUN_ID_ENV] = prior_run_id
        if installed_lineage:
            if prior_payload is None:
                os.environ.pop(TENSOR_INPUT_LINEAGE_ENV, None)
            else:
                os.environ[TENSOR_INPUT_LINEAGE_ENV] = prior_payload


def tensor_runner_entry(
    runner: Callable[Concatenate[RecordContext, _P], _R],
) -> Callable[Concatenate[RecordContext, _P], _R]:
    """Wrap one public Tensor runner with direct-call Finish lineage ownership."""

    @wraps(runner)
    def wrapped(
        context: RecordContext,
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _R:
        with tensor_runner_input_lineage(context) as ready:
            if not ready:
                return cast(
                    _R,
                    (False, TENSOR_CURRENT_FINISH_REQUIRED_MESSAGE),
                )
            return runner(context, *args, **kwargs)

    return cast(Callable[Concatenate[RecordContext, _P], _R], wrapped)


def trgc_runner_entry(
    runner: Callable[Concatenate[RecordContext, _P], _R],
) -> Callable[Concatenate[RecordContext, _P], _R]:
    """Wrap one TRGC runner with Finish lineage and non-empty run identity."""

    @wraps(runner)
    def wrapped(
        context: RecordContext,
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _R:
        with tensor_runner_input_lineage(context, require_run_id=True) as ready:
            if not ready:
                return cast(
                    _R,
                    (False, TENSOR_CURRENT_FINISH_REQUIRED_MESSAGE),
                )
            return runner(context, *args, **kwargs)

    return cast(Callable[Concatenate[RecordContext, _P], _R], wrapped)


def _decode_tensor_input_lineage(
    raw_payload: str,
) -> tuple[RecordContext, dict[str, str | None]]:
    try:
        payload = json.loads(raw_payload)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Build Tensor input lineage payload is invalid.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Build Tensor input lineage payload is invalid.")
    context_payload = payload.get("context")
    input_payload = payload.get("input_generations")
    if not isinstance(context_payload, dict) or not isinstance(input_payload, dict):
        raise RuntimeError("Build Tensor input lineage payload is invalid.")
    project_root = str(context_payload.get("project_root", "")).strip()
    subject = str(context_payload.get("subject", "")).strip()
    record = str(context_payload.get("record", "")).strip()
    if not project_root or not subject or not record:
        raise RuntimeError("Build Tensor input lineage payload is invalid.")
    normalized: dict[str, str | None] = {}
    for raw_ref, raw_generation_id in input_payload.items():
        ref = str(raw_ref).strip()
        if not ref or (
            raw_generation_id is not None and not isinstance(raw_generation_id, str)
        ):
            raise RuntimeError("Build Tensor input lineage payload is invalid.")
        normalized[ref] = raw_generation_id
    return (
        RecordContext(
            project_root=Path(project_root),
            subject=subject,
            record=record,
        ),
        normalized,
    )


def tensor_input_generations_from_environment() -> dict[str, str | None] | None:
    """Return the parent-captured Build receipt available to this worker."""
    raw_payload = os.environ.get(TENSOR_INPUT_LINEAGE_ENV, "").strip()
    if not raw_payload:
        return None
    _context, input_generations = _decode_tensor_input_lineage(raw_payload)
    return input_generations


def require_tensor_input_generation_unchanged() -> None:
    """Abort candidate promotion when the Build input changed during compute."""
    raw_payload = os.environ.get(TENSOR_INPUT_LINEAGE_ENV, "").strip()
    if not raw_payload:
        return
    context, expected = _decode_tensor_input_lineage(raw_payload)
    if not tensor_input_generation_matches(PathResolver(context), expected):
        raise RuntimeError(
            "Preprocess Finish changed while Build Tensor was running; "
            "candidate outputs were not accepted."
        )


def tensor_input_generation_changed_during_run() -> bool:
    """Return whether an active Build snapshot no longer matches Finish."""
    raw_payload = os.environ.get(TENSOR_INPUT_LINEAGE_ENV, "").strip()
    if not raw_payload:
        return False
    context, expected = _decode_tensor_input_lineage(raw_payload)
    return not tensor_input_generation_matches(PathResolver(context), expected)


def _endpoint_count_changed(low: float, high: float, step: float) -> bool:
    """Check whether the former binary step count differs from the decimal one."""
    return int((high - low) / step) + 1 != frequency_bin_count(low, high, step)


def _frequency_support_is_current(
    resolver: PathResolver, metric_key: str, params: Mapping[str, Any]
) -> bool:
    if metric_key == "burst":
        if params.get("method") not in {"morlet", "multitaper"}:
            return True
        return _spectral_burst_grid_is_current(resolver, params)
    if not {"low_freq", "high_freq", "step_hz", "n_freqs"} <= params.keys():
        # PSI selects frequencies strictly inside each band and stores a band axis.
        return True
    low, high, step = (float(params[k]) for k in ("low_freq", "high_freq", "step_hz"))
    if _endpoint_count_changed(low, high, step):
        if params["n_freqs"] != regular_frequency_grid(low, high, step).size:
            return False
    if metric_key in {"periodic", "aperiodic", "periodic_aperiodic"}:
        fit_low, fit_high = params.get("freq_range_hz", [low, high])
        if _endpoint_count_changed(fit_low, fit_high, step):
            # A successful SpecParam fit has a clean bin at each model boundary.
            return (
                params.get("consumed_tfr_frequency_max_hz")
                == regular_frequency_grid(fit_low, fit_high, step)[-1]
            )
    return True


def _spectral_burst_grid_is_current(
    resolver: PathResolver, params: Mapping[str, Any]
) -> bool:
    from lfptensorpipe.io.burst_thresholds import recorded_burst_band_grid

    bands = [
        band
        for band in params["thresholds_used"]["bands"]
        if "frequency_grid_hz" not in band
    ]
    if not bands:
        return True
    step = float(params["estimator_signature"]["freq_step_hz"])
    from .frequency import normalize_tensor_metric_notch_params

    centers, radii = normalize_tensor_metric_notch_params(
        params.get("notches"), params.get("notch_radii")
    )
    intervals = [
        (center - radius, center + radius) for center, radius in zip(centers, radii)
    ]
    precise = decimal_notch_intervals(centers, radii)
    nominal = []
    clipped_edge = False
    grid_changed = False
    for band in bands:
        segments = band["segments_hz"]
        low, high = segments[0][0], segments[-1][1]
        nominal.append({"name": band["name"], "start": low, "end": high})
        clipped_edge |= any(
            start <= edge <= stop for edge in (low, high) for start, stop in intervals
        )
        grid_changed |= _endpoint_count_changed(low, high, step) or any(
            old != exact and old[1] >= low and old[0] <= high
            for old, exact in zip(intervals, precise)
        )
    if not (clipped_edge or grid_changed):
        return True
    if clipped_edge:
        # Only the accepted configuration retained the original bounds when
        # a notch cut an outer edge of an old spectral Burst band.
        with tensor_metric_config_path(resolver, "burst").open(
            encoding="utf-8"
        ) as stream:
            nominal = yaml.safe_load(stream)["bands"]
    from .runners.burst import _build_runtime_bands

    _, grids = _build_runtime_bands(
        bands=nominal,
        low_freq=params["low_freq"],
        high_freq=params["high_freq"],
        notch_intervals=intervals,
        spectral_step=step,
        spectral_notches=precise,
    )
    return all(
        grids[band["name"]] == recorded_burst_band_grid(band, step) for band in bands
    )


@cache_in_run_log_read_snapshot(
    lambda resolver, metric_key: (resolver.context, metric_key)
)
def tensor_metric_lineage_is_current(
    resolver: PathResolver,
    metric_key: str,
) -> bool:
    """Return whether one persisted Tensor metric matches current Finish."""
    if not preproc_step_lineage_is_current(resolver, "finish"):
        return False
    log_path = tensor_metric_log_path(resolver, metric_key, create=False)
    payload = _read_payload(log_path)
    if (
        payload is None
        or payload.get("completed") is not True
        or not all(
            tensor_metric_tensor_path(resolver, key).is_file()
            for key in tensor_output_metric_keys([metric_key])
        )
    ):
        return False
    finish_payload = _read_payload(_finish_log_path(resolver))
    receipt = next(
        (
            candidate
            for candidate in finish_tensor_input_receipts(finish_payload)
            if input_generation_receipts_match(
                payload,
                expected={preproc_generation_ref("finish"): candidate},
            )
        ),
        None,
    )
    if receipt is None:
        return False
    try:
        if not _frequency_support_is_current(resolver, metric_key, payload["params"]):
            return False
    except (KeyError, OSError, TypeError, ValueError, ArithmeticError, yaml.YAMLError):
        return False
    return metric_key != "burst" or _burst_baseline_is_current(
        payload,
        finish_payload,
        receipt=receipt,
    )


def _burst_baseline_is_current(
    payload: Mapping[str, Any],
    finish_payload: Mapping[str, Any] | None,
    *,
    receipt: str,
) -> bool:
    """Return whether Burst's baseline annotations still match Finish.

    The Finish receipt covers the signal and the BAD/EDGE mask only; a Burst
    threshold computed from baseline labels also depends on those labels.
    """
    params = payload.get("params")
    baseline_keep = params.get("baseline_keep") if isinstance(params, Mapping) else None
    if not isinstance(baseline_keep, list) or not baseline_keep:
        return True
    current_view = finish_current_annotation_view(finish_payload)
    if current_view is None:
        # A Finish without a fingerprint is matched by its exact generation,
        # which already covers every annotation.
        return True
    expected = annotation_dependency_digest(current_view[1], exact=baseline_keep)
    recorded = params.get("baseline_annotation_digest")
    if isinstance(recorded, str):
        return recorded == expected
    bridge = finish_legacy_bridge(finish_payload)
    if bridge is None or bridge["finish_generation_id"] != receipt:
        return False
    return (
        annotation_dependency_digest(
            bridge["annotation_label_digests"],
            exact=baseline_keep,
        )
        == expected
    )


@cache_in_run_log_read_snapshot(
    lambda resolver, metric_key: (resolver.context, metric_key)
)
def tensor_metric_result_generation_id(
    resolver: PathResolver,
    metric_key: str,
) -> str | None:
    """Return a Tensor result ID only when the metric is recursively current."""
    if not tensor_metric_lineage_is_current(resolver, metric_key):
        return None
    return accepted_result_generation_id(
        _read_payload(tensor_metric_log_path(resolver, metric_key, create=False))
    )


__all__ = [
    "TENSOR_INPUT_LINEAGE_ENV",
    "TENSOR_CURRENT_FINISH_REQUIRED_MESSAGE",
    "capture_tensor_input_generation",
    "require_tensor_input_generation_unchanged",
    "serialize_tensor_input_lineage",
    "tensor_runner_entry",
    "tensor_runner_input_lineage",
    "trgc_runner_entry",
    "tensor_input_generation_matches",
    "tensor_input_generation_changed_during_run",
    "tensor_input_generations_from_environment",
    "tensor_metric_lineage_is_current",
    "tensor_metric_result_generation_id",
]
