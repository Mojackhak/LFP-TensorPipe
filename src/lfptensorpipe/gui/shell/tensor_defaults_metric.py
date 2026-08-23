"""Metric-default helpers for tensor GUI defaults."""

from __future__ import annotations

from lfptensorpipe.app.tensor.orchestration_plan_validation import (
    validate_metric_storage_params,
)
from lfptensorpipe.gui.shell.common import (
    Any,
    RecordContext,
    TENSOR_METRIC_DEFAULTS_KEY,
    TENSOR_UNDIRECTED_METRIC_KEYS,
    _deep_merge_dict,
    build_tensor_metric_notch_payload,
    default_tensor_metric_notch_params,
    load_tensor_frequency_defaults,
)
from lfptensorpipe.io.burst_thresholds import normalize_burst_threshold_payload
from lfptensorpipe.lfp.burst.semantics import (
    BURST_NATIVE_DECIM,
    BURST_NATIVE_HOP_S,
)


def _tensor_supported_methods() -> tuple[str, ...]:
    return ("morlet", "multitaper")


def _default_periodic_time_smooth_kernel_size(
    time_resolution_s: float,
    hop_s: float,
) -> int:
    try:
        ratio = float(time_resolution_s) / float(hop_s)
    except Exception:
        ratio = 1.0
    if not bool(0.0 < ratio < float("inf")):
        ratio = 1.0
    kernel = max(1, int(round(ratio)))
    if kernel % 2 == 0:
        kernel += 1
    return int(kernel)


def _default_tensor_metric_params(
    self,
    metric_key: str,
    *,
    context: RecordContext | None,
) -> dict[str, Any]:
    if context is None:
        low_freq = 1.0
        high_freq = 100.0
        step_hz = 0.5
    else:
        low_freq, high_freq, step_hz = load_tensor_frequency_defaults(context)

    common = {
        "low_freq_hz": float(low_freq),
        "high_freq_hz": float(high_freq),
        "freq_step_hz": float(step_hz),
        "time_resolution_s": 0.5,
        "hop_s": 0.025,
        **default_tensor_metric_notch_params(),
    }
    if metric_key == "raw_power":
        return {
            **common,
            "method": "morlet",
            "min_cycles": 3.0,
            "max_cycles": None,
            "mt_time_bandwidth_product": 4.0,
            "mt_min_cycles": 3.0,
            "mt_max_cycles": None,
        }
    if metric_key == "periodic_aperiodic":
        return {
            **common,
            "method": "morlet",
            "freq_range_hz": [
                float(common["low_freq_hz"]),
                float(common["high_freq_hz"]),
            ],
            "min_cycles": 3.0,
            "max_cycles": None,
            "mt_time_bandwidth_product": 4.0,
            "mt_min_cycles": 3.0,
            "mt_max_cycles": None,
            "freq_smooth_enabled": True,
            "freq_smooth_sigma": 1.5,
            "time_smooth_enabled": True,
            "time_smooth_kernel_size": _default_periodic_time_smooth_kernel_size(
                float(common["time_resolution_s"]),
                float(common["hop_s"]),
            ),
            "aperiodic_mode": "fixed",
            "peak_width_limits_hz": [2.0, 12.0],
            "max_n_peaks": "inf",
            "min_peak_height": 0.0,
            "peak_threshold": 2.0,
            "fit_qc_threshold": 0.6,
        }
    if metric_key in TENSOR_UNDIRECTED_METRIC_KEYS | {"trgc"}:
        payload = {
            **common,
            "method": "morlet",
            "mt_time_bandwidth_product": 4.0,
            "mt_min_cycles": 3.0,
            "mt_max_cycles": None,
            "min_cycles": 3.0,
            "max_cycles": None,
            "gc_n_lags": 20 if metric_key == "trgc" else None,
        }
        if metric_key == "trgc":
            payload["group_by_samples"] = False
            payload["round_ms"] = 50.0
        return payload
    if metric_key == "psi":
        return {
            **default_tensor_metric_notch_params(),
            "freq_step_hz": float(step_hz),
            "bands": [
                dict(item) for item in self._load_tensor_metric_bands_defaults("psi")
            ],
            "method": "morlet",
            "mt_time_bandwidth_product": 4.0,
            "mt_min_cycles": 3.0,
            "mt_max_cycles": None,
            "time_resolution_s": 0.5,
            "hop_s": 0.025,
            "min_cycles": 3.0,
            "max_cycles": None,
        }
    if metric_key == "burst":
        return {
            **default_tensor_metric_notch_params(),
            "bands": [
                dict(item) for item in self._load_tensor_metric_bands_defaults("burst")
            ],
            "percentile": 75.0,
            "baseline_keep": None,
            "boundary_isolated_filter": True,
            "method": "hilbert",
            "hilbert_filter_method": "iir",
            "freq_step_hz": 1.0,
            "morlet_n_cycles": 6.0,
            "mt_n_cycles": 7.0,
            "mt_time_bandwidth_product": 4.0,
            "hilbert_edge_tolerance_pct": 10.0,
            "min_cycles": 2.0,
            "max_cycles": None,
            "hop_s": None,
            "decim": 1,
            "thresholds_source_path": None,
            "thresholds": None,
        }
    return dict(common)


def _load_tensor_metric_defaults_payload(self) -> dict[str, Any]:
    payload = self._config_store.read_yaml("tensor.yml", default={})
    if not isinstance(payload, dict):
        return {}
    defaults = payload.get(TENSOR_METRIC_DEFAULTS_KEY)
    if not isinstance(defaults, dict):
        return {}
    return defaults


def _load_tensor_metric_default_params(
    self,
    metric_key: str,
    *,
    context: RecordContext | None,
) -> dict[str, Any]:
    base = self._default_tensor_metric_params(metric_key, context=context)
    built_in = dict(base)
    defaults_payload = self._load_tensor_metric_defaults_payload()
    node = defaults_payload.get(metric_key)
    explicit_node = node if isinstance(node, dict) else {}
    legacy_notch_fields = False
    if isinstance(node, dict):
        normalized_node = dict(node)
        legacy_notch_fields = "notch_widths" in normalized_node
        legacy_notch_radii = normalized_node.pop("notch_widths", None)
        if legacy_notch_fields and "notch_radii" not in normalized_node:
            normalized_node["notch_radii"] = legacy_notch_radii
        accepted: dict[str, Any] = {}
        repaired: list[str] = []
        ordered_keys = [key for key in built_in if key in normalized_node]
        ordered_keys.extend(key for key in normalized_node if key not in built_in)
        for key in ordered_keys:
            value = normalized_node[key]
            candidate = _deep_merge_dict(base, {**accepted, key: value})
            try:
                validate_metric_storage_params(
                    metric_key=metric_key,
                    metric_label=metric_key,
                    metric_params=candidate,
                )
            except (TypeError, ValueError):
                if key in built_in:
                    repaired.append(key)
                    continue
                accepted[key] = value
            else:
                accepted[key] = value
        explicit_node = accepted
        base = _deep_merge_dict(base, accepted)
        if repaired:
            report_key = (metric_key, tuple(sorted(repaired)))
            reported = getattr(self, "_reported_tensor_default_repairs", set())
            if report_key not in reported:
                reported = set(reported)
                reported.add(report_key)
                setattr(self, "_reported_tensor_default_repairs", reported)
                status_bar_getter = getattr(self, "statusBar", None)
                if callable(status_bar_getter):
                    status_bar_getter().showMessage(
                        "Restored invalid Tensor defaults for "
                        f"{metric_key}: {', '.join(sorted(repaired))}."
                    )
    base.update(
        build_tensor_metric_notch_payload(
            base.get("notches"),
            base.get("notch_radii"),
            legacy_mismatched_list_broadcast=legacy_notch_fields,
        )
    )
    if metric_key in {"psi", "burst"}:
        if "bands" in explicit_node and explicit_node.get("bands") is not None:
            base["bands"] = explicit_node.get("bands")
        else:
            base["bands"] = [
                dict(item)
                for item in self._load_tensor_metric_bands_defaults(metric_key)
            ]
    if metric_key == "burst":
        base["hop_s"] = BURST_NATIVE_HOP_S
        base["decim"] = BURST_NATIVE_DECIM
        base.pop("thresholds_path", None)
        base.pop("thresholds_artifact_path", None)
        try:
            if base.get("thresholds") is not None:
                base["thresholds"] = normalize_burst_threshold_payload(
                    base.get("thresholds")
                )
        except ValueError as exc:
            base["thresholds"] = None
            base["thresholds_source_path"] = None
            self.statusBar().showMessage(
                "Ignored invalid saved Burst thresholds: " + str(exc)
            )
    if metric_key == "periodic_aperiodic":
        base.pop("smooth_enabled", None)
        base.pop("kernel_size", None)
    return base


def _save_tensor_metric_default_params(
    self,
    metric_key: str,
    params: dict[str, Any],
) -> None:
    payload = self._config_store.read_yaml("tensor.yml", default={})
    if not isinstance(payload, dict):
        payload = {}
    defaults = payload.get(TENSOR_METRIC_DEFAULTS_KEY)
    if not isinstance(defaults, dict):
        defaults = {}
    serialized = dict(params)
    serialized.update(
        build_tensor_metric_notch_payload(
            serialized.get("notches"),
            serialized.get("notch_radii"),
        )
    )
    if "selected_channels" in serialized:
        serialized["selected_channels"] = [
            str(item)
            for item in self._coerce_tensor_channels(
                serialized.get("selected_channels")
            )
        ]
    mode = self._tensor_metric_pair_mode(metric_key)
    if mode is not None and "selected_pairs" in serialized:
        serialized["selected_pairs"] = [
            [source, target]
            for source, target in self._coerce_tensor_pairs(
                serialized.get("selected_pairs"),
                directed=(mode == "directed"),
            )
        ]
    if metric_key in {"psi", "burst"}:
        serialized["bands"] = [
            dict(item)
            for item in self._normalize_tensor_bands_rows(serialized.get("bands"))
        ]
    if metric_key == "burst":
        serialized["hop_s"] = BURST_NATIVE_HOP_S
        serialized["decim"] = BURST_NATIVE_DECIM
        serialized.pop("thresholds_path", None)
        serialized.pop("thresholds_source_path", None)
        serialized.pop("thresholds_artifact_path", None)
        if serialized.get("thresholds") is not None:
            serialized["thresholds"] = normalize_burst_threshold_payload(
                serialized.get("thresholds")
            )
    if metric_key == "periodic_aperiodic":
        serialized.pop("smooth_enabled", None)
        serialized.pop("kernel_size", None)
    defaults[metric_key] = serialized
    payload[TENSOR_METRIC_DEFAULTS_KEY] = defaults
    self._config_store.write_yaml("tensor.yml", payload)


def _tensor_metric_default_override_node(self, metric_key: str) -> dict[str, Any]:
    payload = self._load_tensor_metric_defaults_payload()
    node = payload.get(metric_key)
    if not isinstance(node, dict):
        return {}
    return node


def _tensor_effective_metric_defaults(
    self,
    metric_key: str,
    *,
    context: RecordContext | None,
    available_channels: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    channels = (
        tuple(available_channels)
        if available_channels is not None
        else tuple(self._tensor_available_channels)
    )
    defaults = self._load_tensor_metric_default_params(metric_key, context=context)
    if self._tensor_metric_requires_channel_selector(metric_key):
        defaults["selected_channels"] = list(
            self._tensor_default_selected_channels_for_metric(
                metric_key,
                available_channels=channels,
            )
        )
    mode = self._tensor_metric_pair_mode(metric_key)
    if mode is not None:
        directed = mode == "directed"
        defaults["selected_pairs"] = [
            [source, target]
            for source, target in self._tensor_default_selected_pairs_for_metric(
                metric_key,
                directed=directed,
                available_channels=channels,
            )
        ]
    return defaults


def _tensor_prepare_metric_default_payload(
    self,
    metric_key: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    prepared = dict(payload)
    prepared.update(
        build_tensor_metric_notch_payload(
            prepared.get("notches"),
            prepared.get("notch_radii"),
        )
    )

    if self._tensor_metric_requires_channel_selector(metric_key):
        if "selected_channels" in prepared:
            channels = self._coerce_tensor_channels(prepared.get("selected_channels"))
        else:
            channels = self._tensor_selected_channels_by_metric.get(metric_key, ())
        prepared["selected_channels"] = [str(item) for item in channels]

    mode = self._tensor_metric_pair_mode(metric_key)
    if mode is not None:
        directed = mode == "directed"
        if "selected_pairs" in prepared:
            pairs = self._coerce_tensor_pairs(
                prepared.get("selected_pairs"), directed=directed
            )
        else:
            pairs = self._tensor_selected_pairs_by_metric.get(metric_key, ())
        prepared["selected_pairs"] = [[source, target] for source, target in pairs]

    if metric_key in {"psi", "burst"}:
        if "bands" not in prepared:
            current = self._tensor_metric_params.get(metric_key, {})
            if "bands" in current and current.get("bands") is not None:
                prepared["bands"] = current.get("bands")
            else:
                prepared["bands"] = [
                    dict(item)
                    for item in self._load_tensor_metric_bands_defaults(metric_key)
                ]
        elif prepared.get("bands") is None:
            prepared["bands"] = [
                dict(item)
                for item in self._load_tensor_metric_bands_defaults(metric_key)
            ]

    return prepared
