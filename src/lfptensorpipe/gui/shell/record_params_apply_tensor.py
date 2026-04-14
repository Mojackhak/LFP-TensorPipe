"""Record-parameter tensor apply MainWindow methods."""

from __future__ import annotations

from lfptensorpipe.gui.shell.common import (
    Any,
    RecordContext,
    TENSOR_CHANNEL_METRIC_KEYS,
    TENSOR_DIRECTED_METRIC_KEYS,
    TENSOR_METRICS,
    TENSOR_UNDIRECTED_METRIC_KEYS,
    _deep_merge_dict,
    _nested_get,
    build_tensor_metric_notch_payload,
)


class MainWindowRecordParamsApplyTensorMixin:
    def _apply_record_params_tensor_snapshot(
        self,
        context: RecordContext,
        snapshot: dict[str, Any],
    ) -> int:
        skipped = 0

        if "tensor.selected_metrics" not in self._record_param_dirty_keys:
            selected_metrics = _nested_get(snapshot, ("tensor", "selected_metrics"))
            if isinstance(selected_metrics, list):
                selected_set = {
                    str(item) for item in selected_metrics if str(item).strip()
                }
                for metric_key, checkbox in self._tensor_metric_checks.items():
                    checkbox.setChecked(metric_key in selected_set)
        else:
            skipped += 1

        if "tensor.metric_params" not in self._record_param_dirty_keys:
            metric_params = _nested_get(snapshot, ("tensor", "metric_params"))
            if isinstance(metric_params, dict):
                merged_params: dict[str, dict[str, Any]] = {}
                for spec in TENSOR_METRICS:
                    default_params = self._tensor_effective_metric_defaults(
                        spec.key,
                        context=context,
                        available_channels=self._tensor_available_channels,
                    )
                    node = metric_params.get(spec.key)
                    if isinstance(node, dict):
                        merged = _deep_merge_dict(default_params, node)
                    else:
                        merged = default_params
                    if spec.key == "periodic_aperiodic":
                        merged.pop("smooth_enabled", None)
                        merged.pop("kernel_size", None)
                    merged.update(
                        build_tensor_metric_notch_payload(
                            merged.get("notches"),
                            merged.get("notch_widths"),
                        )
                    )
                    merged_params[spec.key] = merged
                self._tensor_metric_params = merged_params
                for spec in TENSOR_METRICS:
                    params = self._tensor_metric_params.get(spec.key, {})
                    if spec.key in TENSOR_CHANNEL_METRIC_KEYS:
                        channels = params.get("selected_channels")
                        if isinstance(channels, list):
                            self._tensor_selected_channels_by_metric[spec.key] = tuple(
                                str(item) for item in channels if str(item).strip()
                            )
                    if spec.key in (
                        TENSOR_UNDIRECTED_METRIC_KEYS | TENSOR_DIRECTED_METRIC_KEYS
                    ):
                        pairs = params.get("selected_pairs")
                        directed = spec.key in TENSOR_DIRECTED_METRIC_KEYS
                        if isinstance(pairs, list):
                            normalized_pairs = self._filter_tensor_pairs(
                                [
                                    (str(item[0]), str(item[1]))
                                    for item in pairs
                                    if isinstance(item, (list, tuple))
                                    and len(item) == 2
                                ],
                                available_channels=self._tensor_available_channels,
                                directed=directed,
                            )
                            self._tensor_selected_pairs_by_metric[spec.key] = (
                                normalized_pairs
                            )
        else:
            skipped += 1

        if "tensor.active_metric" not in self._record_param_dirty_keys:
            metric_key = _nested_get(snapshot, ("tensor", "active_metric"))
            if isinstance(metric_key, str) and metric_key:
                self._tensor_active_metric_key = metric_key
        else:
            skipped += 1

        if "tensor.mask_edge_effects" not in self._record_param_dirty_keys:
            mask_edge = _nested_get(snapshot, ("tensor", "mask_edge_effects"))
            if (
                isinstance(mask_edge, bool)
                and self._tensor_mask_edge_checkbox is not None
            ):
                self._tensor_mask_edge_checkbox.setChecked(mask_edge)
        else:
            skipped += 1

        return skipped
