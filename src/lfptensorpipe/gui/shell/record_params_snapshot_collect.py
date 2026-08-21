"""Record-parameter snapshot collection MainWindow methods."""

from __future__ import annotations

from copy import deepcopy

from lfptensorpipe.gui.shell.common import (
    Any,
    np,
)


class MainWindowRecordParamsSnapshotCollectMixin:
    @staticmethod
    def _safe_float(value: Any, fallback: float) -> float:
        try:
            parsed = float(value)
        except Exception:
            return float(fallback)
        if not np.isfinite(parsed):
            return float(fallback)
        return float(parsed)

    @staticmethod
    def _draft_number(value: Any) -> float | str | None:
        token = str(value).strip() if value is not None else ""
        if not token:
            return None
        try:
            parsed = float(token)
        except Exception:
            return token
        return parsed if np.isfinite(parsed) else token

    @classmethod
    def _draft_number_list(cls, value: Any) -> list[float] | str:
        token = str(value).strip() if value is not None else ""
        if not token:
            return []
        parts = [item.strip() for item in token.split(",")]
        if any(not item for item in parts):
            return token
        parsed = [cls._draft_number(item) for item in parts]
        if any(not isinstance(item, float) for item in parsed):
            return token
        return [float(item) for item in parsed]

    def _collect_annotations_rows_for_params(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        if self._preproc_annotations_table is None:
            return rows
        for row_idx in range(self._preproc_annotations_table.rowCount()):
            description_item = self._preproc_annotations_table.item(row_idx, 0)
            onset_item = self._preproc_annotations_table.item(row_idx, 1)
            duration_item = self._preproc_annotations_table.item(row_idx, 2)
            description = (
                description_item.text() if description_item is not None else ""
            ).strip()
            onset = (onset_item.text() if onset_item is not None else "").strip()
            duration = (
                duration_item.text() if duration_item is not None else ""
            ).strip()
            if not description and not onset and not duration:
                continue
            rows.append(
                {
                    "description": description,
                    "onset": onset,
                    "duration": duration,
                }
            )
        return rows

    def _collect_tensor_bands_for_params(self) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {"psi": [], "burst": []}
        for metric_key in ("psi", "burst"):
            params = self._tensor_metric_params.get(metric_key, {})
            out[metric_key] = [
                dict(item)
                for item in self._normalize_tensor_bands_rows(params.get("bands"))
            ]
        return out

    def _collect_preproc_record_params_snapshot(self) -> dict[str, Any]:
        basic_defaults = self._load_filter_basic_defaults()
        notches_text = (
            self._preproc_filter_notches_edit.text().strip()
            if self._preproc_filter_notches_edit is not None
            else self._format_filter_notches(basic_defaults.get("notches", []))
        )
        low_freq_value = (
            self._preproc_filter_low_freq_edit.text().strip()
            if self._preproc_filter_low_freq_edit is not None
            else basic_defaults.get("l_freq")
        )
        high_freq_value = (
            self._preproc_filter_high_freq_edit.text().strip()
            if self._preproc_filter_high_freq_edit is not None
            else basic_defaults.get("h_freq")
        )
        basic = {
            "notches": self._draft_number_list(notches_text),
            "l_freq": self._draft_number(low_freq_value),
            "h_freq": self._draft_number(high_freq_value),
        }
        ecg_params_by_method = deepcopy(self._preproc_ecg_params_by_method)
        return {
            "filter": {
                "basic": dict(basic),
                "advance": dict(self._preproc_filter_advance_params),
            },
            "annotations": {
                "rows": self._collect_annotations_rows_for_params(),
            },
            "ecg": {
                "method": str(
                    self._preproc_ecg_method_combo.currentData()
                    if self._preproc_ecg_method_combo is not None
                    else "svd"
                ),
                "selected_channels": list(self._preproc_ecg_selected_channels),
                "params_by_method": deepcopy(ecg_params_by_method),
                "review": dict(self._preproc_ecg_review_params),
            },
            "viz": {
                "psd_params": dict(self._preproc_viz_psd_params),
                "tfr_params": dict(self._preproc_viz_tfr_params),
                "selected_channels": list(self._preproc_viz_selected_channels),
                "selected_step": self._current_preproc_viz_step(),
            },
            "step_params": {
                "filter": {
                    "basic": dict(basic),
                    "advance": dict(self._preproc_filter_advance_params),
                },
                "ecg": {
                    "method": str(
                        self._preproc_ecg_method_combo.currentData()
                        if self._preproc_ecg_method_combo is not None
                        else "svd"
                    ),
                    "selected_channels": list(self._preproc_ecg_selected_channels),
                    "params_by_method": deepcopy(ecg_params_by_method),
                    "review": dict(self._preproc_ecg_review_params),
                },
                "viz": {
                    "psd_params": dict(self._preproc_viz_psd_params),
                    "tfr_params": dict(self._preproc_viz_tfr_params),
                    "selected_channels": list(self._preproc_viz_selected_channels),
                    "selected_step": self._current_preproc_viz_step(),
                },
            },
        }

    def _collect_tensor_record_params_snapshot(self) -> dict[str, Any]:
        active_metric = self._tensor_active_metric_key
        cpu_text = (
            self._tensor_cpu_percent_edit.text().strip()
            if self._tensor_cpu_percent_edit is not None
            else ""
        )
        if not cpu_text:
            cpu_percent: Any = None
        else:
            try:
                parsed_cpu = float(cpu_text)
            except (TypeError, ValueError):
                cpu_percent = cpu_text
            else:
                cpu_percent = parsed_cpu if np.isfinite(parsed_cpu) else cpu_text
        return {
            "selected_metrics": self._selected_tensor_metrics_snapshot(),
            "active_metric": active_metric,
            "metric_params": {
                key: dict(value) for key, value in self._tensor_metric_params.items()
            },
            "bands": self._collect_tensor_bands_for_params(),
            "mask_edge_effects": bool(
                self._tensor_mask_edge_checkbox.isChecked()
                if self._tensor_mask_edge_checkbox is not None
                else True
            ),
            "cpu_percent": cpu_percent,
        }

    def _collect_alignment_record_params_snapshot(self) -> dict[str, Any]:
        paradigm = self._current_alignment_paradigm()
        alignment_method = (
            self._alignment_method_combo.currentData()
            if self._alignment_method_combo is not None
            else None
        )
        alignment_epoch_metric = (
            self._alignment_epoch_metric_combo.currentData()
            if self._alignment_epoch_metric_combo is not None
            else None
        )
        alignment_epoch_channel = (
            self._alignment_epoch_channel_combo.currentData()
            if self._alignment_epoch_channel_combo is not None
            else None
        )
        method_params = (
            deepcopy(paradigm.get("method_params", {}))
            if isinstance(paradigm, dict)
            and isinstance(paradigm.get("method_params"), dict)
            else {}
        )
        raw_cache = (
            paradigm.get("method_params_by_method", {})
            if isinstance(paradigm, dict)
            else {}
        )
        method_params_by_method = (
            {
                str(key): deepcopy(value)
                for key, value in raw_cache.items()
                if isinstance(key, str) and isinstance(value, dict)
            }
            if isinstance(raw_cache, dict)
            else {}
        )
        return {
            "trial_slug": self._current_alignment_paradigm_slug(),
            "method": (
                str(alignment_method) if isinstance(alignment_method, str) else None
            ),
            "method_params": method_params,
            "method_params_by_method": method_params_by_method,
            "sample_rate": (
                float(self._alignment_n_samples_edit.text().strip())
                if self._alignment_n_samples_edit is not None
                and self._alignment_n_samples_edit.text().strip()
                else None
            ),
            "epoch_metric": (
                str(alignment_epoch_metric)
                if isinstance(alignment_epoch_metric, str)
                else None
            ),
            "epoch_channel": (
                int(alignment_epoch_channel)
                if isinstance(alignment_epoch_channel, (int, np.integer))
                else None
            ),
            "picked_epoch_indices": self._collect_alignment_pick_indices(),
        }

    def _collect_features_record_params_snapshot(self) -> dict[str, Any]:
        current_slug = self._current_features_paradigm_slug()
        if isinstance(current_slug, str) and current_slug:
            self._features_trial_params_by_slug[current_slug] = (
                self._collect_current_features_trial_params()
            )
        trial_params_by_slug: dict[str, dict[str, Any]] = {}
        for slug, node in self._features_trial_params_by_slug.items():
            if (
                not isinstance(slug, str)
                or not slug.strip()
                or not isinstance(node, dict)
            ):
                continue
            normalized = self._normalize_features_trial_params(slug, node)
            for key in ("active_metric", "selected_relative_stem"):
                if normalized.get(key) == "":
                    normalized[key] = None
            for mapping_key in ("subset", "filters", "plot_labels"):
                mapping = normalized.get(mapping_key)
                if not isinstance(mapping, dict):
                    continue
                normalized[mapping_key] = {
                    key: (None if value == "" else value)
                    for key, value in mapping.items()
                }
            trial_params_by_slug[slug] = normalized
        return {
            "paradigm_slug": self._shared_stage_trial_slug(),
            "trial_params_by_slug": trial_params_by_slug,
        }

    def _collect_localize_record_params_snapshot(self) -> dict[str, Any]:
        return {
            "atlas": self._localize_selected_atlas,
            "selected_regions": list(self._localize_selected_regions),
            "match": dict(self._localize_match_payload or {}),
            "params": {
                "atlas": self._localize_selected_atlas,
                "selected_regions": list(self._localize_selected_regions),
                "match": dict(self._localize_match_payload or {}),
            },
        }

    def _collect_record_params_snapshot(self) -> dict[str, Any]:
        self._commit_active_tensor_panel_to_params()
        self._sync_tensor_selector_maps_into_metric_params()
        return {
            "preproc": self._collect_preproc_record_params_snapshot(),
            "tensor": self._collect_tensor_record_params_snapshot(),
            "alignment": self._collect_alignment_record_params_snapshot(),
            "features": self._collect_features_record_params_snapshot(),
            "localize": self._collect_localize_record_params_snapshot(),
        }
