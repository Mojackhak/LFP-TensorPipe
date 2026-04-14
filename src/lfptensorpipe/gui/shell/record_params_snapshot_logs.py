"""Record-parameter log-priority snapshot MainWindow methods."""

from __future__ import annotations

from lfptensorpipe.app import load_alignment_epoch_picks
from lfptensorpipe.gui.shell.common import (
    Any,
    Path,
    PathResolver,
    RecordContext,
    TENSOR_METRICS,
    _deep_merge_dict,
    _nested_get,
    alignment_paradigm_log_path,
    load_annotations_csv_rows,
    read_run_log,
    tensor_metric_log_path,
)


class MainWindowRecordParamsSnapshotLogsMixin:
    @staticmethod
    def _read_completed_log_params(path: Path) -> dict[str, Any]:
        try:
            payload = read_run_log(path)
        except Exception:
            return {}
        if payload is None or not bool(payload.get("completed")):
            return {}
        params = payload.get("params", {})
        return dict(params) if isinstance(params, dict) else {}

    def _build_master_record_params_snapshot(
        self, context: RecordContext
    ) -> dict[str, Any]:
        return {
            "preproc": {
                "filter": {
                    "basic": dict(self._load_filter_basic_defaults()),
                    "advance": dict(self._load_filter_advance_defaults()),
                },
                "annotations": {"rows": []},
                "ecg": {
                    "method": "svd",
                    "selected_channels": [],
                },
                "viz": {
                    "psd_params": dict(self._load_preproc_viz_psd_defaults()),
                    "tfr_params": dict(self._load_preproc_viz_tfr_defaults()),
                    "selected_channels": [],
                    "selected_step": None,
                },
            },
            "tensor": {
                "selected_metrics": [],
                "active_metric": self._tensor_active_metric_key,
                "metric_params": {
                    spec.key: self._tensor_effective_metric_defaults(
                        spec.key,
                        context=context,
                        available_channels=self._tensor_available_channels,
                    )
                    for spec in TENSOR_METRICS
                },
                "bands": {
                    "psi": self._load_tensor_metric_bands_defaults("psi"),
                    "burst": self._load_tensor_metric_bands_defaults("burst"),
                },
                "mask_edge_effects": True,
            },
            "alignment": {
                "trial_slug": self._current_alignment_paradigm_slug(),
                "method": None,
                "sample_rate": None,
                "epoch_metric": None,
                "picked_epoch_indices": [],
            },
            "features": {
                "paradigm_slug": self._shared_stage_trial_slug(),
                "trial_params_by_slug": {},
            },
            "localize": {
                "atlas": None,
                "match": dict(self._localize_match_payload or {}),
            },
        }

    def _merge_preproc_logs_into_snapshot(
        self, snapshot: dict[str, Any], resolver: PathResolver
    ) -> None:
        filter_params = self._read_completed_log_params(
            resolver.preproc_root / "filter" / "lfptensorpipe_log.json"
        )
        if filter_params:
            basic = dict(snapshot["preproc"]["filter"]["basic"])
            if "notches" in filter_params and isinstance(
                filter_params["notches"], list
            ):
                basic["notches"] = [float(item) for item in filter_params["notches"]]
            if "low_freq" in filter_params:
                basic["l_freq"] = self._safe_float(
                    filter_params.get("low_freq"), basic["l_freq"]
                )
            if "high_freq" in filter_params:
                basic["h_freq"] = self._safe_float(
                    filter_params.get("high_freq"), basic["h_freq"]
                )
            snapshot["preproc"]["filter"]["basic"] = basic
            advance = dict(snapshot["preproc"]["filter"]["advance"])
            for key in (
                "notch_widths",
                "epoch_dur",
                "p2p_thresh",
                "autoreject_correct_factor",
            ):
                if key in filter_params:
                    advance[key] = filter_params[key]
            snapshot["preproc"]["filter"]["advance"] = advance

        annotations_csv = resolver.preproc_root / "annotations" / "annotations.csv"
        if annotations_csv.is_file():
            ok_rows, rows, _ = load_annotations_csv_rows(annotations_csv)
            if ok_rows:
                snapshot["preproc"]["annotations"]["rows"] = rows

        ecg_params = self._read_completed_log_params(
            resolver.preproc_root / "ecg_artifact_removal" / "lfptensorpipe_log.json"
        )
        if ecg_params:
            method = ecg_params.get("method")
            picks = ecg_params.get("picks")
            if isinstance(method, str):
                snapshot["preproc"]["ecg"]["method"] = method
            if isinstance(picks, list):
                snapshot["preproc"]["ecg"]["selected_channels"] = [
                    str(item) for item in picks if str(item).strip()
                ]

    def _merge_tensor_logs_into_snapshot(
        self, snapshot: dict[str, Any], resolver: PathResolver
    ) -> None:
        selected_tensor_metrics: list[str] = []
        for metric_key in sorted(self._tensor_metric_checks.keys()):
            metric_params = self._read_completed_log_params(
                tensor_metric_log_path(resolver, metric_key)
            )
            if not metric_params:
                continue
            selected_tensor_metrics.append(metric_key)
            current_params = dict(
                _nested_get(snapshot, ("tensor", "metric_params", metric_key)) or {}
            )
            if "low_freq" in metric_params:
                current_params["low_freq_hz"] = self._safe_float(
                    metric_params.get("low_freq"),
                    current_params.get("low_freq_hz", 1.0),
                )
            if "high_freq" in metric_params:
                current_params["high_freq_hz"] = self._safe_float(
                    metric_params.get("high_freq"),
                    current_params.get("high_freq_hz", 100.0),
                )
            if "step_hz" in metric_params:
                current_params["freq_step_hz"] = self._safe_float(
                    metric_params.get("step_hz"),
                    current_params.get("freq_step_hz", 0.5),
                )
            for key, value in metric_params.items():
                if key in {"low_freq", "high_freq", "step_hz"}:
                    continue
                current_params[key] = value
            channels = metric_params.get("selected_channels")
            if isinstance(channels, list):
                current_params["selected_channels"] = [
                    str(item) for item in channels if str(item).strip()
                ]
            pairs = metric_params.get("selected_pairs")
            if isinstance(pairs, list):
                current_params["selected_pairs"] = [
                    [str(pair[0]), str(pair[1])]
                    for pair in pairs
                    if isinstance(pair, (list, tuple)) and len(pair) == 2
                ]
            if metric_key in {"psi", "burst"} and "bands" in metric_params:
                bands = self._normalize_tensor_bands_rows(metric_params.get("bands"))
                if bands:
                    current_params["bands"] = [dict(item) for item in bands]
            snapshot["tensor"]["metric_params"][metric_key] = current_params
        if selected_tensor_metrics:
            snapshot["tensor"]["selected_metrics"] = selected_tensor_metrics

    def _merge_alignment_and_features_logs_into_snapshot(
        self, snapshot: dict[str, Any], resolver: PathResolver
    ) -> None:
        alignment_slug = snapshot["alignment"].get("trial_slug")
        if not isinstance(alignment_slug, str) or not alignment_slug:
            legacy_alignment_slug = snapshot["alignment"].get("paradigm_slug")
            if isinstance(legacy_alignment_slug, str) and legacy_alignment_slug:
                alignment_slug = legacy_alignment_slug
        if not isinstance(alignment_slug, str) or not alignment_slug:
            return

        alignment_params = self._read_completed_log_params(
            alignment_paradigm_log_path(resolver, alignment_slug)
        )
        if alignment_params and "sample_rate" in alignment_params:
            try:
                snapshot["alignment"]["sample_rate"] = float(
                    alignment_params["sample_rate"]
                )
            except Exception:
                pass

    def _merge_alignment_epoch_picks_into_snapshot(
        self, snapshot: dict[str, Any], resolver: PathResolver
    ) -> None:
        alignment_slug = snapshot.get("alignment", {}).get("trial_slug")
        if not isinstance(alignment_slug, str) or not alignment_slug:
            return
        persisted_picks = load_alignment_epoch_picks(
            resolver.context,
            paradigm_slug=alignment_slug,
        )
        if persisted_picks is None:
            return
        snapshot.setdefault("alignment", {})["picked_epoch_indices"] = list(
            persisted_picks
        )

    def _merge_localize_logs_into_snapshot(
        self, snapshot: dict[str, Any], resolver: PathResolver
    ) -> None:
        localize_params = self._read_completed_log_params(
            resolver.lfp_root / "localize" / "lfptensorpipe_log.json"
        )
        if not localize_params:
            return
        atlas = localize_params.get("atlas")
        if isinstance(atlas, str):
            snapshot["localize"]["atlas"] = atlas
        selected_regions = localize_params.get("selected_regions_signature")
        if isinstance(selected_regions, list):
            snapshot["localize"]["selected_regions"] = [
                str(item).strip() for item in selected_regions if str(item).strip()
            ]

    def _build_log_priority_snapshot(
        self, context: RecordContext, *, include_master: bool
    ) -> dict[str, Any]:
        resolver = PathResolver(context)
        snapshot = (
            self._build_master_record_params_snapshot(context)
            if include_master
            else self._collect_record_params_snapshot()
        )
        self._merge_preproc_logs_into_snapshot(snapshot, resolver)
        self._merge_tensor_logs_into_snapshot(snapshot, resolver)
        self._merge_alignment_and_features_logs_into_snapshot(snapshot, resolver)
        self._merge_localize_logs_into_snapshot(snapshot, resolver)

        if include_master:
            ok_master, master_params, _ = self._load_record_params_payload(context)
            if ok_master:
                snapshot = _deep_merge_dict(snapshot, master_params)
            else:
                self.statusBar().showMessage("log读取失败，加载默认值")
        self._merge_alignment_epoch_picks_into_snapshot(snapshot, resolver)
        return snapshot
