"""Features trial discovery and indicator MainWindow methods."""

from __future__ import annotations

from lfptensorpipe.gui.shell.common import (
    Any,
    PathResolver,
    QColor,
    QListWidgetItem,
    Qt,
    _deep_merge_dict,
)


class MainWindowFeaturesTrialsMixin:
    def _features_metric_keys_for_trial_slug(self, slug: str | None) -> list[str]:
        context = self._record_context()
        if context is None or not isinstance(slug, str):
            return []
        token = slug.strip()
        if not token:
            return []
        resolver = PathResolver(context)
        root = resolver.alignment_root / token
        if not root.exists():
            return []
        metrics: list[str] = []
        seen: set[str] = set()
        for path in sorted(root.glob("*/na-raw.pkl")):
            metric_key = path.parent.name
            if metric_key and metric_key not in seen:
                seen.add(metric_key)
                metrics.append(metric_key)
        return metrics

    def _default_features_trial_params(self) -> dict[str, Any]:
        return {
            "active_metric": "",
            "axes_by_metric": {},
            "selected_relative_stem": "",
            "subset": {"band": "", "channel": "", "region": ""},
            "filters": {"feature": ""},
            "plot_labels": {"x": "", "y": "", "cbar": ""},
            "plot_advance": dict(self._load_features_plot_advance_defaults()),
        }

    def _normalize_features_trial_params(
        self,
        trial_slug: str | None,
        node: Any,
    ) -> dict[str, Any]:
        defaults = self._default_features_trial_params()
        source = node if isinstance(node, dict) else {}
        metric_keys = self._features_metric_keys_for_trial_slug(trial_slug)
        valid_metric_keys = set(metric_keys)
        raw_axes_by_metric = source.get("axes_by_metric")
        normalized_axes: dict[str, dict[str, list[dict[str, Any]]]] = {}
        if isinstance(raw_axes_by_metric, dict):
            for raw_metric_key, raw_axis_node in raw_axes_by_metric.items():
                metric_key = str(raw_metric_key).strip()
                if not metric_key:
                    continue
                if valid_metric_keys and metric_key not in valid_metric_keys:
                    continue
                if not isinstance(raw_axis_node, dict):
                    continue
                if self._features_metric_uses_auto_bands(metric_key):
                    bands: list[dict[str, Any]] = []
                else:
                    bands = [
                        dict(item)
                        for item in self._normalize_feature_axis_rows(
                            raw_axis_node.get("bands"),
                            min_start=0.0,
                            max_end=None,
                            allow_duplicate_names=False,
                        )
                    ]
                times = [
                    dict(item)
                    for item in self._normalize_feature_axis_rows(
                        raw_axis_node.get("times"),
                        min_start=0.0,
                        max_end=100.0,
                        allow_duplicate_names=True,
                    )
                ]
                normalized_axes[metric_key] = {
                    "bands": bands,
                    "times": times,
                }

        active_metric = str(source.get("active_metric", "")).strip()
        ordered_metric_keys = list(metric_keys)
        if not ordered_metric_keys:
            ordered_metric_keys = list(normalized_axes.keys())
        if active_metric and active_metric not in ordered_metric_keys:
            active_metric = ""
        if not active_metric and ordered_metric_keys:
            active_metric = ordered_metric_keys[0]

        filters = source.get("filters")
        plot_labels = source.get("plot_labels")
        plot_advance = source.get("plot_advance")
        selected_relative_stem = str(source.get("selected_relative_stem", "")).strip()
        return {
            "active_metric": active_metric,
            "axes_by_metric": normalized_axes,
            "selected_relative_stem": selected_relative_stem,
            "subset": self._normalize_features_subset_selection(
                source.get("subset"),
            ),
            "filters": {
                "feature": (
                    str(filters.get("feature", "")).strip()
                    if isinstance(filters, dict)
                    else ""
                ),
            },
            "plot_labels": {
                "x": (
                    str(plot_labels.get("x", "")).strip()
                    if isinstance(plot_labels, dict)
                    else ""
                ),
                "y": (
                    str(plot_labels.get("y", "")).strip()
                    if isinstance(plot_labels, dict)
                    else ""
                ),
                "cbar": (
                    str(plot_labels.get("cbar", "")).strip()
                    if isinstance(plot_labels, dict)
                    else ""
                ),
            },
            "plot_advance": (
                _deep_merge_dict(defaults["plot_advance"], plot_advance)
                if isinstance(plot_advance, dict)
                else dict(defaults["plot_advance"])
            ),
        }

    def _normalize_features_trial_params_map(
        self, node: Any
    ) -> dict[str, dict[str, Any]]:
        if not isinstance(node, dict):
            return {}
        normalized: dict[str, dict[str, Any]] = {}
        for raw_slug, raw_params in node.items():
            slug = str(raw_slug).strip()
            if not slug:
                continue
            normalized[slug] = self._normalize_features_trial_params(slug, raw_params)
        return normalized

    def _legacy_features_trial_params_snapshot(
        self,
        snapshot: dict[str, Any],
    ) -> tuple[str | None, dict[str, Any] | None]:
        features_node = snapshot.get("features")
        if not isinstance(features_node, dict):
            return None, None
        slug = features_node.get("paradigm_slug")
        if not isinstance(slug, str) or not slug.strip():
            alignment_node = snapshot.get("alignment")
            if isinstance(alignment_node, dict):
                slug = alignment_node.get("trial_slug")
        if not isinstance(slug, str) or not slug.strip():
            return None, None
        legacy_keys = {
            "active_metric",
            "axes_by_metric",
            "subset",
            "filters",
            "plot_labels",
            "plot_advance",
        }
        if not any(key in features_node for key in legacy_keys):
            return None, None
        return slug.strip(), self._normalize_features_trial_params(slug, features_node)

    def _collect_current_features_trial_params(
        self,
        trial_slug: str | None = None,
    ) -> dict[str, Any]:
        slug = (
            trial_slug.strip()
            if isinstance(trial_slug, str) and trial_slug.strip()
            else self._current_features_paradigm_slug()
        )
        metric_keys = self._features_metric_keys_for_trial_slug(slug)
        active_metric = self._current_features_axis_metric() or ""
        if active_metric not in metric_keys:
            active_metric = metric_keys[0] if metric_keys else ""
        selected = self._selected_features_file()
        node = {
            "active_metric": active_metric,
            "axes_by_metric": {
                metric_key: {
                    "bands": [
                        dict(item)
                        for item in self._normalized_features_axes_for_metric(
                            metric_key
                        )["bands"]
                    ],
                    "times": [
                        dict(item)
                        for item in self._normalized_features_axes_for_metric(
                            metric_key
                        )["times"]
                    ],
                }
                for metric_key in metric_keys
            },
            "selected_relative_stem": (
                str(selected.get("relative_stem", "")).strip()
                if isinstance(selected, dict)
                else ""
            ),
            "subset": self._current_features_subset_selection(),
            "filters": {
                "feature": (
                    self._features_filter_feature_edit.text().strip()
                    if self._features_filter_feature_edit is not None
                    else ""
                ),
            },
            "plot_labels": {
                "x": (
                    self._features_x_label_edit.text().strip()
                    if self._features_x_label_edit is not None
                    else ""
                ),
                "y": (
                    self._features_y_label_edit.text().strip()
                    if self._features_y_label_edit is not None
                    else ""
                ),
                "cbar": (
                    self._features_cbar_label_edit.text().strip()
                    if self._features_cbar_label_edit is not None
                    else ""
                ),
            },
            "plot_advance": dict(self._features_plot_advance_params),
        }
        return self._normalize_features_trial_params(slug, node)

    def _capture_features_trial_params(self, trial_slug: str | None) -> None:
        if not isinstance(trial_slug, str) or not trial_slug.strip():
            return
        slug = trial_slug.strip()
        self._features_trial_params_by_slug[slug] = (
            self._collect_current_features_trial_params(slug)
        )

    def _load_features_trial_params_from_log(
        self,
        trial_slug: str | None,
    ) -> dict[str, Any] | None:
        context = self._record_context()
        if context is None or not isinstance(trial_slug, str) or not trial_slug.strip():
            return None
        params = self._read_completed_log_params(
            PathResolver(context).features_root
            / trial_slug.strip()
            / "lfptensorpipe_log.json"
        )
        if not params:
            return None
        axes_by_metric = params.get("axes_by_metric")
        if not isinstance(axes_by_metric, dict):
            return None
        return self._normalize_features_trial_params(
            trial_slug.strip(),
            {"axes_by_metric": axes_by_metric},
        )

    def _apply_features_trial_params_to_ui(
        self,
        params: dict[str, Any],
        *,
        respect_dirty_keys: bool,
    ) -> int:
        skipped = 0
        selected_relative_stem = str(params.get("selected_relative_stem", "")).strip()
        if not (
            respect_dirty_keys and "features.axes" in self._record_param_dirty_keys
        ):
            self._features_axes_by_metric = {
                str(metric_key): {
                    "bands": [dict(item) for item in axis_node.get("bands", [])],
                    "times": [dict(item) for item in axis_node.get("times", [])],
                }
                for metric_key, axis_node in params.get("axes_by_metric", {}).items()
                if isinstance(metric_key, str) and isinstance(axis_node, dict)
            }
            self._refresh_features_axis_metric_combo()
            active_metric = str(params.get("active_metric", "")).strip()
            if active_metric and self._features_axis_metric_combo is not None:
                idx = self._features_axis_metric_combo.findData(active_metric)
                if idx >= 0:
                    self._features_axis_metric_combo.setCurrentIndex(idx)
        else:
            skipped += 1

        filters = params.get("filters", {})
        if not (
            respect_dirty_keys and "features.filters" in self._record_param_dirty_keys
        ):
            if self._features_filter_feature_edit is not None:
                self._features_filter_feature_edit.setText(
                    str(filters.get("feature", ""))
                )
        else:
            skipped += 1

        plot_labels = params.get("plot_labels", {})
        if not (
            respect_dirty_keys
            and "features.plot_labels" in self._record_param_dirty_keys
        ):
            if self._features_x_label_edit is not None:
                self._features_x_label_edit.setText(str(plot_labels.get("x", "")))
            if self._features_y_label_edit is not None:
                self._features_y_label_edit.setText(str(plot_labels.get("y", "")))
            if self._features_cbar_label_edit is not None:
                self._features_cbar_label_edit.setText(str(plot_labels.get("cbar", "")))
        else:
            skipped += 1

        if not (
            respect_dirty_keys
            and "features.plot_advance" in self._record_param_dirty_keys
        ):
            defaults = self._load_features_plot_advance_defaults()
            self._features_plot_advance_params = _deep_merge_dict(
                defaults,
                params.get("plot_advance", {}),
            )
        else:
            skipped += 1

        self._refresh_features_available_files(
            preferred_relative_stem=selected_relative_stem or None,
        )

        if not (
            respect_dirty_keys and "features.subset" in self._record_param_dirty_keys
        ):
            self._sync_features_subset_options(
                preferred_selection=params.get("subset"),
            )
        else:
            skipped += 1

        self._refresh_features_controls()
        return skipped

    def _restore_features_trial_params(
        self,
        trial_slug: str | None,
        *,
        respect_dirty_keys: bool,
    ) -> int:
        slug = str(trial_slug).strip() if isinstance(trial_slug, str) else ""
        params: dict[str, Any]
        if slug:
            cached = self._features_trial_params_by_slug.get(slug)
            if isinstance(cached, dict):
                params = self._normalize_features_trial_params(slug, cached)
                self._features_trial_params_by_slug[slug] = params
            else:
                log_params = self._load_features_trial_params_from_log(slug)
                if log_params is not None:
                    params = log_params
                    self._features_trial_params_by_slug[slug] = params
                else:
                    params = self._default_features_trial_params()
        else:
            params = self._default_features_trial_params()
        return self._apply_features_trial_params_to_ui(
            params,
            respect_dirty_keys=respect_dirty_keys,
        )

    def _features_trial_is_selectable(self, slug: str | None) -> bool:
        context = self._record_context()
        if context is None or not isinstance(slug, str) or not slug.strip():
            return False
        resolver = PathResolver(context)
        return (
            self._alignment_trial_stage_state_runtime(
                resolver,
                paradigm_slug=slug.strip(),
            )
            == "green"
        )

    def _features_trial_row_for_slug(self, slug: str | None) -> int:
        if self._features_paradigm_list is None or not isinstance(slug, str):
            return -1
        for idx in range(self._features_paradigm_list.count()):
            item = self._features_paradigm_list.item(idx)
            if item is not None and item.data(Qt.UserRole) == slug:
                return idx
        return -1

    def _sync_features_paradigm_selection(self, slug: str | None) -> int:
        if self._features_paradigm_list is None:
            return -1
        target_slug = slug if self._features_trial_is_selectable(slug) else None
        row = self._features_trial_row_for_slug(target_slug)
        self._features_paradigm_list.blockSignals(True)
        try:
            if row >= 0:
                self._features_paradigm_list.setCurrentRow(row)
            else:
                self._features_paradigm_list.setCurrentRow(-1)
        finally:
            self._features_paradigm_list.blockSignals(False)
        return row

    def _refresh_features_trial_list_states(self) -> None:
        context = self._record_context()
        resolver = PathResolver(context) if context is not None else None
        if self._features_paradigm_list is None:
            return
        for idx in range(self._features_paradigm_list.count()):
            item = self._features_paradigm_list.item(idx)
            if item is None:
                continue
            slug = item.data(Qt.UserRole)
            selectable = False
            if resolver is not None and isinstance(slug, str) and slug.strip():
                selectable = (
                    self._alignment_trial_stage_state_runtime(
                        resolver,
                        paradigm_slug=slug.strip(),
                    )
                    == "green"
                )
            flags = item.flags() | Qt.ItemIsEnabled | Qt.ItemIsSelectable
            if not selectable:
                flags &= ~Qt.ItemIsEnabled
                flags &= ~Qt.ItemIsSelectable
                item.setForeground(QColor("#8A8A8A"))
            else:
                item.setForeground(QColor("#000000"))
            item.setFlags(flags)
        self._sync_features_paradigm_selection(self._shared_stage_trial_slug())

    def _current_features_paradigm_slug(self) -> str | None:
        shared_slug = self._shared_stage_trial_slug()
        if (
            isinstance(shared_slug, str)
            and shared_slug
            and (
                not hasattr(self, "_alignment_trial_stage_state_runtime")
                or self._features_trial_is_selectable(shared_slug)
            )
        ):
            return shared_slug
        if self._features_paradigm_list is None:
            return None
        row = self._features_paradigm_list.currentRow()
        if row < 0 or row >= len(self._features_paradigms):
            return None
        slug = self._features_paradigms[row].get("slug")
        if not isinstance(slug, str):
            return None
        return slug

    def _discover_features_trials(self) -> list[dict[str, Any]]:
        context = self._record_context()
        if context is None:
            return []
        trials: list[dict[str, Any]] = []
        source = self._alignment_paradigms
        if not source:
            source = self._load_alignment_paradigms_runtime(
                self._config_store,
                context=context,
            )
        for item in source:
            if not isinstance(item, dict):
                continue
            slug = str(item.get("slug", "")).strip()
            if not slug:
                continue
            trials.append(
                {
                    "name": str(item.get("name", slug)),
                    "trial_slug": slug,
                    "slug": slug,
                }
            )
        return trials

    def _reload_features_paradigms(self, *, preferred_slug: str | None = None) -> None:
        paradigms = self._discover_features_trials()
        self._features_paradigms = paradigms
        if self._features_paradigm_list is None:
            return
        selected_slug = preferred_slug or self._current_features_paradigm_slug()
        self._features_paradigm_list.blockSignals(True)
        self._features_paradigm_list.clear()
        selected_row = -1
        for idx, paradigm in enumerate(paradigms):
            name = str(paradigm.get("name", paradigm.get("slug", f"Trial {idx + 1}")))
            slug = str(paradigm.get("slug", ""))
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, slug)
            self._features_paradigm_list.addItem(item)
            if slug and slug == selected_slug:
                selected_row = idx
        self._features_paradigm_list.blockSignals(False)
        self._refresh_features_trial_list_states()
        if (
            paradigms
            and selected_row >= 0
            and self._features_trial_is_selectable(paradigms[selected_row].get("slug"))
        ):
            self._sync_features_paradigm_selection(
                str(paradigms[selected_row].get("slug", ""))
            )
        else:
            self._sync_features_paradigm_selection(self._shared_stage_trial_slug())
        self._on_features_paradigm_selected(self._features_paradigm_list.currentRow())

    def _on_features_paradigm_selected(self, row: int) -> None:
        previous_slug = self._shared_stage_trial_slug()
        restore_slug: str | None = None
        if not self._syncing_shared_trial_selection:
            slug = None
            if self._features_paradigm_list is not None and row >= 0:
                item = self._features_paradigm_list.item(row)
                if item is not None:
                    value = item.data(Qt.UserRole)
                    if isinstance(value, str) and self._features_trial_is_selectable(
                        value
                    ):
                        slug = value.strip()
            if slug is not None:
                if (
                    not self._record_param_syncing
                    and previous_slug is not None
                    and previous_slug != slug
                ):
                    self._capture_features_trial_params(previous_slug)
                self._set_shared_stage_trial_slug(slug)
                self._syncing_shared_trial_selection = True
                try:
                    self._sync_alignment_paradigm_selection(slug)
                finally:
                    self._syncing_shared_trial_selection = False
                self._refresh_alignment_selected_paradigm()
                if not self._record_param_syncing and previous_slug != slug:
                    restore_slug = slug
        self._features_last_plot_figure = None
        self._features_last_plot_data = None
        self._features_last_plot_name = ""
        if restore_slug is not None:
            self._restore_features_trial_params(
                restore_slug,
                respect_dirty_keys=False,
            )
        else:
            self._refresh_features_axis_metric_combo()
            self._refresh_features_available_files()
        self._refresh_stage_states_from_context()
        self._refresh_features_controls()

    def _refresh_features_indicators(self) -> None:
        context = self._record_context()
        slug = self._current_features_paradigm_slug()
        extract_state = "gray"
        if context is not None and slug is not None:
            resolver = PathResolver(context)
            metric_keys = self._features_metric_keys_for_selected_trial()
            axes_by_metric = self._collect_features_axes_for_run(metric_keys)
            extract_state = self._features_panel_state_runtime(
                resolver,
                paradigm_slug=slug,
                axes_by_metric=axes_by_metric,
            )
        if self._features_extract_indicator is not None:
            self._set_indicator_color(self._features_extract_indicator, extract_state)
            self._features_extract_indicator.setToolTip(
                "Feature extraction state: gray=not run, yellow=stale or failed, "
                f"green=current axes and trial have successful outputs. Current: {extract_state}."
            )

    def _features_metric_keys_for_selected_trial(self) -> list[str]:
        return self._features_metric_keys_for_trial_slug(
            self._current_features_paradigm_slug()
        )
