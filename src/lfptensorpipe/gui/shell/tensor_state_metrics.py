"""Tensor metric state and panel MainWindow methods."""

from __future__ import annotations

from lfptensorpipe.app.tensor import service as tensor_service
from lfptensorpipe.app.tensor.cpu_budget import normalize_tensor_cpu_percent
from lfptensorpipe.app.tensor.orchestration_plan_validation import (
    prepare_metric_plan_inputs,
)
from lfptensorpipe.gui.shell.common import (
    Any,
    QLineEdit,
    RecordContext,
    TENSOR_BASIC_PARAM_ROWS_BY_METRIC,
    TENSOR_COMMON_BASIC_METRIC_KEYS,
    TENSOR_METRICS,
    QWidget,
    build_tensor_metric_notch_payload,
    np,
    set_control_validation_error,
)

_COMBO_DRAFT_VALUE_PROPERTY = "lfptpDraftValue"


def _tensor_metric_specs(owner: Any) -> tuple[Any, ...]:
    getter = getattr(owner, "_stage_tensor_metric_specs", None)
    if callable(getter):
        return tuple(getter())
    return tuple(TENSOR_METRICS)


class MainWindowTensorStateMetricsMixin:
    def _set_tensor_form_row_visible(
        self, widget: QWidget | None, visible: bool
    ) -> None:
        form = self._tensor_metric_params_form
        if widget is None or form is None:
            return
        set_row_visible = getattr(form, "setRowVisible", None)
        if callable(set_row_visible):
            try:
                set_row_visible(widget, visible)
            except TypeError:
                pass
        label = form.labelForField(widget)
        if label is not None:
            label.setVisible(visible)
        widget.setVisible(visible)

    def _on_tensor_param_metric_changed(self, index: int) -> None:
        # Backward-compatible shim for legacy tests that still target combo-based flow.
        _ = index
        if self._tensor_param_metric_combo is None:
            self._set_active_tensor_metric(self._tensor_active_metric_key)
            return
        metric_key = self._tensor_param_metric_combo.currentData()
        if isinstance(metric_key, str):
            self._set_active_tensor_metric(metric_key)

    @staticmethod
    def _safe_float(value: Any, fallback: float) -> float:
        try:
            parsed = float(value)
        except Exception:
            return float(fallback)
        if not np.isfinite(parsed):
            return float(fallback)
        return float(parsed)

    def _tensor_metric_display_name(self, metric_key: str) -> str:
        spec = self._tensor_metric_spec(metric_key)
        return spec.display_name if spec is not None else metric_key

    def _tensor_metric_spec(self, metric_key: str) -> Any | None:
        return next(
            (item for item in _tensor_metric_specs(self) if item.key == metric_key),
            None,
        )

    def _tensor_visible_basic_param_rows(self, metric_key: str) -> set[str]:
        spec = self._tensor_metric_spec(metric_key)
        if spec is None or not bool(getattr(spec, "supported", False)):
            return set()
        return set(TENSOR_BASIC_PARAM_ROWS_BY_METRIC.get(metric_key, ()))

    def _set_tensor_basic_param_row_visible(self, row_key: str, visible: bool) -> None:
        widget = self._tensor_basic_param_widgets.get(row_key)
        self._set_tensor_form_row_visible(widget, visible)

    def _refresh_tensor_basic_param_row_visibility(self) -> set[str]:
        visible_rows = self._tensor_visible_basic_param_rows(
            self._tensor_active_metric_key
        )
        for row_key in self._tensor_basic_param_widgets:
            self._set_tensor_basic_param_row_visible(row_key, row_key in visible_rows)
        return visible_rows

    def _refresh_tensor_selector_row_visibility(self) -> None:
        metric_key = self._tensor_active_metric_key
        self._set_tensor_form_row_visible(
            self._tensor_channels_button,
            self._tensor_metric_requires_channel_selector(metric_key),
        )
        self._set_tensor_form_row_visible(
            self._tensor_pairs_button,
            self._tensor_metric_pair_mode(metric_key) is not None,
        )

    def _tensor_active_metric_params(self) -> dict[str, Any]:
        return dict(self._tensor_metric_params.get(self._tensor_active_metric_key, {}))

    def _sync_tensor_time_resolution_label(self) -> None:
        edit = self._tensor_time_resolution_edit
        form = self._tensor_metric_params_form
        if edit is None or form is None:
            return
        label = form.labelForField(edit)
        if label is None:
            return
        label.setText("Time resolution (s)")
        edit.setToolTip(
            "Target Morlet time scale or minimum Multitaper window. "
            "Low frequencies may use longer Multitaper windows."
        )

    def _apply_tensor_metric_payload(
        self, metric_key: str, payload: dict[str, Any]
    ) -> None:
        params = dict(payload)
        if metric_key == "periodic_aperiodic":
            params.pop("smooth_enabled", None)
            params.pop("kernel_size", None)
        try:
            params.update(
                build_tensor_metric_notch_payload(
                    params.get("notches"),
                    params.get("notch_radii"),
                )
            )
        except (TypeError, ValueError):
            pass
        if metric_key in {"psi", "burst"}:
            params["bands"] = [
                dict(item)
                for item in self._normalize_tensor_bands_rows(params.get("bands"))
            ]
        if self._tensor_metric_requires_channel_selector(metric_key):
            channels = self._coerce_tensor_channels(params.get("selected_channels"))
            if self._tensor_available_channels:
                allowed = set(self._tensor_available_channels)
                channels = tuple(channel for channel in channels if channel in allowed)
            self._tensor_selected_channels_by_metric[metric_key] = channels
            params["selected_channels"] = list(channels)
        mode = self._tensor_metric_pair_mode(metric_key)
        if mode is not None:
            directed = mode == "directed"
            pairs = self._coerce_tensor_pairs(
                params.get("selected_pairs"),
                directed=directed,
            )
            if self._tensor_available_channels:
                pairs = self._filter_tensor_pairs(
                    pairs,
                    available_channels=self._tensor_available_channels,
                    directed=directed,
                )
            self._tensor_selected_pairs_by_metric[metric_key] = pairs
            params["selected_pairs"] = [[source, target] for source, target in pairs]
        self._tensor_metric_params[metric_key] = params

    def _ensure_tensor_metric_state_from_defaults(
        self, context: RecordContext | None
    ) -> None:
        available_channels = tuple(self._tensor_available_channels)
        for spec in _tensor_metric_specs(self):
            if spec.key not in self._tensor_metric_params:
                self._tensor_metric_params[spec.key] = (
                    self._tensor_effective_metric_defaults(
                        spec.key,
                        context=context,
                        available_channels=available_channels,
                    )
                )
            if self._tensor_metric_requires_channel_selector(spec.key):
                if spec.key not in self._tensor_selected_channels_by_metric:
                    params = self._tensor_metric_params.get(spec.key, {})
                    current = self._coerce_tensor_channels(
                        params.get("selected_channels")
                    )
                    if available_channels:
                        allowed = set(available_channels)
                        current = tuple(
                            channel for channel in current if channel in allowed
                        )
                    if not current and "selected_channels" not in params:
                        current = self._tensor_default_selected_channels_for_metric(
                            spec.key,
                            available_channels=available_channels,
                        )
                    self._tensor_selected_channels_by_metric[spec.key] = current
            mode = self._tensor_metric_pair_mode(spec.key)
            if mode is None:
                continue
            if spec.key not in self._tensor_selected_pairs_by_metric:
                directed = mode == "directed"
                params = self._tensor_metric_params.get(spec.key, {})
                current = self._coerce_tensor_pairs(
                    params.get("selected_pairs"), directed=directed
                )
                if available_channels:
                    current = self._filter_tensor_pairs(
                        current,
                        available_channels=available_channels,
                        directed=directed,
                    )
                if not current and "selected_pairs" not in params:
                    current = self._tensor_default_selected_pairs_for_metric(
                        spec.key,
                        directed=directed,
                        available_channels=available_channels,
                    )
                self._tensor_selected_pairs_by_metric[spec.key] = current

    @staticmethod
    def _parse_optional_float(text: str) -> float | None:
        value = text.strip()
        if not value:
            return None
        return float(value)

    @staticmethod
    def _parse_optional_int(text: str) -> int | None:
        value = text.strip()
        if not value:
            return None
        return int(value)

    @staticmethod
    def _parse_freq_range_text(text: str) -> list[float] | None:
        token = text.strip()
        if not token:
            return None
        parts = [item.strip() for item in token.split(",") if item.strip()]
        if len(parts) != 2:
            raise ValueError("SpecParam freq range must be two numbers: low,high.")
        low = float(parts[0])
        high = float(parts[1])
        if high <= low:
            raise ValueError("SpecParam freq range must satisfy high > low.")
        return [float(low), float(high)]

    @staticmethod
    def _format_freq_range(value: Any) -> str:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                low = float(value[0])
                high = float(value[1])
            except Exception:
                return str(value)
            return f"{low:g}, {high:g}"
        return "" if value is None else str(value)

    def _commit_active_tensor_panel_to_params(self) -> None:
        metric_key = self._tensor_active_metric_key
        params = dict(self._tensor_metric_params.get(metric_key, {}))

        def parse_float_field(edit: QLineEdit | None, key: str) -> None:
            if edit is None:
                return
            text = edit.text().strip()
            if not text:
                params[key] = None
                return
            try:
                params[key] = float(text)
            except Exception:
                params[key] = text

        if metric_key in TENSOR_COMMON_BASIC_METRIC_KEYS:
            parse_float_field(self._tensor_low_freq_edit, "low_freq_hz")
            parse_float_field(self._tensor_high_freq_edit, "high_freq_hz")
            parse_float_field(self._tensor_step_edit, "freq_step_hz")
            parse_float_field(self._tensor_time_resolution_edit, "time_resolution_s")
            parse_float_field(self._tensor_hop_edit, "hop_s")
        elif metric_key == "psi":
            parse_float_field(self._tensor_step_edit, "freq_step_hz")
            parse_float_field(self._tensor_time_resolution_edit, "time_resolution_s")
            parse_float_field(self._tensor_hop_edit, "hop_s")
        if (
            metric_key in {"raw_power", "periodic_aperiodic"}
            and self._tensor_method_combo is not None
        ):
            method = self._tensor_method_combo.currentData()
            if isinstance(method, str):
                params["method"] = method
                self._tensor_method_combo.setProperty(_COMBO_DRAFT_VALUE_PROPERTY, None)
            elif self._tensor_method_combo.currentIndex() < 0:
                params["method"] = self._tensor_method_combo.property(
                    _COMBO_DRAFT_VALUE_PROPERTY
                )
        if (
            metric_key == "periodic_aperiodic"
            and self._tensor_freq_range_edit is not None
        ):
            range_text = self._tensor_freq_range_edit.text().strip()
            try:
                parsed_range = self._parse_freq_range_text(range_text)
                if parsed_range is not None:
                    params["freq_range_hz"] = parsed_range
            except Exception:
                params["freq_range_hz"] = range_text
            if not range_text:
                params["freq_range_hz"] = None
        if metric_key == "burst":
            parse_float_field(self._tensor_percentile_edit, "percentile")
            parse_float_field(self._tensor_min_cycles_basic_edit, "min_cycles")

        if metric_key in self._tensor_selected_channels_by_metric:
            params["selected_channels"] = list(
                self._tensor_selected_channels_by_metric.get(metric_key, ())
            )
        if metric_key in self._tensor_selected_pairs_by_metric:
            params["selected_pairs"] = [
                [a, b]
                for a, b in self._tensor_selected_pairs_by_metric.get(metric_key, ())
            ]
        self._tensor_metric_params[metric_key] = params

    def _active_tensor_panel_indicator_params(self) -> dict[str, Any]:
        """Return the active metric draft exactly as currently visible in the panel."""
        metric_key = self._tensor_active_metric_key
        params = dict(self._tensor_metric_params.get(metric_key, {}))

        def overlay_text_field(edit: QLineEdit | None, key: str) -> None:
            if edit is None:
                return
            text = edit.text().strip()
            params[key] = text if text else None

        if metric_key in TENSOR_COMMON_BASIC_METRIC_KEYS:
            overlay_text_field(self._tensor_low_freq_edit, "low_freq_hz")
            overlay_text_field(self._tensor_high_freq_edit, "high_freq_hz")
            overlay_text_field(self._tensor_step_edit, "freq_step_hz")
            overlay_text_field(self._tensor_time_resolution_edit, "time_resolution_s")
            overlay_text_field(self._tensor_hop_edit, "hop_s")
        elif metric_key == "psi":
            overlay_text_field(self._tensor_step_edit, "freq_step_hz")
            overlay_text_field(self._tensor_time_resolution_edit, "time_resolution_s")
            overlay_text_field(self._tensor_hop_edit, "hop_s")
        if (
            metric_key in {"raw_power", "periodic_aperiodic"}
            and self._tensor_method_combo is not None
        ):
            method = self._tensor_method_combo.currentData()
            if isinstance(method, str):
                params["method"] = method
            elif self._tensor_method_combo.currentIndex() < 0:
                params["method"] = self._tensor_method_combo.property(
                    _COMBO_DRAFT_VALUE_PROPERTY
                )
        if (
            metric_key == "periodic_aperiodic"
            and self._tensor_freq_range_edit is not None
        ):
            freq_range_text = self._tensor_freq_range_edit.text().strip()
            if not freq_range_text:
                params["freq_range_hz"] = None
            else:
                try:
                    params["freq_range_hz"] = self._parse_freq_range_text(
                        freq_range_text
                    )
                except Exception:
                    params["freq_range_hz"] = freq_range_text
        if metric_key == "burst":
            overlay_text_field(self._tensor_percentile_edit, "percentile")
            overlay_text_field(self._tensor_min_cycles_basic_edit, "min_cycles")
        if metric_key in self._tensor_selected_channels_by_metric:
            params["selected_channels"] = list(
                self._tensor_selected_channels_by_metric.get(metric_key, ())
            )
        if metric_key in self._tensor_selected_pairs_by_metric:
            params["selected_pairs"] = [
                [a, b]
                for a, b in self._tensor_selected_pairs_by_metric.get(metric_key, ())
            ]
        return params

    @staticmethod
    def _tensor_error_control_keys(message: str) -> set[str]:
        lowered = str(message).lower()
        keys = {
            key
            for key in (
                "low_freq_hz",
                "high_freq_hz",
                "freq_step_hz",
                "time_resolution_s",
                "hop_s",
                "method",
                "freq_range_hz",
                "percentile",
                "min_cycles",
            )
            if key in lowered
        }
        if "frequency" in lowered and not keys:
            keys.update({"low_freq_hz", "high_freq_hz", "freq_step_hz"})
        if "band" in lowered:
            keys.add("bands")
        if "selected channel" in lowered:
            keys.add("selected_channels")
        if "selected pair" in lowered:
            keys.add("selected_pairs")
        return keys

    def _set_tensor_metric_validation_marks(
        self,
        metric_key: str,
        message: str | None,
    ) -> None:
        if metric_key != self._tensor_active_metric_key:
            return
        widgets = dict(self._tensor_basic_param_widgets)
        widgets["selected_channels"] = self._tensor_channels_button
        widgets["selected_pairs"] = self._tensor_pairs_button
        for widget in widgets.values():
            set_control_validation_error(widget, None)
        set_control_validation_error(self._tensor_advance_button, None)
        error = str(message or "").strip()
        if not error:
            return
        keys = self._tensor_error_control_keys(error)
        marked = False
        for key in keys:
            widget = widgets.get(key)
            if widget is not None and widget.isEnabled():
                set_control_validation_error(widget, error)
                marked = True
        if not marked and self._tensor_advance_button is not None:
            if self._tensor_advance_button.isEnabled():
                set_control_validation_error(self._tensor_advance_button, error)

    def _tensor_metric_draft_error(
        self,
        metric_key: str,
        params: dict[str, Any],
    ) -> str:
        context = self._record_context()
        if context is None:
            return ""
        try:
            prepare_metric_plan_inputs(
                tensor_service,
                context,
                metric_key=metric_key,
                metric_label=self._tensor_metric_display_name(metric_key),
                metric_params=dict(params),
            )
        except Exception as exc:  # noqa: BLE001
            return str(exc)
        return ""

    def _validate_active_tensor_draft(self) -> str:
        metric_key = self._tensor_active_metric_key
        params = self._active_tensor_panel_indicator_params()
        error = self._tensor_metric_draft_error(metric_key, params)
        self._set_tensor_metric_validation_marks(metric_key, error)
        return error

    def _on_tensor_basic_field_finished(self) -> None:
        self._commit_active_tensor_panel_to_params()
        self._validate_active_tensor_draft()
        self._refresh_tensor_metric_indicators_from_draft()

    def _validate_tensor_cpu_draft(self) -> str:
        edit = getattr(self, "_tensor_cpu_percent_edit", None)
        if edit is None:
            return ""
        try:
            normalize_tensor_cpu_percent(edit.text().strip())
        except ValueError as exc:
            error = str(exc)
            set_control_validation_error(edit, error)
            return error
        set_control_validation_error(edit, None)
        return ""

    def _apply_active_tensor_params_to_panel(self) -> None:
        metric_key = self._tensor_active_metric_key
        params = self._tensor_metric_params.get(metric_key, {})

        def set_float(edit: QLineEdit | None, value: Any) -> None:
            if edit is None:
                return
            if value is None:
                edit.clear()
                return
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                edit.setText(str(value))
                return
            edit.setText(f"{parsed:g}")

        if self._tensor_metric_title_label is not None:
            self._tensor_metric_title_label.setText(
                self._tensor_metric_display_name(metric_key)
            )

        if self._tensor_low_freq_edit is not None:
            set_float(self._tensor_low_freq_edit, params.get("low_freq_hz", 1.0))
        if self._tensor_high_freq_edit is not None:
            set_float(self._tensor_high_freq_edit, params.get("high_freq_hz", 100.0))
        if self._tensor_step_edit is not None:
            set_float(self._tensor_step_edit, params.get("freq_step_hz", 0.5))
        if self._tensor_time_resolution_edit is not None:
            set_float(
                self._tensor_time_resolution_edit, params.get("time_resolution_s", 0.5)
            )
        if self._tensor_hop_edit is not None:
            set_float(self._tensor_hop_edit, params.get("hop_s", 0.025))
        if self._tensor_method_combo is not None:
            method = params.get("method", "morlet")
            idx = self._tensor_method_combo.findData(str(method))
            self._tensor_method_combo.setProperty(
                _COMBO_DRAFT_VALUE_PROPERTY,
                method if idx < 0 else None,
            )
            self._tensor_method_combo.setCurrentIndex(idx)
        self._sync_tensor_time_resolution_label()
        if self._tensor_freq_range_edit is not None:
            self._tensor_freq_range_edit.setText(
                self._format_freq_range(params.get("freq_range_hz"))
            )
        if self._tensor_percentile_edit is not None:
            set_float(self._tensor_percentile_edit, params.get("percentile", 75.0))
        if self._tensor_min_cycles_basic_edit is not None:
            set_float(self._tensor_min_cycles_basic_edit, params.get("min_cycles", 2.0))

        for key, button in self._tensor_metric_name_buttons.items():
            button.setStyleSheet(
                "text-align: left; padding: 0px; border: none; font-weight: 700;"
                if key == metric_key
                else "text-align: left; padding: 0px; border: none;"
            )
        self._refresh_tensor_bands_button_text()

    def _set_active_tensor_metric(self, metric_key: str) -> None:
        if not metric_key:
            return
        self._commit_active_tensor_panel_to_params()
        self._set_tensor_metric_validation_marks(
            self._tensor_active_metric_key,
            None,
        )
        self._ensure_tensor_metric_state_from_defaults(self._record_context())
        self._tensor_active_metric_key = metric_key
        spec = next(
            (item for item in _tensor_metric_specs(self) if item.key == metric_key),
            None,
        )
        if self._tensor_metric_notice_label is not None:
            if spec is None:
                self._tensor_metric_notice_label.setText("Unknown metric.")
            elif spec.supported:
                self._tensor_metric_notice_label.setText("Ready in current slice.")
            else:
                self._tensor_metric_notice_label.setText(
                    "Pending implementation in next slices."
                )
        self._apply_active_tensor_params_to_panel()
        self._refresh_tensor_controls()
        self._mark_record_param_dirty("tensor.active_metric")
