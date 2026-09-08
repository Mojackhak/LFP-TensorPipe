"""Filter-advance dialog."""

from __future__ import annotations

from .common import *  # noqa: F403
from lfptensorpipe.preproc.notch import MODEL_LABELS, default_notch_model


class FilterAdvanceDialog(QDialog):
    """Advance dialog for preprocess filter parameters."""

    def __init__(
        self,
        *,
        session_params: dict[str, Any],
        default_params: dict[str, Any],
        set_default_callback: Callable[[dict[str, Any]], None] | None = None,
        parent: QWidget | None = None,
        channel_names: list[str] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Filter Advance")
        self.setModal(True)
        self.resize(520, 250)
        self._selected_action: str | None = None
        self._selected_params: dict[str, Any] | None = None
        self._default_params = default_params
        self._set_default_callback = set_default_callback
        self._channel_names = list(channel_names or [])
        self._channel_thresholds = {}
        self._restore_callback: Callable[[], None] | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)

        self._notch_widths_edit = QLineEdit()
        self._epoch_dur_edit = QLineEdit()
        self._p2p_thresh_edit = QLineEdit()
        self._autoreject_factor_edit = QLineEdit()
        self._isolate_bad_boundaries_check = QCheckBox()
        self._mark_filter_edges_check = QCheckBox()

        notch_widths_tooltip = (
            "Notch filter bandwidth (Hz) used for each notch in Filter > Notches. "
            "Use one value (e.g., 2) or a comma-separated list. Values must be > 0."
        )
        epoch_dur_tooltip = (
            "Epoch length in seconds for bad-segment detection. Smaller values catch "
            "short artifacts; larger values are smoother. Must be > 0."
        )
        p2p_tooltip = (
            "Peak-to-peak amplitude range in Volts: min,max. Epochs outside this "
            "range are marked BAD. Leave blank to disable only fixed peak-to-peak "
            "rejection; AutoReject remains active. Otherwise require two finite "
            "values with 0 <= min < max (e.g., 1e-6,1e-3)."
        )
        autoreject_tooltip = (
            "Multiplier for AutoReject channel thresholds. Higher values are more "
            "tolerant (fewer rejections); lower values are stricter. Must be > 0."
        )
        isolate_bad_boundaries_tooltip = (
            "On (default): filter every valid interval between BAD/EDGE boundaries "
            "independently so reviewed BAD values cannot enter an adjacent filter "
            "input. Off: filter the reviewed Raw continuously."
        )
        mark_filter_edges_tooltip = (
            "Mark combined filter support at each interval edge as EDGE_filter. "
            "CleanLine and removePLI can mark whole valid segments because their "
            "estimation depends on the whole segment. Off (default) accepts MNE "
            "padding results without adding these annotations."
        )

        notch_widths_label = QLabel("notch widths")
        notch_widths_label.setToolTip(notch_widths_tooltip)
        self._notch_widths_edit.setToolTip(notch_widths_tooltip)
        form.addRow(notch_widths_label, self._notch_widths_edit)

        self._model_enabled = QCheckBox("Model")
        self._model_method = QComboBox()
        for key, label in MODEL_LABELS.items():
            self._model_method.addItem(label, key)
        self._model_enabled.setToolTip("Estimate and subtract line-noise components.")
        form.addRow(self._model_enabled, self._model_method)
        self._loading_fields = False
        self._model_extras = {}
        self._model_groups = {}
        self._model_edits = {}
        names = {
            "limit_over_subtraction": "Limit over-subtraction",
            "background_radius_hz": "Background radius (Hz)",
            "background_bandwidth_hz": "Background bandwidth (Hz)",
            "window_length_s": "Window length (s)",
            "window_overlap_percent": "Window overlap (%)",
            "fit_width_hz": "Fit width (Hz)",
            "multitaper_bandwidth_hz": "Multitaper bandwidth (Hz)",
            "frequency_search_enabled": "Frequency search",
            "search_radius_hz": "Search radius (Hz)",
            "significance_threshold": "Significance threshold",
            "per_channel_thresholds_enabled": "Per-channel thresholds",
            "fundamental_frequency_hz": "Fundamental frequency (Hz)",
            "harmonic_count": "Harmonic count",
            "amplitude_phase_settling_time_s": "Amplitude/phase settling time (s)",
            "frequency_tracking_bandwidth.initial_hz": "Initial tracking bandwidth (Hz)",
            "frequency_tracking_bandwidth.final_hz": "Final tracking bandwidth (Hz)",
            "frequency_tracking_bandwidth.transition_time_s": "Bandwidth transition time (s)",
            "frequency_tracking_settling_time.initial_s": "Initial frequency settling time (s)",
            "frequency_tracking_settling_time.final_s": "Final frequency settling time (s)",
            "frequency_tracking_settling_time.transition_time_s": "Settling-time transition (s)",
        }
        for method, fields in default_notch_model()["params_by_method"].items():
            group = QWidget()
            layout = QFormLayout(group)
            edits = {}
            flat_fields = {
                (f"{key}.{subkey}" if isinstance(value, dict) else key): subvalue
                for key, value in fields.items()
                for subkey, subvalue in (
                    value.items() if isinstance(value, dict) else [("", value)]
                )
            }
            advanced = []
            for key, value in flat_fields.items():
                edit = QCheckBox() if isinstance(value, bool) else QLineEdit()
                if method == "mne_spectrum_fit" and key == "multitaper_bandwidth_hz":
                    edit.setPlaceholderText("Auto")
                    edit.setToolTip("Leave blank for MNE's automatic taper bandwidth.")
                elif key == "limit_over_subtraction":
                    edit.setToolTip(
                        "Fit one subtraction coefficient per channel, line and continuous processing segment to an interpolated background spectrum."
                    )
                elif key == "background_radius_hz":
                    edit.setToolTip(
                        "Background candidates extend this far on each side of the requested line center; detected noise bands are excluded."
                    )
                elif key == "background_bandwidth_hz":
                    edit.setToolTip(
                        "Full multitaper bandwidth for background PSD. Uses the existing window length and overlap; independent of detection bandwidth."
                    )
                elif key == "fit_width_hz":
                    edit.setToolTip(
                        "Full fitting width around each center; 0 fits the nearest Fourier bin."
                    )
                elif key == "search_radius_hz":
                    edit.setToolTip(
                        "Search each center +/- this radius. Adaptive subtraction also excludes this radius around detected peaks, even when frequency search is off."
                    )
                elif key == "significance_threshold":
                    edit.setToolTip(
                        "F-test p-value threshold; smaller values remove fewer components."
                    )
                elif "settling" in key or "transition_time" in key:
                    edit.setToolTip(
                        "Time to reach 95% of the estimator's asymptotic response."
                    )
                layout.addRow(names[key], edit)
                edits[key] = edit
                if (
                    key in {"multitaper_bandwidth_hz", "significance_threshold"}
                    or "." in key
                ):
                    advanced.append(edit)
            if method == "cleanline":
                self._channel_thresholds_button = QPushButton("Configure...")
                self._channel_thresholds_button.setToolTip(
                    "Set thresholds by channel name. Blank values inherit the global threshold."
                )
                self._channel_thresholds_button.clicked.connect(
                    self._configure_channel_thresholds
                )
                layout.insertRow(
                    layout.getWidgetPosition(edits["per_channel_thresholds_enabled"])[0]
                    + 1,
                    self._channel_thresholds_button,
                )
            if advanced:
                extra = QCheckBox("More model parameters")
                layout.insertRow(layout.getWidgetPosition(advanced[0])[0], extra)
                self._model_extras[method] = extra
                for edit in advanced:
                    layout.setRowVisible(edit, False)
                extra.toggled.connect(
                    lambda checked, form=layout, controls=advanced: [
                        form.setRowVisible(field, checked) for field in controls
                    ]
                )
            form.addRow(group)
            self._model_groups[method] = group
            self._model_edits[method] = edits

        epoch_dur_label = QLabel("epoch duration")
        epoch_dur_label.setToolTip(epoch_dur_tooltip)
        self._epoch_dur_edit.setToolTip(epoch_dur_tooltip)
        form.addRow(epoch_dur_label, self._epoch_dur_edit)

        p2p_label = QLabel("peak-to-peak threshold (min, max)")
        p2p_label.setToolTip(p2p_tooltip)
        self._p2p_thresh_edit.setToolTip(p2p_tooltip)
        form.addRow(p2p_label, self._p2p_thresh_edit)

        autoreject_label = QLabel("autoreject correct factor")
        autoreject_label.setToolTip(autoreject_tooltip)
        self._autoreject_factor_edit.setToolTip(autoreject_tooltip)
        form.addRow(autoreject_label, self._autoreject_factor_edit)

        isolate_bad_boundaries_label = QLabel("isolate BAD boundaries")
        isolate_bad_boundaries_label.setToolTip(isolate_bad_boundaries_tooltip)
        self._isolate_bad_boundaries_check.setToolTip(isolate_bad_boundaries_tooltip)
        form.addRow(
            isolate_bad_boundaries_label,
            self._isolate_bad_boundaries_check,
        )

        mark_filter_edges_label = QLabel("mark filter edges")
        mark_filter_edges_label.setToolTip(mark_filter_edges_tooltip)
        self._mark_filter_edges_check.setToolTip(mark_filter_edges_tooltip)
        form.addRow(mark_filter_edges_label, self._mark_filter_edges_check)
        root.addLayout(form)

        button_row = QWidget()
        button_layout = QHBoxLayout(button_row)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(6)
        save_button = QPushButton("Save")
        default_button = QPushButton("Set as Default")
        restore_button = QPushButton("Restore Defaults")
        cancel_button = QPushButton("Cancel")

        default_button.setToolTip(
            "Save current Advance values and Filter basic values (Notches, Low freq, "
            "High freq) as defaults."
        )
        restore_button.setToolTip(
            "Restore saved defaults for Advance values and Filter basic values "
            "(Notches, Low freq, High freq)."
        )
        save_button.setToolTip(
            "Apply these advanced filter parameters to the current session."
        )
        cancel_button.setToolTip("Close without changing session values.")

        save_button.clicked.connect(lambda: self._on_submit("save"))
        default_button.clicked.connect(lambda: self._on_submit("set_default"))
        restore_button.clicked.connect(self._on_restore_defaults)
        cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(save_button)
        button_layout.addWidget(default_button)
        button_layout.addWidget(restore_button)
        button_layout.addWidget(cancel_button)
        button_layout.addStretch(1)
        root.addWidget(button_row)

        self._apply_to_fields(session_params)
        self._model_enabled.toggled.connect(self._sync_model_controls)
        self._model_method.currentIndexChanged.connect(self._sync_model_controls)
        self._model_enabled.toggled.connect(self._refresh_validation)
        self._model_method.currentIndexChanged.connect(self._refresh_validation)
        for edits in self._model_edits.values():
            for edit in edits.values():
                if isinstance(edit, QCheckBox):
                    edit.toggled.connect(self._sync_model_controls)
                    edit.toggled.connect(self._refresh_validation)
                else:
                    edit.editingFinished.connect(self._refresh_validation)
        self._isolate_bad_boundaries_check.toggled.connect(
            self._sync_mark_filter_edges_enabled
        )
        self._mark_filter_edges_check.toggled.connect(self._refresh_validation)
        for edit in (
            self._notch_widths_edit,
            self._epoch_dur_edit,
            self._p2p_thresh_edit,
            self._autoreject_factor_edit,
        ):
            edit.editingFinished.connect(self._refresh_validation)
        self._refresh_validation()

    @property
    def selected_action(self) -> str | None:
        return self._selected_action

    @property
    def selected_params(self) -> dict[str, Any] | None:
        return self._selected_params

    def _show_warning(self, title: str, message: str) -> int:
        return QMessageBox.warning(self, title, message)

    def set_restore_callback(self, callback: Callable[[], None] | None) -> None:
        self._restore_callback = callback

    def _on_restore_defaults(self) -> None:
        for extra in self._model_extras.values():
            extra.setChecked(False)
        self._apply_to_fields(self._default_params)
        self._refresh_validation()
        if self._restore_callback is not None:
            self._restore_callback()

    @staticmethod
    def _stringify_notch_widths(value: float | list[float]) -> str:
        if isinstance(value, list):
            return ", ".join(f"{item:g}" for item in value)
        return f"{float(value):g}"

    @staticmethod
    def _draft_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return f"{value:g}"
        return str(value)

    def _apply_to_fields(self, params: dict[str, Any]) -> None:
        self._loading_fields = True
        model = params.get("notch_model") or default_notch_model()
        self._model_enabled.setChecked(model.get("enabled", False))
        method = model.get("method", default_notch_model()["method"])
        index = self._model_method.findData(method)
        if index < 0:
            self._model_method.addItem(f"Unsupported model: {method}", method)
            index = self._model_method.count() - 1
        self._model_method.setCurrentIndex(index)
        defaults = default_notch_model()["params_by_method"]
        cleanline = model.get("params_by_method", {}).get("cleanline", {})
        if model.get("method") == "cleanline" and "params" in model:
            cleanline = model["params"]
        self._channel_thresholds = dict(
            cleanline.get("significance_thresholds_by_channel", {})
        )
        for method, edits in self._model_edits.items():
            values = model.get("params_by_method", {}).get(method, defaults[method])
            if method == model.get("method") and "params" in model:
                values = model["params"]
            for key, edit in edits.items():
                path = key.split(".")
                source, fallback = values, defaults[method]
                for part in path[:-1]:
                    source = source.get(part, {})
                    fallback = fallback[part]
                value = source.get(path[-1], fallback[path[-1]])
                if isinstance(edit, QCheckBox):
                    edit.setChecked(value is True)
                else:
                    edit.setText(self._draft_text(value))
        self._sync_model_controls()
        notch_value = params.get("notch_widths", 2.0)
        p2p_value = params.get("p2p_thresh", [1e-6, 1e-3])
        if p2p_value is None:
            p2p_text = ""
        elif isinstance(p2p_value, (list, tuple)) and len(p2p_value) == 2:
            p2p_text = ", ".join(self._draft_text(item) for item in p2p_value)
        else:
            p2p_text = str(p2p_value)

        if isinstance(notch_value, list):
            notch_text = ", ".join(self._draft_text(item) for item in notch_value)
        else:
            notch_text = self._draft_text(notch_value)
        self._notch_widths_edit.setText(notch_text)
        self._epoch_dur_edit.setText(self._draft_text(params.get("epoch_dur", 1.0)))
        self._p2p_thresh_edit.setText(p2p_text)
        self._autoreject_factor_edit.setText(
            self._draft_text(params.get("autoreject_correct_factor", 1.5))
        )
        self._isolate_bad_boundaries_check.setChecked(
            params.get("isolate_bad_boundaries", True) is True
        )
        self._mark_filter_edges_check.setChecked(
            params.get("mark_filter_edges", False) is True
        )
        self._sync_mark_filter_edges_enabled(
            self._isolate_bad_boundaries_check.isChecked()
        )

        self._loading_fields = False

    def _sync_mark_filter_edges_enabled(self, isolated: bool) -> None:
        self._mark_filter_edges_check.setEnabled(bool(isolated))
        if not isolated:
            self._mark_filter_edges_check.setChecked(False)
        self._refresh_validation()

    def _sync_model_controls(self, *_args) -> None:
        enabled = self._model_enabled.isChecked()
        self._model_method.setEnabled(enabled)
        self._notch_widths_edit.setEnabled(not enabled)
        for method, group in self._model_groups.items():
            group.setVisible(enabled and method == self._model_method.currentData())
        cleanline = self._model_edits["cleanline"]
        self._channel_thresholds_button.setEnabled(
            cleanline["per_channel_thresholds_enabled"].isChecked()
        )
        cleanline["search_radius_hz"].setEnabled(
            cleanline["frequency_search_enabled"].isChecked()
            or cleanline["limit_over_subtraction"].isChecked()
        )
        for key in ("background_radius_hz", "background_bandwidth_hz"):
            cleanline[key].setEnabled(cleanline["limit_over_subtraction"].isChecked())

    @staticmethod
    def _parse_notch_widths(text: str) -> float | list[float]:
        parts = [item.strip() for item in text.split(",") if item.strip()]
        if not parts:
            raise ValueError("notch_widths cannot be empty.")
        if len(parts) == 1:
            return float(parts[0])
        return [float(item) for item in parts]

    @staticmethod
    def _parse_p2p_thresh(text: str) -> list[float] | None:
        parts = [item.strip() for item in text.split(",") if item.strip()]
        if not parts:
            return None
        if len(parts) != 2:
            raise ValueError(
                "p2p_thresh must be empty or provided as two numbers: min,max."
            )
        return [float(parts[0]), float(parts[1])]

    @staticmethod
    def _draft_number(text: str) -> float | str | None:
        token = text.strip()
        if not token:
            return None
        try:
            value = float(token)
        except Exception:
            return token
        return value if np.isfinite(value) else token

    @classmethod
    def _draft_number_list(
        cls,
        text: str,
        *,
        empty_value: Any,
    ) -> Any:
        token = text.strip()
        if not token:
            return empty_value
        parts = [item.strip() for item in token.split(",")]
        if any(not item for item in parts):
            return token
        parsed = [cls._draft_number(item) for item in parts]
        if any(not isinstance(item, float) for item in parsed):
            return token
        return parsed[0] if len(parsed) == 1 else parsed

    def _configure_channel_thresholds(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("CleanLine channel thresholds")
        dialog.resize(480, 400)
        layout = QVBoxLayout(dialog)
        layout.addWidget(
            QLabel(
                "Blank = global Significance threshold. Clear absent-channel overrides to remove them."
            )
        )
        names = list(dict.fromkeys([*self._channel_names, *self._channel_thresholds]))
        table = QTableWidget(len(names), 2)
        table.setHorizontalHeaderLabels(["Channel", "Significance threshold"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for row, name in enumerate(names):
            item = QTableWidgetItem(name)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            if name not in self._channel_names:
                item.setToolTip(
                    "Absent from current Filter input; clear the override to remove it."
                )
            table.setItem(row, 0, item)
            table.setItem(
                row,
                1,
                QTableWidgetItem(self._draft_text(self._channel_thresholds.get(name))),
            )
        layout.addWidget(table)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)

        def accept():
            values = {}
            for row, name in enumerate(names):
                token = table.item(row, 1).text().strip()
                if not token:
                    continue
                value = self._draft_number(token)
                if not isinstance(value, float) or not 0 < value < 1:
                    QMessageBox.warning(
                        dialog,
                        "Invalid threshold",
                        f"{name}: enter a value strictly between 0 and 1, or leave blank.",
                    )
                    return
                values[name] = value
            self._channel_thresholds = values
            dialog.accept()

        buttons.accepted.connect(accept)
        buttons.rejected.connect(dialog.reject)
        dialog.exec()
        self._refresh_validation()

    def _collect_model_drafts(self) -> dict[str, Any]:
        drafts = {}
        for method, edits in self._model_edits.items():
            values = {}
            for key, edit in edits.items():
                value = (
                    edit.isChecked()
                    if isinstance(edit, QCheckBox)
                    else self._draft_number(edit.text())
                )
                if "." in key:
                    parent, child = key.split(".")
                    values.setdefault(parent, {})[child] = value
                else:
                    values[key] = value
            if method == "cleanline":
                values["significance_thresholds_by_channel"] = dict(
                    self._channel_thresholds
                )
            drafts[method] = values
        return drafts

    def _collect_draft_params(self) -> dict[str, Any]:
        return {
            "notch_model": {
                "enabled": self._model_enabled.isChecked(),
                "method": self._model_method.currentData(),
                "params_by_method": self._collect_model_drafts(),
            },
            "notch_widths": self._draft_number_list(
                self._notch_widths_edit.text(),
                empty_value=None,
            ),
            "epoch_dur": self._draft_number(self._epoch_dur_edit.text()),
            "p2p_thresh": self._draft_number_list(
                self._p2p_thresh_edit.text(),
                empty_value=None,
            ),
            "autoreject_correct_factor": self._draft_number(
                self._autoreject_factor_edit.text()
            ),
            "isolate_bad_boundaries": (self._isolate_bad_boundaries_check.isChecked()),
            "mark_filter_edges": self._mark_filter_edges_check.isChecked(),
        }

    def _collect_params(self) -> dict[str, Any]:
        candidate = self._collect_draft_params()
        valid, normalized, message = normalize_filter_advance_params(candidate)
        if not valid:
            raise ValueError(message)
        return normalized

    def _refresh_validation(self) -> None:
        if self._loading_fields:
            return
        controls = (
            self._model_method,
            self._channel_thresholds_button,
            self._notch_widths_edit,
            self._epoch_dur_edit,
            self._p2p_thresh_edit,
            self._autoreject_factor_edit,
            *[edit for edits in self._model_edits.values() for edit in edits.values()],
        )
        for control in controls:
            set_control_validation_error(control, None)
        valid, _, message = normalize_filter_advance_params(
            self._collect_draft_params()
        )
        if valid:
            return
        lowered = message.lower()
        if "notch_model" in lowered:
            method = self._model_method.currentData()
            edits = self._model_edits.get(method, {})
            targets = (self._model_method,)
            for key, edit in edits.items():
                if f"notch_model.{method}.{key}" in message:
                    targets = (edit,)
                    break
            if "significance_thresholds_by_channel" in message:
                targets = (self._channel_thresholds_button,)
            extra = self._model_extras.get(method)
            if extra is not None and any(
                edit in targets
                and (
                    key in {"multitaper_bandwidth_hz", "significance_threshold"}
                    or "." in key
                )
                for key, edit in edits.items()
            ):
                extra.setChecked(True)
        elif "notch_width" in lowered:
            targets = (self._notch_widths_edit,)
        elif "epoch_dur" in lowered:
            targets = (self._epoch_dur_edit,)
        elif "p2p" in lowered:
            targets = (self._p2p_thresh_edit,)
        elif "autoreject" in lowered:
            targets = (self._autoreject_factor_edit,)
        else:
            targets = controls
        for control in targets:
            set_control_validation_error(control, message)

    def _on_submit(self, action: str) -> None:
        if action == "set_default":
            try:
                params = self._collect_params()
            except Exception as exc:  # noqa: BLE001
                self._refresh_validation()
                self._show_warning("Filter Advance", f"Invalid parameters:\n{exc}")
                return
            if self._set_default_callback is not None:
                try:
                    self._set_default_callback(dict(params))
                except Exception as exc:  # noqa: BLE001
                    self._show_warning(
                        self.windowTitle(), f"Set as default failed:\n{exc}"
                    )
                    return
            self._default_params = dict(params)
            self._selected_action = action
            self._selected_params = params
            return
        params = self._collect_draft_params()
        self._selected_action = action
        self._selected_params = params
        self.accept()
