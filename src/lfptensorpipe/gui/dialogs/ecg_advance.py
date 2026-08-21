"""Method-specific ECG Advance dialog."""

from __future__ import annotations

from .common import *  # noqa: F403

_TECHNICAL_FLOAT_LIMIT = 1_000_000_000.0
_TECHNICAL_INT_LIMIT = 1_000_000_000
_DATA_DRIVEN_SENTINEL = -_TECHNICAL_FLOAT_LIMIT


class ECGAdvanceDialog(QDialog):
    """Edit advanced parameters for one ECG artifact-removal method."""

    def __init__(
        self,
        *,
        method: str,
        session_params: dict[str, Any],
        default_params: dict[str, Any],
        session_review_params: dict[str, Any] | None = None,
        default_review_params: dict[str, Any] | None = None,
        set_default_callback: (
            Callable[[dict[str, Any], dict[str, bool]], None] | None
        ) = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._method = str(method).strip().lower()
        if self._method not in ECG_METHODS:
            raise ValueError(f"Unknown ECG method: {method}")
        self.setWindowTitle(f"ECG Advance — {self._method}")
        self.setModal(True)
        self.resize(620, 620)

        self._selected_action: str | None = None
        self._selected_params: dict[str, Any] | None = None
        self._selected_review_params: dict[str, bool] | None = None
        self._default_params = self._normalized_or_builtin(default_params)
        self._default_review_params = self._normalized_review_or_builtin(
            default_review_params
        )
        self._set_default_callback = set_default_callback
        self._manual_peak_max = 3.5
        self._manual_threshold_start = 0.0
        self._manual_threshold_step = 0.01

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        self._content_layout = QVBoxLayout(scroll_content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(8)
        scroll.setWidget(scroll_content)
        root.addWidget(scroll, stretch=1)

        if self._method in {"template", "svd"}:
            self._build_template_fields()
        else:
            self._build_perceive_fields()
        self._build_review_fields()
        self._content_layout.addStretch(1)

        button_row = QWidget()
        button_layout = QHBoxLayout(button_row)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(6)
        save_button = QPushButton("Save")
        default_button = QPushButton("Set as Default")
        restore_button = QPushButton("Restore Defaults")
        cancel_button = QPushButton("Cancel")
        save_button.setToolTip("Save these parameters for the current record.")
        default_button.setToolTip(
            "Save these parameters as defaults for this ECG method."
        )
        restore_button.setToolTip("Load the saved defaults into this dialog.")
        cancel_button.setToolTip("Close without changing record parameters.")
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

        session_candidate = dict(default_ecg_method_params(self._method))
        if isinstance(session_params, dict):
            session_candidate.update(
                {
                    key: session_params[key]
                    for key in session_candidate
                    if key in session_params
                }
            )
        try:
            self._apply_to_fields(session_candidate)
        except (KeyError, TypeError, ValueError):
            self._apply_to_fields(default_ecg_method_params(self._method))
        review_candidate = self._normalized_review_or_builtin(session_review_params)
        self._mark_filter_edges_check.setChecked(review_candidate["mark_filter_edges"])
        self._connect_validation_signals()
        self._refresh_validation()

    @property
    def selected_action(self) -> str | None:
        return self._selected_action

    @property
    def selected_params(self) -> dict[str, Any] | None:
        return (
            dict(self._selected_params)
            if isinstance(self._selected_params, dict)
            else None
        )

    @property
    def selected_review_params(self) -> dict[str, bool] | None:
        return (
            dict(self._selected_review_params)
            if isinstance(self._selected_review_params, dict)
            else None
        )

    def _normalized_or_builtin(self, params: dict[str, Any]) -> dict[str, Any]:
        ok, normalized, _ = normalize_ecg_method_params(self._method, params)
        if ok:
            return normalized
        return default_ecg_method_params(self._method)

    @staticmethod
    def _normalized_review_or_builtin(
        params: dict[str, Any] | None,
    ) -> dict[str, bool]:
        ok, normalized, _ = normalize_ecg_review_params(params)
        if ok:
            return normalized
        return default_ecg_review_params()

    @staticmethod
    def _form() -> QFormLayout:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        return form

    def _group(self, title: str) -> tuple[QGroupBox, QFormLayout]:
        group = QGroupBox(title)
        form = self._form()
        group.setLayout(form)
        self._content_layout.addWidget(group)
        return group, form

    @staticmethod
    def _add_row(
        form: QFormLayout,
        text: str,
        widget: QWidget,
        tooltip: str,
    ) -> QLabel:
        label = QLabel(text)
        label.setToolTip(tooltip)
        widget.setToolTip(tooltip)
        form.addRow(label, widget)
        return label

    @staticmethod
    def _double_spin(
        *,
        decimals: int,
        step: float,
        minimum: float = 0.0,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setRange(minimum, _TECHNICAL_FLOAT_LIMIT)
        spin.setReadOnly(False)
        return spin

    @staticmethod
    def _int_spin() -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(1, _TECHNICAL_INT_LIMIT)
        spin.setSingleStep(1)
        spin.setReadOnly(False)
        return spin

    def _ms_spin(self) -> QDoubleSpinBox:
        return self._double_spin(decimals=3, step=1.0)

    def _build_template_fields(self) -> None:
        _, detection = self._group("Detection")
        self._window_ms_spin = self._ms_spin()
        self._add_row(
            detection,
            "Baseline window (ms)",
            self._window_ms_spin,
            "Set the baseline-removal window for QRS detection.",
        )

        self._peak_min_spin = self._double_spin(
            decimals=3,
            step=0.1,
        )
        self._add_row(
            detection,
            "Peak height minimum (z-score)",
            self._peak_min_spin,
            "Set the minimum z-score for QRS peak detection.",
        )

        peak_max_row = QWidget()
        peak_max_layout = QHBoxLayout(peak_max_row)
        peak_max_layout.setContentsMargins(0, 0, 0, 0)
        peak_max_layout.setSpacing(6)
        self._peak_max_limit_check = QCheckBox("Limit maximum peak height")
        self._peak_max_spin = self._double_spin(
            decimals=3,
            step=0.1,
        )
        self._peak_max_spin.setSpecialValueText("No limit")
        peak_max_tooltip = "Set the maximum z-score, or leave the limit disabled."
        self._peak_max_limit_check.setToolTip(peak_max_tooltip)
        self._peak_max_spin.setToolTip(peak_max_tooltip)
        peak_max_layout.addWidget(self._peak_max_limit_check)
        peak_max_layout.addWidget(self._peak_max_spin, stretch=1)
        self._add_row(
            detection,
            "Peak height maximum (z-score)",
            peak_max_row,
            peak_max_tooltip,
        )
        self._peak_max_limit_check.toggled.connect(self._on_peak_max_limit_toggled)

        self._min_interpeak_ms_spin = self._ms_spin()
        self._add_row(
            detection,
            "Minimum interpeak interval (ms)",
            self._min_interpeak_ms_spin,
            "Set the minimum interval between detected QRS peaks.",
        )

        self._orientation_combo = QComboBox()
        self._orientation_combo.addItem("Dominant", None)
        self._orientation_combo.addItem("Positive", "positive")
        self._orientation_combo.addItem("Negative", "negative")
        self._add_row(
            detection,
            "Peak orientation",
            self._orientation_combo,
            "Choose dominant, positive, or negative peak detection.",
        )

        _, artifact = self._group("Artifact Window")
        self._pre_ms_spin = self._ms_spin()
        self._add_row(
            artifact,
            "Pre-peak duration (ms)",
            self._pre_ms_spin,
            "Set the artifact window before each QRS peak.",
        )
        self._post_ms_spin = self._ms_spin()
        self._add_row(
            artifact,
            "Post-peak duration (ms)",
            self._post_ms_spin,
            "Set the artifact window after each QRS peak.",
        )
        self._tail_ms_spin = self._ms_spin()
        self._add_row(
            artifact,
            "Boundary tail (ms)",
            self._tail_ms_spin,
            "Set the edge region used to blend artifact subtraction.",
        )
        self._qrs_duration_ms_spin = self._ms_spin()
        self._add_row(
            artifact,
            "QRS duration (ms)",
            self._qrs_duration_ms_spin,
            "Set the QRS duration used for template subtraction.",
        )
        self._pqrst_check = QCheckBox()
        self._add_row(
            artifact,
            "Use full PQRST",
            self._pqrst_check,
            "Use the full PQRST window instead of the QRS region.",
        )

        if self._method == "svd":
            _, reconstruction = self._group("Reconstruction")
            self._components_spin = self._int_spin()
            self._add_row(
                reconstruction,
                "SVD components",
                self._components_spin,
                "Set the number of SVD components used for reconstruction.",
            )

    def _build_perceive_fields(self) -> None:
        _, initial = self._group("Initial Template")
        self._epoch_length_ms_spin = self._ms_spin()
        self._add_row(
            initial,
            "Epoch length (ms)",
            self._epoch_length_ms_spin,
            "Set the epoch length used to build the initial ECG template.",
        )
        self._window_ms_spin = self._ms_spin()
        self._add_row(
            initial,
            "Baseline window (ms)",
            self._window_ms_spin,
            "Set the baseline-removal window for template construction.",
        )
        self._threshold_uv_spin = self._double_spin(
            decimals=3,
            step=1.0,
        )
        self._add_row(
            initial,
            "Amplitude threshold (µV)",
            self._threshold_uv_spin,
            "Set the amplitude threshold used to find template candidates.",
        )
        self._pad_ms_spin = self._ms_spin()
        self._add_row(
            initial,
            "Crop padding (ms)",
            self._pad_ms_spin,
            "Set the padding around the detected QRS template.",
        )

        _, heart_rate = self._group("Heart-rate Limits")
        self._min_bpm_spin = self._int_spin()
        self._add_row(
            heart_rate,
            "Minimum heart rate (BPM)",
            self._min_bpm_spin,
            "Set the minimum expected heart rate.",
        )
        self._max_bpm_spin = self._int_spin()
        self._add_row(
            heart_rate,
            "Maximum heart rate (BPM)",
            self._max_bpm_spin,
            "Set the maximum expected heart rate.",
        )

        _, search = self._group("Threshold Search")
        self._threshold_mode_combo = QComboBox()
        self._threshold_mode_combo.addItem("Data-driven", "data_driven")
        self._threshold_mode_combo.addItem("Manual", "manual")
        self._add_row(
            search,
            "Threshold mode",
            self._threshold_mode_combo,
            "Choose data-driven or manual correlation thresholds.",
        )
        self._threshold_start_spin = self._double_spin(
            decimals=6,
            step=0.01,
            minimum=_DATA_DRIVEN_SENTINEL,
        )
        self._threshold_start_spin.setSpecialValueText("Data-driven")
        self._add_row(
            search,
            "Threshold start",
            self._threshold_start_spin,
            "Set the starting correlation threshold.",
        )
        self._threshold_step_spin = self._double_spin(
            decimals=6,
            step=0.01,
            minimum=_DATA_DRIVEN_SENTINEL,
        )
        self._threshold_step_spin.setSpecialValueText("Data-driven")
        self._add_row(
            search,
            "Threshold step",
            self._threshold_step_spin,
            "Set the correlation-threshold increment.",
        )
        self._max_threshold_tries_spin = self._int_spin()
        self._add_row(
            search,
            "Maximum attempts",
            self._max_threshold_tries_spin,
            "Set the maximum number of threshold-search attempts.",
        )
        self._pass_rate_spin = self._double_spin(
            decimals=2,
            step=1.0,
        )
        self._pass_rate_spin.setMaximum(100.0)
        self._add_row(
            search,
            "Pass rate (%)",
            self._pass_rate_spin,
            "Set the required percentage of valid interpeak intervals.",
        )
        self._threshold_mode_combo.currentIndexChanged.connect(
            self._on_threshold_mode_changed
        )

        _, refined = self._group("Refined Template")
        self._before_ms_spin = self._ms_spin()
        self._add_row(
            refined,
            "Before peak (ms)",
            self._before_ms_spin,
            "Set the refined template duration before each peak.",
        )
        self._after_ms_spin = self._ms_spin()
        self._add_row(
            refined,
            "After peak (ms)",
            self._after_ms_spin,
            "Set the refined template duration after each peak.",
        )
        self._enforce_max_interval_check = QCheckBox()
        self._add_row(
            refined,
            "Enforce maximum interval",
            self._enforce_max_interval_check,
            "Require interpeak intervals to stay within both BPM limits.",
        )

    def _build_review_fields(self) -> None:
        _, review = self._group("Review")
        self._mark_filter_edges_check = QCheckBox()
        self._add_row(
            review,
            "mark filter edges",
            self._mark_filter_edges_check,
            "After ECG plot review, mark the accepted Filter support only around "
            "new or expanded BAD boundaries. This requires ECG to consume Filter "
            "directly.",
        )

    def _on_peak_max_limit_toggled(self, checked: bool) -> None:
        if checked:
            peak_min = float(self._peak_min_spin.value())
            candidate = max(float(self._manual_peak_max), peak_min + 1.0)
            self._peak_max_spin.setValue(candidate)
            self._peak_max_spin.setEnabled(True)
            return
        if self._peak_max_spin.isEnabled():
            self._manual_peak_max = float(self._peak_max_spin.value())
        self._peak_max_spin.setValue(self._peak_max_spin.minimum())
        self._peak_max_spin.setEnabled(False)

    def _on_threshold_mode_changed(self) -> None:
        manual = self._threshold_mode_combo.currentData() == "manual"
        if manual:
            self._threshold_start_spin.setValue(self._manual_threshold_start)
            self._threshold_step_spin.setValue(self._manual_threshold_step)
        else:
            if self._threshold_start_spin.isEnabled():
                self._manual_threshold_start = float(self._threshold_start_spin.value())
                self._manual_threshold_step = float(self._threshold_step_spin.value())
            self._threshold_start_spin.setValue(_DATA_DRIVEN_SENTINEL)
            self._threshold_step_spin.setValue(_DATA_DRIVEN_SENTINEL)
        self._threshold_start_spin.setEnabled(manual)
        self._threshold_step_spin.setEnabled(manual)

    def _apply_to_fields(self, params: dict[str, Any]) -> None:
        if self._method in {"template", "svd"}:
            self._window_ms_spin.setValue(float(params["window_ms"]))
            peak_min, peak_max = params["peak_height_range"]
            self._peak_min_spin.setValue(float(peak_min))
            if peak_max is None:
                self._manual_peak_max = float(peak_min) + 1.0
                self._peak_max_limit_check.setChecked(False)
                self._on_peak_max_limit_toggled(False)
            else:
                self._manual_peak_max = float(peak_max)
                self._peak_max_limit_check.setChecked(True)
                self._peak_max_spin.setValue(float(peak_max))
                self._peak_max_spin.setEnabled(True)
            self._min_interpeak_ms_spin.setValue(float(params["min_interpeak_ms"]))
            orientation_index = self._orientation_combo.findData(
                params["force_orientation"]
            )
            self._orientation_combo.setCurrentIndex(max(orientation_index, 0))
            self._pre_ms_spin.setValue(float(params["pre_ms"]))
            self._post_ms_spin.setValue(float(params["post_ms"]))
            self._tail_ms_spin.setValue(float(params["tail_ms"]))
            self._qrs_duration_ms_spin.setValue(float(params["qrs_duration_ms"]))
            self._pqrst_check.setChecked(bool(params["pqrst"]))
            if self._method == "svd":
                self._components_spin.setValue(int(params["components"]))
            return

        self._epoch_length_ms_spin.setValue(float(params["epoch_length_ms"]))
        self._window_ms_spin.setValue(float(params["window_ms"]))
        self._threshold_uv_spin.setValue(float(params["threshold_v"]) * 1e6)
        self._pad_ms_spin.setValue(float(params["pad_ms"]))
        self._min_bpm_spin.setValue(int(params["min_bpm"]))
        self._max_bpm_spin.setValue(int(params["max_bpm"]))
        threshold_start = params["threshold_start"]
        threshold_step = params["threshold_step"]
        if threshold_start is None:
            self._manual_threshold_start = 0.0
            self._manual_threshold_step = 0.01
            mode_index = self._threshold_mode_combo.findData("data_driven")
            self._threshold_mode_combo.setCurrentIndex(mode_index)
            self._on_threshold_mode_changed()
        else:
            self._manual_threshold_start = float(threshold_start)
            self._manual_threshold_step = float(threshold_step)
            mode_index = self._threshold_mode_combo.findData("manual")
            self._threshold_mode_combo.setCurrentIndex(mode_index)
            self._on_threshold_mode_changed()
        self._max_threshold_tries_spin.setValue(int(params["max_threshold_tries"]))
        self._pass_rate_spin.setValue(float(params["pass_rate"]) * 100.0)
        self._before_ms_spin.setValue(float(params["before_ms"]))
        self._after_ms_spin.setValue(float(params["after_ms"]))
        self._enforce_max_interval_check.setChecked(
            bool(params["enforce_max_interval"])
        )

    def _collect_draft_params(self) -> dict[str, Any]:
        if self._method in {"template", "svd"}:
            candidate: dict[str, Any] = {
                "window_ms": self._window_ms_spin.value(),
                "peak_height_range": [
                    self._peak_min_spin.value(),
                    (
                        self._peak_max_spin.value()
                        if self._peak_max_limit_check.isChecked()
                        else None
                    ),
                ],
                "min_interpeak_ms": self._min_interpeak_ms_spin.value(),
                "force_orientation": self._orientation_combo.currentData(),
                "pre_ms": self._pre_ms_spin.value(),
                "post_ms": self._post_ms_spin.value(),
                "tail_ms": self._tail_ms_spin.value(),
                "qrs_duration_ms": self._qrs_duration_ms_spin.value(),
                "pqrst": self._pqrst_check.isChecked(),
            }
            if self._method == "svd":
                candidate["components"] = self._components_spin.value()
        else:
            manual = self._threshold_mode_combo.currentData() == "manual"
            candidate = {
                "epoch_length_ms": self._epoch_length_ms_spin.value(),
                "window_ms": self._window_ms_spin.value(),
                "threshold_v": self._threshold_uv_spin.value() / 1_000_000.0,
                "pad_ms": self._pad_ms_spin.value(),
                "min_bpm": self._min_bpm_spin.value(),
                "max_bpm": self._max_bpm_spin.value(),
                "threshold_start": (
                    self._threshold_start_spin.value() if manual else None
                ),
                "threshold_step": (
                    self._threshold_step_spin.value() if manual else None
                ),
                "max_threshold_tries": self._max_threshold_tries_spin.value(),
                "pass_rate": self._pass_rate_spin.value() / 100.0,
                "before_ms": self._before_ms_spin.value(),
                "after_ms": self._after_ms_spin.value(),
                "enforce_max_interval": (self._enforce_max_interval_check.isChecked()),
            }
        return candidate

    def _collect_params(self) -> dict[str, Any]:
        valid, normalized, message = normalize_ecg_method_params(
            self._method,
            self._collect_draft_params(),
        )
        if not valid:
            raise ValueError(message)
        return normalized

    def _collect_review_params(self) -> dict[str, bool]:
        valid, normalized, message = normalize_ecg_review_params(
            {"mark_filter_edges": self._mark_filter_edges_check.isChecked()}
        )
        if not valid:
            raise ValueError(message)
        return normalized

    def _on_restore_defaults(self) -> None:
        self._apply_to_fields(self._default_params)
        self._mark_filter_edges_check.setChecked(
            self._default_review_params["mark_filter_edges"]
        )
        self._refresh_validation()

    def _connect_validation_signals(self) -> None:
        for widget in self.findChildren(QDoubleSpinBox):
            widget.valueChanged.connect(lambda _value: self._refresh_validation())
        for widget in self.findChildren(QSpinBox):
            widget.valueChanged.connect(lambda _value: self._refresh_validation())
        for widget in self.findChildren(QComboBox):
            widget.currentIndexChanged.connect(
                lambda _index: self._refresh_validation()
            )
        for widget in self.findChildren(QCheckBox):
            widget.toggled.connect(lambda _checked: self._refresh_validation())

    def _refresh_validation(self) -> None:
        widgets = [
            *self.findChildren(QDoubleSpinBox),
            *self.findChildren(QSpinBox),
            *self.findChildren(QComboBox),
        ]
        for widget in widgets:
            set_control_validation_error(widget, None)
        valid, _, message = normalize_ecg_method_params(
            self._method,
            self._collect_draft_params(),
        )
        if valid:
            return
        lowered = message.lower()
        if self._method == "perceive":
            field_widgets = {
                "epoch_length_ms": [self._epoch_length_ms_spin],
                "window_ms": [self._window_ms_spin],
                "threshold_v": [self._threshold_uv_spin],
                "pad_ms": [self._pad_ms_spin],
                "min_bpm": [self._min_bpm_spin, self._max_bpm_spin],
                "max_bpm": [self._min_bpm_spin, self._max_bpm_spin],
                "threshold_start": [
                    self._threshold_start_spin,
                    self._threshold_step_spin,
                ],
                "threshold_step": [
                    self._threshold_start_spin,
                    self._threshold_step_spin,
                ],
                "max_threshold_tries": [self._max_threshold_tries_spin],
                "pass_rate": [self._pass_rate_spin],
                "before_ms": [self._before_ms_spin],
                "after_ms": [self._after_ms_spin],
            }
        else:
            field_widgets = {
                "window_ms": [self._window_ms_spin],
                "peak_height_range": [self._peak_min_spin, self._peak_max_spin],
                "min_interpeak_ms": [self._min_interpeak_ms_spin],
                "pre_ms": [self._pre_ms_spin],
                "post_ms": [self._post_ms_spin],
                "tail_ms": [self._tail_ms_spin],
                "qrs_duration_ms": [self._qrs_duration_ms_spin],
            }
            if self._method == "svd":
                field_widgets["components"] = [self._components_spin]
        targets = next(
            (
                candidate_widgets
                for token, candidate_widgets in field_widgets.items()
                if token in lowered
            ),
            widgets,
        )
        for widget in targets:
            if widget.isEnabled():
                set_control_validation_error(widget, message)

    def _show_warning(self, title: str, message: str) -> int:
        return QMessageBox.warning(self, title, message)

    def _on_submit(self, action: str) -> None:
        if action == "set_default":
            try:
                params = self._collect_params()
                review_params = self._collect_review_params()
            except Exception as exc:  # noqa: BLE001
                self._refresh_validation()
                self._show_warning(self.windowTitle(), f"Invalid parameters:\n{exc}")
                return
            if self._set_default_callback is not None:
                try:
                    self._set_default_callback(dict(params), dict(review_params))
                except Exception as exc:  # noqa: BLE001
                    self._show_warning(
                        self.windowTitle(),
                        f"Set as default failed:\n{exc}",
                    )
                    return
            self._default_params = dict(params)
            self._default_review_params = dict(review_params)
            self._selected_action = action
            self._selected_params = dict(params)
            self._selected_review_params = dict(review_params)
            return
        params = self._collect_draft_params()
        review_params = self._collect_review_params()
        self._selected_action = action
        self._selected_params = dict(params)
        self._selected_review_params = dict(review_params)
        self.accept()
