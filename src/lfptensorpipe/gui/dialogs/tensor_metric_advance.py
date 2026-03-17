"""Tensor metric-advance dialog."""

from __future__ import annotations

from .common import *  # noqa: F403


class TensorMetricAdvanceDialog(QDialog):
    """Advance dialog for tensor metric parameters."""

    def __init__(
        self,
        *,
        metric_key: str,
        metric_label: str,
        session_params: dict[str, Any],
        default_params: dict[str, Any],
        burst_baseline_annotations: tuple[str, ...] = (),
        set_default_callback: Callable[[dict[str, Any]], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{metric_label} Advance")
        self.resize(560, 360)
        self._metric_key = metric_key
        self._selected_action: str | None = None
        self._selected_params: dict[str, Any] | None = None
        self._default_params = dict(default_params)
        self._working_base_params = dict(session_params)
        self._burst_baseline_annotations = tuple(
            str(item).strip()
            for item in burst_baseline_annotations
            if str(item).strip()
        )
        self._set_default_callback = set_default_callback
        self._fields: dict[str, Any] = {}
        self._loaded_thresholds: Any = None
        self._loaded_thresholds_path: str | None = None
        self._baseline_annotations_combo: QComboBox | None = None
        self._trgc_group_by_samples_checkbox: QCheckBox | None = None
        self._trgc_round_ms_edit: QLineEdit | None = None
        self._periodic_freq_smooth_checkbox: QCheckBox | None = None
        self._periodic_freq_smooth_sigma_edit: QLineEdit | None = None
        self._periodic_time_smooth_checkbox: QCheckBox | None = None
        self._periodic_time_smooth_kernel_edit: QLineEdit | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        self._build_fields(form)
        root.addLayout(form)

        button_row = QWidget()
        button_layout = QHBoxLayout(button_row)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(6)
        save_button = QPushButton("Save")
        default_button = QPushButton("Set as Default")
        restore_button = QPushButton("Restore Defaults")
        cancel_button = QPushButton("Cancel")
        save_button.setToolTip(
            "Apply these advanced metric parameters to the current session."
        )
        default_button.setToolTip(
            "Save the current advanced metric parameters as defaults."
        )
        restore_button.setToolTip("Restore saved defaults for this metric.")
        cancel_button.setToolTip("Close without changing session values.")
        save_button.clicked.connect(lambda: self._on_submit("save"))
        default_button.clicked.connect(lambda: self._on_submit("set_default"))
        restore_button.clicked.connect(self._on_restore_defaults)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(save_button)
        button_layout.addWidget(default_button)
        button_layout.addWidget(restore_button)
        button_layout.addStretch(1)
        button_layout.addWidget(cancel_button)
        root.addWidget(button_row)

        self._apply_to_fields(self._working_base_params)

    @property
    def selected_action(self) -> str | None:
        return self._selected_action

    @property
    def selected_params(self) -> dict[str, Any] | None:
        return self._selected_params

    def _append_shared_notch_fields(self, form: QFormLayout) -> None:
        notches = QLineEdit()
        notch_widths = QLineEdit()
        notches.setToolTip(
            "Comma-separated notch center frequencies in Hz. Leave blank to disable metric-specific notch exclusion."
        )
        notch_widths.setToolTip(
            "Notch filter bandwidth (Hz). A single value broadcasts to all metric notches. Leave blank to use 2 Hz."
        )
        form.addRow("Notches", notches)
        form.addRow("Notch widths", notch_widths)
        self._fields["notches"] = notches
        self._fields["notch_widths"] = notch_widths

    @staticmethod
    def _stringify_notches(value: Any) -> str:
        payload = build_tensor_metric_notch_payload(value, 2.0)
        return ", ".join(f"{float(item):g}" for item in payload["notches"])

    @staticmethod
    def _stringify_notch_widths(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (list, tuple)):
            return ", ".join(f"{float(item):g}" for item in value)
        return f"{float(value):g}"

    @staticmethod
    def _parse_notches(text: str) -> list[float]:
        payload = build_tensor_metric_notch_payload(text, 2.0)
        return [float(item) for item in payload["notches"]]

    @staticmethod
    def _parse_notch_widths(text: str) -> float | list[float] | None:
        token = text.strip()
        if not token:
            return None
        parts = [item.strip() for item in token.split(",") if item.strip()]
        if not parts:
            return None
        values = [float(item) for item in parts]
        if any((not np.isfinite(float(item))) or float(item) <= 0.0 for item in values):
            raise ValueError("notch_widths must contain positive finite numbers.")
        if len(values) == 1:
            return float(values[0])
        return [float(item) for item in values]

    def _build_fields(self, form: QFormLayout) -> None:
        if self._metric_key == "raw_power":
            method_combo = QComboBox()
            for item in ("morlet", "multitaper"):
                method_combo.addItem(item, item)
            method_combo.setToolTip("Spectral method (morlet/multitaper)")
            min_cycles = QLineEdit()
            min_cycles.setToolTip("Minimum cycles used for spectral estimation.")
            max_cycles = QLineEdit()
            max_cycles.setToolTip("Optional maximum cycles used for spectral estimation.")
            time_bandwidth = QLineEdit()
            time_bandwidth.setToolTip(
                "Used for multitaper spectral estimation. Higher values increase frequency smoothing and stability but reduce frequency resolution."
            )
            form.addRow("Method", method_combo)
            form.addRow("Min cycles", min_cycles)
            form.addRow("Max cycles", max_cycles)
            form.addRow("Time bandwidth", time_bandwidth)
            self._fields = {
                "method": method_combo,
                "min_cycles": min_cycles,
                "max_cycles": max_cycles,
                "time_bandwidth": time_bandwidth,
            }
            self._append_shared_notch_fields(form)
            return
        if self._metric_key == "periodic_aperiodic":
            method_combo = QComboBox()
            for item in ("morlet", "multitaper"):
                method_combo.addItem(item, item)
            method_combo.setToolTip("Spectral method (morlet/multitaper)")
            min_cycles = QLineEdit()
            min_cycles.setToolTip("Minimum cycles used for spectral estimation.")
            max_cycles = QLineEdit()
            max_cycles.setToolTip("Optional maximum cycles used for spectral estimation.")
            time_bandwidth = QLineEdit()
            time_bandwidth.setToolTip(
                "Used for multitaper spectral estimation. Higher values increase frequency smoothing and stability but reduce frequency resolution."
            )
            freq_smooth = QCheckBox()
            freq_smooth.setToolTip(
                "Enable pre-decomposition frequency-axis Gaussian smoothing."
            )
            freq_smooth_sigma = QLineEdit()
            freq_smooth_sigma.setToolTip(
                "Gaussian sigma in frequency bins for the SpecParam input spectrum."
            )
            time_smooth = QCheckBox()
            time_smooth.setToolTip(
                "Enable pre-decomposition time-axis median smoothing."
            )
            time_smooth_kernel_size = QLineEdit()
            time_smooth_kernel_size.setToolTip(
                "Median-filter kernel size in time bins. Defaults to an odd value derived from time_resolution_s / hop_s."
            )
            aperiodic_mode = QComboBox()
            for item in ("fixed", "knee"):
                aperiodic_mode.addItem(item, item)
            aperiodic_mode.setToolTip("Choose fixed or knee aperiodic fit.")
            peak_width_limits = QLineEdit()
            peak_width_limits.setToolTip("Peak width limits in Hz as low,high.")
            max_n_peaks = QLineEdit()
            max_n_peaks.setToolTip(
                "Maximum number of peaks to fit; use inf for no limit."
            )
            min_peak_height = QLineEdit()
            min_peak_height.setToolTip("Minimum peak height for peak finding.")
            peak_threshold = QLineEdit()
            peak_threshold.setToolTip("Peak detection threshold.")
            fit_qc = QLineEdit()
            fit_qc.setToolTip(
                "Minimum fit-quality threshold to keep a decomposition."
            )
            form.addRow("Method", method_combo)
            form.addRow("Min cycles", min_cycles)
            form.addRow("Max cycles", max_cycles)
            form.addRow("Time bandwidth", time_bandwidth)
            form.addRow("Freq", freq_smooth)
            form.addRow("Freq smooth sigma", freq_smooth_sigma)
            form.addRow("Time", time_smooth)
            form.addRow("Time smooth kernel size", time_smooth_kernel_size)
            form.addRow("Aperiodic mode", aperiodic_mode)
            form.addRow("Peak width limits", peak_width_limits)
            form.addRow("Max n peaks", max_n_peaks)
            form.addRow("Min peak height", min_peak_height)
            form.addRow("Peak threshold", peak_threshold)
            form.addRow("Fit QC threshold", fit_qc)
            self._fields = {
                "method": method_combo,
                "min_cycles": min_cycles,
                "max_cycles": max_cycles,
                "time_bandwidth": time_bandwidth,
                "freq_smooth_enabled": freq_smooth,
                "freq_smooth_sigma": freq_smooth_sigma,
                "time_smooth_enabled": time_smooth,
                "time_smooth_kernel_size": time_smooth_kernel_size,
                "aperiodic_mode": aperiodic_mode,
                "peak_width_limits_hz": peak_width_limits,
                "max_n_peaks": max_n_peaks,
                "min_peak_height": min_peak_height,
                "peak_threshold": peak_threshold,
                "fit_qc_threshold": fit_qc,
            }
            self._periodic_freq_smooth_checkbox = freq_smooth
            self._periodic_freq_smooth_sigma_edit = freq_smooth_sigma
            self._periodic_time_smooth_checkbox = time_smooth
            self._periodic_time_smooth_kernel_edit = time_smooth_kernel_size
            freq_smooth.stateChanged.connect(
                self._sync_periodic_smoothing_fields_enabled
            )
            time_smooth.stateChanged.connect(
                self._sync_periodic_smoothing_fields_enabled
            )
            self._append_shared_notch_fields(form)
            return
        if self._metric_key in {"coherence", "plv", "ciplv", "pli", "wpli", "trgc"}:
            method_combo = QComboBox()
            for item in ("morlet", "multitaper"):
                method_combo.addItem(item, item)
            method_combo.setToolTip("Spectral method (morlet/multitaper)")
            mt_bandwidth = QLineEdit()
            mt_bandwidth.setToolTip("Multitaper bandwidth parameter.")
            min_cycles = QLineEdit()
            min_cycles.setToolTip("Minimum cycles used for spectral estimation.")
            max_cycles = QLineEdit()
            max_cycles.setToolTip("Optional maximum cycles used for spectral estimation.")
            form.addRow("Method", method_combo)
            form.addRow("MT bandwidth", mt_bandwidth)
            form.addRow("Min cycles", min_cycles)
            form.addRow("Max cycles", max_cycles)
            self._fields = {
                "method": method_combo,
                "mt_bandwidth": mt_bandwidth,
                "min_cycles": min_cycles,
                "max_cycles": max_cycles,
            }
            if self._metric_key == "trgc":
                gc_n_lags = QLineEdit()
                gc_n_lags.setToolTip("Number of lags for Granger/TRGC modeling.")
                group_by_samples = QCheckBox()
                group_by_samples.setToolTip(
                    "Group TRGC frequencies by exact window length in samples. Recommended only when you want grouping tied to the recording sample rate; for most runs leave this off and use Round ms."
                )
                round_ms = QLineEdit()
                round_ms.setToolTip(
                    "Millisecond grid used to group TRGC window lengths when Group by samples is off. Recommended: keep 50 ms for most runs; smaller values preserve finer timing differences but can create more groups."
                )
                form.addRow("GC lags", gc_n_lags)
                form.addRow("Group by samples", group_by_samples)
                form.addRow("Round ms", round_ms)
                self._fields["gc_n_lags"] = gc_n_lags
                self._fields["group_by_samples"] = group_by_samples
                self._fields["round_ms"] = round_ms
                self._trgc_group_by_samples_checkbox = group_by_samples
                self._trgc_round_ms_edit = round_ms
                group_by_samples.stateChanged.connect(
                    self._sync_trgc_round_ms_enabled
                )
            self._append_shared_notch_fields(form)
            return
        if self._metric_key == "psi":
            method_combo = QComboBox()
            for item in ("morlet", "multitaper"):
                method_combo.addItem(item, item)
            method_combo.setToolTip("Spectral method (morlet/multitaper)")
            mt_bandwidth = QLineEdit()
            mt_bandwidth.setToolTip("Multitaper bandwidth parameter.")
            min_cycles = QLineEdit()
            min_cycles.setToolTip("Minimum cycles used for spectral estimation.")
            max_cycles = QLineEdit()
            max_cycles.setToolTip("Optional maximum cycles used for spectral estimation.")
            form.addRow("Method", method_combo)
            form.addRow("MT bandwidth", mt_bandwidth)
            form.addRow("Min cycles", min_cycles)
            form.addRow("Max cycles", max_cycles)
            self._fields = {
                "method": method_combo,
                "mt_bandwidth": mt_bandwidth,
                "min_cycles": min_cycles,
                "max_cycles": max_cycles,
            }
            self._append_shared_notch_fields(form)
            return
        if self._metric_key == "burst":
            thresholds_row = QWidget()
            thresholds_layout = QHBoxLayout(thresholds_row)
            thresholds_layout.setContentsMargins(0, 0, 0, 0)
            thresholds_layout.setSpacing(6)
            self._thresholds_path_label = QLabel("No file loaded")
            self._thresholds_path_label.setToolTip(
                "Currently loaded burst thresholds file."
            )
            load_button = QPushButton("Load thresholds.pkl")
            clear_button = QPushButton("Clear thresholds")
            load_button.clicked.connect(self._on_load_thresholds)
            clear_button.clicked.connect(self._on_clear_thresholds)
            load_button.setToolTip(
                "Load precomputed burst thresholds from a pickle file."
            )
            clear_button.setToolTip("Remove the loaded burst thresholds file.")
            thresholds_layout.addWidget(self._thresholds_path_label, stretch=1)
            thresholds_layout.addWidget(load_button)
            thresholds_layout.addWidget(clear_button)
            baseline_combo = QComboBox()
            baseline_combo.setToolTip(
                "Optional annotation label used for Burst baseline thresholding. "
                "Only finish-step annotations with duration > 0 are listed."
            )
            baseline_combo.addItem("", None)
            for label in self._burst_baseline_choice_labels():
                baseline_combo.addItem(label, label)
            self._baseline_annotations_combo = baseline_combo
            min_cycles = QLineEdit()
            min_cycles.setToolTip("Minimum cycles used for burst detection.")
            max_cycles = QLineEdit()
            max_cycles.setToolTip("Optional maximum cycles used for burst detection.")
            form.addRow("Thresholds", thresholds_row)
            form.addRow("Baseline annotations", baseline_combo)
            form.addRow("Min cycles", min_cycles)
            form.addRow("Max cycles", max_cycles)
            self._fields = {"min_cycles": min_cycles, "max_cycles": max_cycles}
            self._append_shared_notch_fields(form)
            return

    @staticmethod
    def _normalize_baseline_keep(value: Any) -> list[str] | None:
        if value is None:
            return None
        items = value if isinstance(value, (list, tuple)) else [value]
        labels: list[str] = []
        seen: set[str] = set()
        for item in items:
            label = str(item).strip()
            if not label or label in seen:
                continue
            seen.add(label)
            labels.append(label)
        return labels or None

    def _burst_baseline_choice_labels(self) -> tuple[str, ...]:
        labels: list[str] = []
        seen: set[str] = set()
        for source in (
            self._burst_baseline_annotations,
            self._normalize_baseline_keep(
                self._working_base_params.get("baseline_keep")
            )
            or [],
            self._normalize_baseline_keep(self._default_params.get("baseline_keep"))
            or [],
        ):
            for item in source:
                label = str(item).strip()
                if not label or label in seen:
                    continue
                seen.add(label)
                labels.append(label)
        return tuple(labels)

    def _apply_to_fields(self, params: dict[str, Any]) -> None:
        for key, widget in self._fields.items():
            value = params.get(key)
            if isinstance(widget, QLineEdit):
                if key == "notches":
                    widget.setText(self._stringify_notches(value))
                elif key == "notch_widths":
                    widget.setText(self._stringify_notch_widths(value))
                elif value is None:
                    widget.clear()
                elif isinstance(value, (list, tuple)) and len(value) == 2:
                    widget.setText(f"{float(value[0]):g}, {float(value[1]):g}")
                else:
                    widget.setText(str(value))
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QComboBox):
                idx = widget.findData(value)
                if idx < 0:
                    idx = 0
                widget.setCurrentIndex(idx)
        self._sync_trgc_round_ms_enabled()
        self._sync_periodic_smoothing_fields_enabled()
        if self._metric_key == "burst":
            if self._baseline_annotations_combo is not None:
                baseline_keep = self._normalize_baseline_keep(
                    params.get("baseline_keep")
                )
                selected_label = baseline_keep[0] if baseline_keep else None
                if selected_label is not None:
                    idx = self._baseline_annotations_combo.findData(selected_label)
                    if idx < 0:
                        self._baseline_annotations_combo.addItem(
                            selected_label, selected_label
                        )
                        idx = self._baseline_annotations_combo.count() - 1
                    self._baseline_annotations_combo.setCurrentIndex(idx)
                else:
                    self._baseline_annotations_combo.setCurrentIndex(0)
            self._loaded_thresholds = params.get("thresholds")
            path = params.get("thresholds_path")
            if isinstance(path, str) and path.strip():
                self._loaded_thresholds_path = path
                self._thresholds_path_label.setText(path)
                self._thresholds_path_label.setToolTip(
                    "Currently loaded burst thresholds file. "
                    f"Loaded: {path}."
                )
            else:
                self._loaded_thresholds_path = None
                self._thresholds_path_label.setText("No file loaded")
                self._thresholds_path_label.setToolTip(
                    "Currently loaded burst thresholds file. Loaded: none."
                )

    def _sync_trgc_round_ms_enabled(self) -> None:
        if (
            self._trgc_group_by_samples_checkbox is None
            or self._trgc_round_ms_edit is None
        ):
            return
        self._trgc_round_ms_edit.setEnabled(
            not self._trgc_group_by_samples_checkbox.isChecked()
        )

    def _sync_periodic_smoothing_fields_enabled(self) -> None:
        if (
            self._periodic_freq_smooth_checkbox is not None
            and self._periodic_freq_smooth_sigma_edit is not None
        ):
            self._periodic_freq_smooth_sigma_edit.setEnabled(
                self._periodic_freq_smooth_checkbox.isChecked()
            )
        if (
            self._periodic_time_smooth_checkbox is not None
            and self._periodic_time_smooth_kernel_edit is not None
        ):
            self._periodic_time_smooth_kernel_edit.setEnabled(
                self._periodic_time_smooth_checkbox.isChecked()
            )

    @staticmethod
    def _normalize_burst_thresholds(value: Any) -> Any:
        if value is None:
            return None
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            if not np.isfinite(float(arr)):
                raise ValueError("Invalid thresholds value.")
            return [float(arr)]
        if not np.all(np.isfinite(arr)):
            raise ValueError("Thresholds contain non-finite values.")
        if arr.ndim == 1:
            return [float(item) for item in arr.tolist()]
        if arr.ndim == 2:
            return [[float(item) for item in row] for row in arr.tolist()]
        raise ValueError("Thresholds must be scalar, 1D, or 2D.")

    def _on_load_thresholds(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load thresholds.pkl",
            "",
            "Pickle files (*.pkl);;All files (*)",
        )
        if not file_path:
            return
        try:
            payload = load_pkl(Path(file_path))
            normalized = self._normalize_burst_thresholds(payload)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(
                self, "Burst Advance", f"Invalid thresholds file:\n{exc}"
            )
            return
        self._loaded_thresholds = normalized
        self._loaded_thresholds_path = file_path
        self._thresholds_path_label.setText(file_path)
        self._thresholds_path_label.setToolTip(
            "Currently loaded burst thresholds file. "
            f"Loaded: {file_path}."
        )

    def _on_clear_thresholds(self) -> None:
        self._loaded_thresholds = None
        self._loaded_thresholds_path = None
        self._thresholds_path_label.setText("No file loaded")
        self._thresholds_path_label.setToolTip(
            "Currently loaded burst thresholds file. Loaded: none."
        )

    def _on_restore_defaults(self) -> None:
        self._working_base_params = dict(self._default_params)
        self._apply_to_fields(self._working_base_params)

    def _collect_params(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, widget in self._fields.items():
            if isinstance(widget, QLineEdit):
                text = widget.text().strip()
                if key == "notches":
                    out[key] = self._parse_notches(text)
                    continue
                if key == "notch_widths":
                    out[key] = self._parse_notch_widths(text)
                    continue
                if not text:
                    out[key] = None
                    continue
                if key in {"time_smooth_kernel_size", "gc_n_lags"}:
                    value = int(text)
                    if value < 1:
                        raise ValueError(f"{key} must be >= 1.")
                    out[key] = value
                elif key == "round_ms":
                    value = float(text)
                    if value <= 0.0:
                        raise ValueError("round_ms must be > 0.")
                    out[key] = value
                elif key == "freq_smooth_sigma":
                    value = float(text)
                    if value <= 0.0:
                        raise ValueError("freq_smooth_sigma must be > 0.")
                    out[key] = value
                elif key == "peak_width_limits_hz":
                    parts = [item.strip() for item in text.split(",") if item.strip()]
                    if len(parts) != 2:
                        raise ValueError(
                            "Peak width limits must be two numbers: low,high."
                        )
                    out[key] = [float(parts[0]), float(parts[1])]
                elif key == "max_n_peaks":
                    out[key] = "inf" if text.lower() == "inf" else float(text)
                else:
                    out[key] = float(text)
            elif isinstance(widget, QCheckBox):
                out[key] = bool(widget.isChecked())
            elif isinstance(widget, QComboBox):
                data = widget.currentData()
                out[key] = str(data) if data is not None else str(widget.currentText())
        if self._metric_key == "burst":
            if self._baseline_annotations_combo is not None:
                selected_label = self._baseline_annotations_combo.currentData()
                if isinstance(selected_label, str) and selected_label.strip():
                    out["baseline_keep"] = [selected_label.strip()]
                else:
                    out["baseline_keep"] = None
            out["thresholds"] = self._loaded_thresholds
            out["thresholds_path"] = self._loaded_thresholds_path
        return out

    def _on_submit(self, action: str) -> None:
        try:
            field_payload = self._collect_params()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(
                self, "Tensor Advance", f"Invalid advanced params:\n{exc}"
            )
            return
        payload = dict(self._working_base_params)
        payload.update(field_payload)
        if action == "set_default":
            if self._set_default_callback is not None:
                try:
                    self._set_default_callback(dict(payload))
                except Exception as exc:  # noqa: BLE001
                    QMessageBox.warning(
                        self, self.windowTitle(), f"Set as default failed:\n{exc}"
                    )
                    return
            self._default_params = dict(payload)
            self._working_base_params = dict(payload)
            self._selected_action = action
            self._selected_params = payload
            return
        self._selected_action = action
        self._selected_params = payload
        self.accept()
