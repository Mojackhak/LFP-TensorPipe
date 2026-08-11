"""Tensor metric-advance dialog."""

from __future__ import annotations

from lfptensorpipe.app.tensor.frequency import (
    validate_periodic_aperiodic_notch_bounds,
)
from lfptensorpipe.io.burst_thresholds import (
    load_burst_threshold_json,
    normalize_burst_threshold_payload,
)

from .common import *  # noqa: F403

MT_TIME_BANDWIDTH_PRODUCT_TOOLTIP = (
    "Dimensionless DPSS time-bandwidth product P = T x B. The default is 4.0. "
    "At fixed P, longer low-frequency windows have a narrower bandwidth in Hz."
)
MT_MIN_CYCLES_TOOLTIP = (
    "Minimum oscillation cycles in a Multitaper window. Low frequencies use a "
    "longer window when needed. The default is 3.0."
)


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
        notch_radii = QLineEdit()
        notches.setToolTip(
            "Comma-separated notch center frequencies in Hz. Leave blank to disable metric-specific notch exclusion."
        )
        notch_radii.setToolTip(
            "Half-width around each notch center. A 50 Hz center with a 2 Hz "
            "radius excludes 48-52 Hz. When inherited from Preprocess, the "
            "numeric notch-width value is preserved as the Tensor radius. "
            "A single value broadcasts to all metric notches. Leave blank to "
            "use 2 Hz."
        )
        form.addRow("Notches", notches)
        form.addRow("Notch radius (Hz)", notch_radii)
        self._fields["notches"] = notches
        self._fields["notch_radii"] = notch_radii

    @staticmethod
    def _stringify_notches(value: Any) -> str:
        payload = build_tensor_metric_notch_payload(value, 2.0)
        return ", ".join(f"{float(item):g}" for item in payload["notches"])

    @staticmethod
    def _stringify_notch_radii(value: Any) -> str:
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
    def _parse_notch_radii(text: str) -> float | list[float] | None:
        token = text.strip()
        if not token:
            return None
        parts = [item.strip() for item in token.split(",") if item.strip()]
        if not parts:
            return None
        values = [float(item) for item in parts]
        if any((not np.isfinite(float(item))) or float(item) <= 0.0 for item in values):
            raise ValueError("notch_radii must contain positive finite numbers.")
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
            max_cycles.setToolTip(
                "Optional maximum cycles used for spectral estimation."
            )
            mt_time_bandwidth_product = QLineEdit()
            mt_time_bandwidth_product.setToolTip(MT_TIME_BANDWIDTH_PRODUCT_TOOLTIP)
            mt_min_cycles = QLineEdit()
            mt_min_cycles.setToolTip(MT_MIN_CYCLES_TOOLTIP)
            form.addRow("Method", method_combo)
            form.addRow("Morlet min cycles", min_cycles)
            form.addRow("Morlet max cycles", max_cycles)
            form.addRow("MT time-bandwidth product", mt_time_bandwidth_product)
            form.addRow("MT minimum cycles", mt_min_cycles)
            self._fields = {
                "method": method_combo,
                "min_cycles": min_cycles,
                "max_cycles": max_cycles,
                "mt_time_bandwidth_product": mt_time_bandwidth_product,
                "mt_min_cycles": mt_min_cycles,
            }
            method_combo.currentIndexChanged.connect(
                self._sync_spectral_method_fields_enabled
            )
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
            max_cycles.setToolTip(
                "Optional maximum cycles used for spectral estimation."
            )
            mt_time_bandwidth_product = QLineEdit()
            mt_time_bandwidth_product.setToolTip(MT_TIME_BANDWIDTH_PRODUCT_TOOLTIP)
            mt_min_cycles = QLineEdit()
            mt_min_cycles.setToolTip(MT_MIN_CYCLES_TOOLTIP)
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
            fit_qc.setToolTip("Minimum fit-quality threshold to keep a decomposition.")
            form.addRow("Method", method_combo)
            form.addRow("Morlet min cycles", min_cycles)
            form.addRow("Morlet max cycles", max_cycles)
            form.addRow("MT time-bandwidth product", mt_time_bandwidth_product)
            form.addRow("MT minimum cycles", mt_min_cycles)
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
                "mt_time_bandwidth_product": mt_time_bandwidth_product,
                "mt_min_cycles": mt_min_cycles,
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
            method_combo.currentIndexChanged.connect(
                self._sync_spectral_method_fields_enabled
            )
            self._append_shared_notch_fields(form)
            return
        if self._metric_key in {
            "coherence",
            "imcoh_abs",
            "plv",
            "ciplv",
            "pli",
            "wpli",
            "trgc",
        }:
            method_combo = QComboBox()
            for item in ("morlet", "multitaper"):
                method_combo.addItem(item, item)
            method_combo.setToolTip("Spectral method (morlet/multitaper)")
            mt_time_bandwidth_product = QLineEdit()
            mt_time_bandwidth_product.setToolTip(MT_TIME_BANDWIDTH_PRODUCT_TOOLTIP)
            mt_min_cycles = QLineEdit()
            mt_min_cycles.setToolTip(MT_MIN_CYCLES_TOOLTIP)
            min_cycles = QLineEdit()
            min_cycles.setToolTip("Minimum cycles used for spectral estimation.")
            max_cycles = QLineEdit()
            max_cycles.setToolTip(
                "Optional maximum cycles used for spectral estimation."
            )
            form.addRow("Method", method_combo)
            form.addRow("MT time-bandwidth product", mt_time_bandwidth_product)
            form.addRow("MT minimum cycles", mt_min_cycles)
            form.addRow("Morlet min cycles", min_cycles)
            form.addRow("Morlet max cycles", max_cycles)
            self._fields = {
                "method": method_combo,
                "mt_time_bandwidth_product": mt_time_bandwidth_product,
                "mt_min_cycles": mt_min_cycles,
                "min_cycles": min_cycles,
                "max_cycles": max_cycles,
            }
            method_combo.currentIndexChanged.connect(
                self._sync_spectral_method_fields_enabled
            )
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
                group_by_samples.stateChanged.connect(self._sync_trgc_round_ms_enabled)
            self._append_shared_notch_fields(form)
            return
        if self._metric_key == "psi":
            method_combo = QComboBox()
            for item in ("morlet", "multitaper"):
                method_combo.addItem(item, item)
            method_combo.setToolTip("Spectral method (morlet/multitaper)")
            mt_time_bandwidth_product = QLineEdit()
            mt_time_bandwidth_product.setToolTip(MT_TIME_BANDWIDTH_PRODUCT_TOOLTIP)
            mt_min_cycles = QLineEdit()
            mt_min_cycles.setToolTip(MT_MIN_CYCLES_TOOLTIP)
            min_cycles = QLineEdit()
            min_cycles.setToolTip("Minimum cycles used for spectral estimation.")
            max_cycles = QLineEdit()
            max_cycles.setToolTip(
                "Optional maximum cycles used for spectral estimation."
            )
            form.addRow("Method", method_combo)
            form.addRow("MT time-bandwidth product", mt_time_bandwidth_product)
            form.addRow("MT minimum cycles", mt_min_cycles)
            form.addRow("Morlet min cycles", min_cycles)
            form.addRow("Morlet max cycles", max_cycles)
            self._fields = {
                "method": method_combo,
                "mt_time_bandwidth_product": mt_time_bandwidth_product,
                "mt_min_cycles": mt_min_cycles,
                "min_cycles": min_cycles,
                "max_cycles": max_cycles,
            }
            method_combo.currentIndexChanged.connect(
                self._sync_spectral_method_fields_enabled
            )
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
            load_button = QPushButton("Load thresholds.json")
            clear_button = QPushButton("Clear thresholds")
            load_button.clicked.connect(self._on_load_thresholds)
            clear_button.clicked.connect(self._on_clear_thresholds)
            load_button.setToolTip(
                "Load a validated Burst threshold JSON snapshot. Compatibility "
                "with the current channels and bands is checked when Burst runs."
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
            max_cycles.setToolTip(
                "Maximum allowed burst duration in cycles. Longer bursts are "
                "excluded; leave blank for no maximum."
            )
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
                elif key == "notch_radii":
                    widget.setText(self._stringify_notch_radii(value))
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
        self._sync_spectral_method_fields_enabled()
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
            path = params.get("thresholds_source_path")
            if isinstance(path, str) and path.strip():
                self._loaded_thresholds_path = path
            else:
                self._loaded_thresholds_path = None
            self._sync_burst_threshold_controls()

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

    def _sync_spectral_method_fields_enabled(self) -> None:
        method_widget = self._fields.get("method")
        if not isinstance(method_widget, QComboBox):
            return
        is_multitaper = str(method_widget.currentData()).strip().lower() == "multitaper"
        for key in ("mt_time_bandwidth_product", "mt_min_cycles"):
            widget = self._fields.get(key)
            if widget is not None:
                widget.setEnabled(is_multitaper)
        for key in ("min_cycles", "max_cycles"):
            widget = self._fields.get(key)
            if widget is not None:
                widget.setEnabled(not is_multitaper)

    def _sync_burst_threshold_controls(self) -> None:
        if self._metric_key != "burst":
            return
        provided = self._loaded_thresholds is not None
        if self._baseline_annotations_combo is not None:
            self._baseline_annotations_combo.setEnabled(not provided)
        if not provided:
            self._thresholds_path_label.setText("No file loaded")
            self._thresholds_path_label.setToolTip(
                "Currently loaded Burst thresholds file. Loaded: none."
            )
            return
        payload = normalize_burst_threshold_payload(self._loaded_thresholds)
        source_name = (
            Path(self._loaded_thresholds_path).name
            if self._loaded_thresholds_path
            else "Stored thresholds"
        )
        summary = (
            f"{source_name} — {len(payload['bands'])} bands × "
            f"{len(payload['channels'])} channels"
        )
        self._thresholds_path_label.setText(summary)
        source_detail = self._loaded_thresholds_path or "stored payload"
        self._thresholds_path_label.setToolTip(
            f"Loaded: {source_detail}. Compatibility is checked when Burst runs."
        )

    def _on_load_thresholds(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load thresholds.json",
            "",
            "JSON files (*.json)",
        )
        if not file_path:
            return
        try:
            normalized = load_burst_threshold_json(Path(file_path))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(
                self, "Burst Advance", f"Invalid thresholds file:\n{exc}"
            )
            return
        self._loaded_thresholds = normalized
        self._loaded_thresholds_path = file_path
        self._sync_burst_threshold_controls()

    def _on_clear_thresholds(self) -> None:
        self._loaded_thresholds = None
        self._loaded_thresholds_path = None
        self._sync_burst_threshold_controls()

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
                if key == "notch_radii":
                    out[key] = self._parse_notch_radii(text)
                    continue
                if not text:
                    if key in {
                        "mt_time_bandwidth_product",
                        "mt_min_cycles",
                    }:
                        raise ValueError(f"{key} is required.")
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
                elif key == "mt_time_bandwidth_product":
                    value = float(text)
                    if not np.isfinite(value) or value < 2.0:
                        raise ValueError(
                            "mt_time_bandwidth_product must be finite and >= 2."
                        )
                    out[key] = value
                elif key == "mt_min_cycles":
                    value = float(text)
                    if not np.isfinite(value) or value <= 0.0:
                        raise ValueError("mt_min_cycles must be finite and > 0.")
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
            out["thresholds_source_path"] = self._loaded_thresholds_path
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
        try:
            payload.update(
                build_tensor_metric_notch_payload(
                    payload.get("notches"),
                    payload.get("notch_radii"),
                )
            )
        except ValueError as exc:
            QMessageBox.warning(
                self, "Tensor Advance", f"Invalid advanced params:\n{exc}"
            )
            return
        if self._metric_key == "periodic_aperiodic":
            try:
                validate_periodic_aperiodic_notch_bounds(payload)
            except ValueError as exc:
                QMessageBox.warning(
                    self, "Tensor Advance", f"Invalid advanced params:\n{exc}"
                )
                return
        if action == "set_default":
            if self._set_default_callback is not None:
                try:
                    self._set_default_callback(dict(payload))
                except Exception as exc:  # noqa: BLE001
                    QMessageBox.warning(
                        self, self.windowTitle(), f"Set as default failed:\n{exc}"
                    )
                    return
            default_payload = dict(payload)
            if self._metric_key == "burst":
                default_payload.pop("thresholds_source_path", None)
            self._default_params = default_payload
            self._working_base_params = dict(payload)
            self._selected_action = action
            self._selected_params = payload
            return
        self._selected_action = action
        self._selected_params = payload
        self.accept()
