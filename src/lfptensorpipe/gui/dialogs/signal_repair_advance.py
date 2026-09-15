"""Independent gap and peak interpolation parameters."""

from __future__ import annotations

from copy import deepcopy

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
)

from lfptensorpipe.preproc.signal_repair import default_signal_repair_params


class SignalRepairAdvanceDialog(QDialog):
    def __init__(
        self,
        *,
        params,
        sfreq: float,
        default_params=None,
        set_default_callback=None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Signal Repair - Advance")
        self.selected_params = None
        self._params = deepcopy(params)
        self._default_params = deepcopy(
            default_params
            if default_params is not None
            else default_signal_repair_params()
        )
        self._set_default_callback = set_default_callback
        self._groups = {}
        layout = QVBoxLayout(self)
        layout.setSizeConstraint(QLayout.SetFixedSize)
        layout.setSpacing(10)
        for kind, title in (
            ("gaps", "Gap interpolation"),
            ("peaks", "Peak interpolation"),
        ):
            values = {**default_signal_repair_params()[kind], **params[kind]}
            group = QGroupBox(title)
            group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            group.setEnabled(values["enabled"])
            form = QFormLayout(group)
            widgets = {}
            detection = None
            if kind == "peaks":
                detection = QComboBox()
                detection.addItem("Amplitude MAD", "amplitude_mad")
                detection.addItem("Local discontinuity", "local_discontinuity")
                detection.addItem("Local z-score", "local_zscore")
                detection.setCurrentIndex(
                    detection.findData(values["detection_method"])
                )
                detection.setObjectName("repair_peaks_detection_method")
                form.addRow("Detection method", detection)
                widgets["detection_method"] = detection
            method = QComboBox()
            method.addItem("Linear", "linear")
            method.addItem("PCHIP", "pchip")
            method.setCurrentIndex(method.findData(values["method"]))
            method.setObjectName(f"repair_{kind}_method")
            form.addRow("Interpolation method", method)
            widgets["method"] = method
            maximum = QSpinBox()
            maximum.setRange(1, 2147483647)
            maximum.setValue(values["max_samples"])
            maximum.setObjectName(f"repair_{kind}_max_samples")
            form.addRow(
                "Max gap samples" if kind == "gaps" else "Max peak samples", maximum
            )
            duration = QLabel()

            def update_duration(value, label=duration, detector=detection):
                if (
                    detector is not None
                    and detector.currentData() == "local_discontinuity"
                ):
                    value = 1
                label.setText(
                    f"{value / sfreq * 1000:.3g} ms at {sfreq:g} Hz (excluding anchors)"
                )

            maximum.valueChanged.connect(update_duration)
            update_duration(maximum.value())
            form.addRow("Duration", duration)
            widgets["max_samples"] = maximum
            context = QSpinBox()
            context.setRange(2, 2147483647)
            context.setValue(values["context_samples"])
            context.setObjectName(f"repair_{kind}_context_samples")
            form.addRow("Context samples per side", context)
            widgets["context_samples"] = context

            def update_method(_index, combo=method, control=context, rows=form):
                rows.setRowVisible(control, combo.currentData() == "pchip")

            method.currentIndexChanged.connect(update_method)
            update_method(method.currentIndex())
            if kind == "peaks":
                single_sample = QSpinBox()
                single_sample.setRange(1, 1)
                single_sample.setEnabled(False)
                single_sample.setToolTip(
                    "Local discontinuity repairs isolated single samples only."
                )
                form.insertRow(3, "Max peak samples", single_sample)
                for key, label in (
                    ("baseline_window_s", "Baseline window (s)"),
                    ("detection_window_s", "Detection window (s)"),
                    ("mad_threshold", "MAD threshold"),
                    ("zscore_threshold", "Z-score threshold"),
                    ("background_window_s", "Background window (s)"),
                    ("guard_interval_ms", "Guard interval (ms)"),
                    ("prediction_threshold", "Prediction residual threshold"),
                    ("slope_threshold", "Boundary slope threshold"),
                ):
                    spin = QDoubleSpinBox()
                    spin.setDecimals(3)
                    spin.setRange(0 if key == "guard_interval_ms" else 0.001, 1000000)
                    spin.setValue(values[key])
                    spin.setObjectName(f"repair_{kind}_{key}")
                    form.addRow(label, spin)
                    widgets[key] = spin
                    if key in ("prediction_threshold", "slope_threshold"):
                        spin.setToolTip(
                            "Strict robust z threshold: abs(value - background median) / (1.4826 * MAD). Must exceed the threshold against both backgrounds."
                        )
                    elif key == "background_window_s":
                        spin.setToolTip(
                            "Total background duration: half on each side, excluding the guard intervals."
                        )
                    elif key == "guard_interval_ms":
                        spin.setToolTip(
                            "Additional excluded duration on each side of the candidate sample."
                        )

                def update_detection(_index):
                    local = detection.currentData() == "local_discontinuity"
                    for key in (
                        "baseline_window_s",
                        "detection_window_s",
                    ):
                        form.setRowVisible(widgets[key], not local)
                    form.setRowVisible(
                        widgets["mad_threshold"],
                        detection.currentData() == "amplitude_mad",
                    )
                    form.setRowVisible(
                        widgets["zscore_threshold"],
                        detection.currentData() == "local_zscore",
                    )
                    for key in (
                        "background_window_s",
                        "guard_interval_ms",
                        "prediction_threshold",
                        "slope_threshold",
                    ):
                        form.setRowVisible(widgets[key], local)
                    form.setRowVisible(maximum, not local)
                    form.setRowVisible(single_sample, local)
                    update_duration(maximum.value())

                detection.currentIndexChanged.connect(update_detection)
                update_detection(detection.currentIndex())
            self._groups[kind] = widgets
            layout.addWidget(group)
        buttons = QHBoxLayout()
        for title, callback in (
            ("Save", self.accept),
            ("Set as Default", self._set_as_default),
            ("Restore Default", self._restore_default),
            ("Cancel", self.reject),
        ):
            button = QPushButton(title)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

    def _collect_params(self):
        params = deepcopy(self._params)
        for kind, widgets in self._groups.items():
            for key, control in widgets.items():
                params[kind][key] = (
                    control.currentData()
                    if isinstance(control, QComboBox)
                    else control.value()
                )
        return params

    def _set_as_default(self):
        params = self._collect_params()
        if self._set_default_callback is not None:
            self._set_default_callback(deepcopy(params))
        self._default_params = params

    def _restore_default(self):
        for kind, widgets in self._groups.items():
            for key, control in widgets.items():
                value = self._default_params[kind][key]
                if isinstance(control, QComboBox):
                    control.setCurrentIndex(control.findData(value))
                else:
                    control.setValue(value)

    def accept(self):
        self.selected_params = self._collect_params()
        super().accept()
