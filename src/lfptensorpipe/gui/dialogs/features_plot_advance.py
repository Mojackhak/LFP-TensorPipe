"""Features plot-advance dialog."""

from __future__ import annotations

from .common import *  # noqa: F403
from .baseline_range import BaselineRangeConfigureDialog


def validate_features_plot_advance_params(
    params: Any,
    *,
    allow_x_log: bool,
    allow_y_log: bool,
    allow_normalize: bool,
) -> dict[str, Any]:
    """Validate active plot controls and return the effective plot payload."""
    if not isinstance(params, dict):
        raise ValueError("Plot advance parameters must be an object.")

    transform_mode = params.get("transform_mode", "none")
    if transform_mode is None:
        raise ValueError("Transform must not be empty.")
    transform_mode = normalize_feature_plot_transform_mode(transform_mode)
    if transform_mode not in FEATURE_PLOT_TRANSFORM_MODES:
        raise ValueError(f"Unsupported transform: {transform_mode!r}.")

    normalize_mode: Any = params.get("normalize_mode", "none")
    if allow_normalize:
        if normalize_mode is None:
            raise ValueError("Normalize must not be empty.")
        normalize_mode = str(normalize_mode).strip()
        if normalize_mode not in FEATURE_PLOT_NORMALIZE_MODES:
            raise ValueError(f"Unsupported normalize mode: {normalize_mode!r}.")
    else:
        normalize_mode = "none"

    baseline_mode: Any = params.get("baseline_mode", "mean")
    ranges: Any = params.get("baseline_percent_ranges", [[0.0, 20.0]])
    normalized_ranges: Any = ranges
    if allow_normalize and normalize_mode != "none":
        if baseline_mode is None:
            raise ValueError("Baseline stat must not be empty.")
        baseline_mode = str(baseline_mode).strip()
        if baseline_mode not in FEATURE_PLOT_BASELINE_MODES:
            raise ValueError(f"Unsupported baseline stat: {baseline_mode!r}.")
        if not isinstance(ranges, list) or not ranges:
            raise ValueError(
                "Baseline ranges are required when normalization is enabled."
            )
        parsed_ranges: list[list[float]] = []
        for item in ranges:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("Each baseline range must contain start and end.")
            if isinstance(item[0], bool) or isinstance(item[1], bool):
                raise ValueError("Baseline range bounds must be numeric.")
            try:
                start = float(item[0])
                end = float(item[1])
            except (TypeError, ValueError) as exc:
                raise ValueError("Baseline range bounds must be numeric.") from exc
            if not np.isfinite(start) or not np.isfinite(end):
                raise ValueError("Baseline range bounds must be finite.")
            if start < 0.0 or end > 100.0 or end <= start:
                raise ValueError(
                    "Baseline ranges must satisfy 0 <= start < end <= 100."
                )
            parsed_ranges.append([start, end])
        parsed_ranges.sort(key=lambda item: item[0])
        for previous, current in zip(parsed_ranges, parsed_ranges[1:]):
            if current[0] < previous[1]:
                raise ValueError("Baseline ranges must not overlap.")
        normalized_ranges = parsed_ranges

    colormap = params.get("colormap", "viridis")
    if colormap is None:
        raise ValueError("Colormap must not be empty.")
    colormap = str(colormap).strip()
    if colormap not in FEATURE_PLOT_COLORMAPS:
        raise ValueError(f"Unsupported colormap: {colormap!r}.")
    if colormap == "cmcrameri.vik":
        try:
            import cmcrameri  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise ValueError("cmcrameri is required for cmcrameri.vik.") from exc

    raw_x_log = params.get("x_log", False)
    if allow_x_log and not isinstance(raw_x_log, bool):
        raise ValueError("x_log must be true or false.")
    raw_y_log = params.get("y_log", False)
    if allow_y_log and not isinstance(raw_y_log, bool):
        raise ValueError("y_log must be true or false.")

    return {
        "transform_mode": transform_mode,
        "normalize_mode": normalize_mode,
        "baseline_mode": baseline_mode,
        "baseline_percent_ranges": normalized_ranges,
        "colormap": colormap,
        "x_log": bool(raw_x_log) if allow_x_log else False,
        "y_log": bool(raw_y_log) if allow_y_log else False,
    }


class FeaturesPlotAdvanceDialog(QDialog):
    """Advance parameters for plot-time transform/normalization."""

    def __init__(
        self,
        *,
        session_params: dict[str, Any],
        default_params: dict[str, Any],
        allow_x_log: bool,
        allow_y_log: bool,
        allow_normalize: bool,
        set_default_callback: Callable[[dict[str, Any]], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Plot Advance")
        self.setModal(True)
        self.resize(560, 300)
        self._selected_action: str | None = None
        self._selected_params: dict[str, Any] | None = None
        self._default_params = dict(default_params)
        self._allow_x_log = bool(allow_x_log)
        self._allow_y_log = bool(allow_y_log)
        self._allow_normalize = bool(allow_normalize)
        self._set_default_callback = set_default_callback
        self._baseline_ranges: Any = []
        self._x_log_raw: Any = False
        self._y_log_raw: Any = False

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)

        self._transform_combo = QComboBox()
        for label, value in FEATURE_PLOT_TRANSFORM_OPTIONS:
            self._transform_combo.addItem(label, value)
        self._transform_combo.setToolTip("Value transform applied before plotting.")
        form.addRow("Transform", self._transform_combo)

        self._normalize_combo = QComboBox()
        for item in FEATURE_PLOT_NORMALIZE_MODES:
            self._normalize_combo.addItem(item, item)
        self._normalize_combo.setToolTip("Baseline normalization mode.")
        form.addRow("Normalize", self._normalize_combo)

        self._baseline_mode_combo = QComboBox()
        for item in FEATURE_PLOT_BASELINE_MODES:
            self._baseline_mode_combo.addItem(item, item)
        self._baseline_mode_combo.setToolTip(
            "Statistic used to summarize baseline values."
        )
        form.addRow("Baseline stat", self._baseline_mode_combo)

        baseline_row = QWidget()
        baseline_layout = QHBoxLayout(baseline_row)
        baseline_layout.setContentsMargins(0, 0, 0, 0)
        baseline_layout.setSpacing(6)
        self._baseline_button = QPushButton("Baseline Configure... (0)")
        self._baseline_button.clicked.connect(self._on_baseline_configure)
        self._baseline_button.setToolTip(
            "Edit baseline percent ranges used for normalization."
        )
        baseline_layout.addWidget(self._baseline_button)
        baseline_layout.addStretch(1)
        form.addRow("Baseline", baseline_row)

        self._colormap_combo = QComboBox()
        for item in FEATURE_PLOT_COLORMAPS:
            self._colormap_combo.addItem(item, item)
        self._colormap_combo.setToolTip("Colormap for matrix-style plots.")
        form.addRow("Colormap", self._colormap_combo)

        scale_row = QWidget()
        scale_layout = QHBoxLayout(scale_row)
        scale_layout.setContentsMargins(0, 0, 0, 0)
        scale_layout.setSpacing(12)
        self._x_log_check = QCheckBox("x_log")
        self._y_log_check = QCheckBox("y_log")
        self._x_log_check.setToolTip("Use log scale on the x-axis when supported.")
        self._y_log_check.setToolTip("Use log scale on the y-axis when supported.")
        scale_layout.addWidget(self._x_log_check)
        scale_layout.addWidget(self._y_log_check)
        scale_layout.addStretch(1)
        form.addRow("Axis log", scale_row)
        root.addLayout(form)
        self._normalize_combo.currentIndexChanged.connect(
            lambda _idx: self._update_dynamic_state()
        )
        for combo in (
            self._transform_combo,
            self._normalize_combo,
            self._baseline_mode_combo,
            self._colormap_combo,
        ):
            combo.currentIndexChanged.connect(lambda _idx: self._refresh_validation())
        self._x_log_check.stateChanged.connect(self._on_x_log_changed)
        self._y_log_check.stateChanged.connect(self._on_y_log_changed)

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)
        save_button = QPushButton("Save")
        default_button = QPushButton("Set as Default")
        restore_button = QPushButton("Restore Defaults")
        cancel_button = QPushButton("Cancel")
        save_button.setToolTip("Apply these plot settings to the current session.")
        default_button.setToolTip("Save these plot settings as defaults.")
        restore_button.setToolTip("Restore saved default plot settings.")
        cancel_button.setToolTip("Close without changing plot settings.")
        save_button.clicked.connect(lambda: self._on_submit("save"))
        default_button.clicked.connect(lambda: self._on_submit("set_default"))
        restore_button.clicked.connect(self._on_restore_defaults)
        cancel_button.clicked.connect(self.reject)
        row_layout.addWidget(save_button)
        row_layout.addWidget(default_button)
        row_layout.addWidget(restore_button)
        row_layout.addStretch(1)
        row_layout.addWidget(cancel_button)
        root.addWidget(row)

        self._apply(session_params)
        self._update_dynamic_state()

    @property
    def selected_action(self) -> str | None:
        return self._selected_action

    @property
    def selected_params(self) -> dict[str, Any] | None:
        return self._selected_params

    @staticmethod
    def _normalize_ranges(value: Any) -> list[list[float]]:
        out: list[list[float]] = []
        if not isinstance(value, list):
            return out
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            try:
                start = float(item[0])
                end = float(item[1])
            except Exception:
                continue
            if not np.isfinite(start) or not np.isfinite(end):
                continue
            if start < 0.0 or end > 100.0 or end <= start:
                continue
            out.append([start, end])
        return out

    @staticmethod
    def _format_ranges(ranges: list[list[float]]) -> str:
        return "; ".join(f"{float(item[0]):g}-{float(item[1]):g}" for item in ranges)

    @staticmethod
    def _set_combo_value(combo: QComboBox, value: Any) -> None:
        idx = combo.findData(value)
        if idx < 0:
            label = "<empty>" if value is None else str(value)
            combo.addItem(f"Invalid: {label}", value)
            idx = combo.count() - 1
        combo.setCurrentIndex(idx)

    def _apply(self, params: dict[str, Any]) -> None:
        transform_mode = params.get("transform_mode", "none")
        if transform_mode is not None:
            transform_mode = normalize_feature_plot_transform_mode(transform_mode)
        normalize_mode = params.get("normalize_mode", "none")
        baseline_mode = params.get("baseline_mode", "mean")
        self._baseline_ranges = params.get("baseline_percent_ranges", [[0.0, 20.0]])
        colormap = params.get("colormap", "viridis")
        self._x_log_raw = params.get("x_log", False)
        self._y_log_raw = params.get("y_log", False)

        self._set_combo_value(self._transform_combo, transform_mode)
        self._set_combo_value(self._normalize_combo, normalize_mode)
        self._set_combo_value(self._baseline_mode_combo, baseline_mode)
        self._set_baseline_button_text()
        self._set_combo_value(self._colormap_combo, colormap)
        self._x_log_check.setChecked(bool(self._x_log_raw))
        self._y_log_check.setChecked(bool(self._y_log_raw))

    def _set_baseline_button_text(self) -> None:
        if self._baseline_button is None:
            return
        count = (
            len(self._baseline_ranges) if isinstance(self._baseline_ranges, list) else 0
        )
        self._baseline_button.setText(f"Baseline Configure... ({count})")

    def _update_dynamic_state(self) -> None:
        normalize_enabled = self._allow_normalize
        self._normalize_combo.setEnabled(normalize_enabled)
        normalize_mode = str(self._normalize_combo.currentData() or "none").strip()
        baseline_controls_enabled = normalize_enabled and normalize_mode != "none"
        self._baseline_mode_combo.setEnabled(baseline_controls_enabled)
        self._baseline_button.setEnabled(baseline_controls_enabled)
        self._x_log_check.setEnabled(self._allow_x_log)
        self._y_log_check.setEnabled(self._allow_y_log)
        self._refresh_validation()

    def _on_x_log_changed(self, _state: int) -> None:
        self._x_log_raw = bool(self._x_log_check.isChecked())
        self._refresh_validation()

    def _on_y_log_changed(self, _state: int) -> None:
        self._y_log_raw = bool(self._y_log_check.isChecked())
        self._refresh_validation()

    def _on_baseline_configure(self) -> None:
        valid_ranges = self._normalize_ranges(
            self._baseline_ranges if isinstance(self._baseline_ranges, list) else []
        )
        dialog = BaselineRangeConfigureDialog(
            current_ranges=tuple(
                [float(item[0]), float(item[1])] for item in valid_ranges
            ),
            parent=self,
        )
        if dialog.exec() != QDialog.Accepted:
            return
        self._baseline_ranges = [list(item) for item in dialog.selected_ranges]
        self._set_baseline_button_text()
        self._refresh_validation()

    def _on_restore_defaults(self) -> None:
        self._apply(self._default_params)
        self._update_dynamic_state()

    def _collect_draft(self) -> dict[str, Any]:
        return {
            "transform_mode": self._transform_combo.currentData(),
            "normalize_mode": self._normalize_combo.currentData(),
            "baseline_mode": self._baseline_mode_combo.currentData(),
            "baseline_percent_ranges": self._baseline_ranges,
            "colormap": self._colormap_combo.currentData(),
            "x_log": self._x_log_raw,
            "y_log": self._y_log_raw,
        }

    def _collect(self) -> dict[str, Any]:
        return validate_features_plot_advance_params(
            self._collect_draft(),
            allow_x_log=self._allow_x_log,
            allow_y_log=self._allow_y_log,
            allow_normalize=self._allow_normalize,
        )

    def _refresh_validation(self) -> None:
        error = ""
        try:
            self._collect()
        except ValueError as exc:
            error = str(exc)
        controls = (
            self._transform_combo,
            self._normalize_combo,
            self._baseline_mode_combo,
            self._baseline_button,
            self._colormap_combo,
            self._x_log_check,
            self._y_log_check,
        )
        for control in controls:
            set_control_validation_error(control, None)
        lowered = error.lower()
        if "transform" in lowered:
            set_control_validation_error(self._transform_combo, error)
        elif "normalize" in lowered:
            set_control_validation_error(self._normalize_combo, error)
        elif "baseline stat" in lowered:
            set_control_validation_error(self._baseline_mode_combo, error)
        elif "baseline range" in lowered:
            set_control_validation_error(self._baseline_button, error)
        elif "colormap" in lowered or "cmcrameri" in lowered:
            set_control_validation_error(self._colormap_combo, error)
        elif "x_log" in lowered:
            set_control_validation_error(self._x_log_check, error)
        elif "y_log" in lowered:
            set_control_validation_error(self._y_log_check, error)

    def _on_submit(self, action: str) -> None:
        draft = self._collect_draft()
        if action == "save":
            self._selected_action = action
            self._selected_params = draft
            self.accept()
            return
        try:
            payload = self._collect()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Plot Advance", f"Invalid advance params:\n{exc}")
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
            self._default_params = dict(payload)
            self._selected_action = action
            self._selected_params = payload
            return
