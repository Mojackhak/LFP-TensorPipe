"""Features plot-advance dialog."""

from __future__ import annotations

from .common import *  # noqa: F403
from .baseline_range import BaselineRangeConfigureDialog
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
        self._baseline_ranges: list[list[float]] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)

        self._transform_combo = QComboBox()
        for label, value in FEATURE_PLOT_TRANSFORM_OPTIONS:
            self._transform_combo.addItem(label, value)
        self._transform_combo.setToolTip(
            "Value transform applied before plotting."
        )
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

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)
        save_button = QPushButton("Save")
        default_button = QPushButton("Set as Default")
        restore_button = QPushButton("Restore Defaults")
        cancel_button = QPushButton("Cancel")
        save_button.setToolTip(
            "Apply these plot settings to the current session."
        )
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

    def _apply(self, params: dict[str, Any]) -> None:
        transform_mode = normalize_feature_plot_transform_mode(
            params.get("transform_mode", "none")
        )
        normalize_mode = str(params.get("normalize_mode", "none")).strip()
        baseline_mode = str(params.get("baseline_mode", "mean")).strip()
        self._baseline_ranges = self._normalize_ranges(
            params.get("baseline_percent_ranges")
        )
        colormap = str(params.get("colormap", "viridis")).strip()
        x_log = bool(params.get("x_log", False))
        y_log = bool(params.get("y_log", False))

        idx_transform = self._transform_combo.findData(transform_mode)
        self._transform_combo.setCurrentIndex(
            idx_transform if idx_transform >= 0 else 0
        )
        idx_norm = self._normalize_combo.findData(normalize_mode)
        self._normalize_combo.setCurrentIndex(idx_norm if idx_norm >= 0 else 0)
        idx_base = self._baseline_mode_combo.findData(baseline_mode)
        self._baseline_mode_combo.setCurrentIndex(idx_base if idx_base >= 0 else 0)
        self._set_baseline_button_text()
        idx_cmap = self._colormap_combo.findData(colormap)
        self._colormap_combo.setCurrentIndex(idx_cmap if idx_cmap >= 0 else 0)
        self._x_log_check.setChecked(x_log)
        self._y_log_check.setChecked(y_log)

    def _set_baseline_button_text(self) -> None:
        if self._baseline_button is None:
            return
        self._baseline_button.setText(
            f"Baseline Configure... ({len(self._baseline_ranges)})"
        )

    def _update_dynamic_state(self) -> None:
        normalize_enabled = self._allow_normalize
        self._normalize_combo.setEnabled(normalize_enabled)
        normalize_mode = str(self._normalize_combo.currentData() or "none").strip()
        baseline_controls_enabled = normalize_enabled and normalize_mode != "none"
        self._baseline_mode_combo.setEnabled(baseline_controls_enabled)
        self._baseline_button.setEnabled(baseline_controls_enabled)
        self._x_log_check.setEnabled(self._allow_x_log)
        self._y_log_check.setEnabled(self._allow_y_log)
        if not self._allow_x_log:
            self._x_log_check.setChecked(False)
        if not self._allow_y_log:
            self._y_log_check.setChecked(False)
        if not normalize_enabled:
            idx = self._normalize_combo.findData("none")
            if idx >= 0:
                self._normalize_combo.setCurrentIndex(idx)

    def _on_baseline_configure(self) -> None:
        dialog = BaselineRangeConfigureDialog(
            current_ranges=tuple(
                [float(item[0]), float(item[1])] for item in self._baseline_ranges
            ),
            parent=self,
        )
        if dialog.exec() != QDialog.Accepted:
            return
        self._baseline_ranges = [list(item) for item in dialog.selected_ranges]
        self._set_baseline_button_text()

    def _on_restore_defaults(self) -> None:
        self._apply(self._default_params)
        self._update_dynamic_state()

    def _collect(self) -> dict[str, Any]:
        transform_mode = str(self._transform_combo.currentData() or "none").strip()
        normalize_mode = str(self._normalize_combo.currentData() or "none").strip()
        baseline_mode = str(self._baseline_mode_combo.currentData() or "mean").strip()
        ranges = [list(item) for item in self._baseline_ranges]
        colormap = str(self._colormap_combo.currentData() or "viridis").strip()
        x_log = bool(self._x_log_check.isChecked() and self._allow_x_log)
        y_log = bool(self._y_log_check.isChecked() and self._allow_y_log)
        if not self._allow_normalize:
            normalize_mode = "none"
            ranges = []
        if normalize_mode != "none" and not ranges:
            raise ValueError(
                "Baseline ranges are required when normalization is enabled."
            )
        if colormap == "cmcrameri.vik":
            try:
                import cmcrameri  # noqa: F401
            except Exception as exc:  # noqa: BLE001
                raise ValueError("cmcrameri is required for cmcrameri.vik.") from exc
        return {
            "transform_mode": transform_mode,
            "normalize_mode": normalize_mode,
            "baseline_mode": baseline_mode,
            "baseline_percent_ranges": ranges,
            "colormap": colormap,
            "x_log": x_log,
            "y_log": y_log,
        }

    def _on_submit(self, action: str) -> None:
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
        self._selected_action = action
        self._selected_params = payload
        self.accept()
