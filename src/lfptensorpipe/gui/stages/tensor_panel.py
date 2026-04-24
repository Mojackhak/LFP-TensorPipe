"""Tensor page and widget builders extracted from MainWindow."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

PAGE_MARGIN = 0
PAGE_SPACING = 0
GRID_SPACING = 0
TENSOR_METRICS: tuple[Any, ...] = ()


def _page_margin(owner) -> int:
    getter = getattr(owner, "_stage_panel_page_margin", None)
    return PAGE_MARGIN if getter is None else int(getter())


def _page_spacing(owner) -> int:
    getter = getattr(owner, "_stage_panel_page_spacing", None)
    return PAGE_SPACING if getter is None else int(getter())


def _grid_spacing(owner) -> int:
    getter = getattr(owner, "_stage_panel_grid_spacing", None)
    return GRID_SPACING if getter is None else int(getter())


def _tensor_metrics(owner) -> tuple[Any, ...]:
    getter = getattr(owner, "_stage_tensor_metric_specs", None)
    return TENSOR_METRICS if getter is None else tuple(getter())


def _build_tensor_page(self) -> QWidget:
    page = QWidget()
    layout = QVBoxLayout(page)
    page_margin = _page_margin(self)
    page_spacing = _page_spacing(self)
    layout.setContentsMargins(page_margin, page_margin, page_margin, page_margin)
    layout.setSpacing(page_spacing)

    layout.addWidget(self._page_title("Build Tensor"))

    columns = QWidget()
    columns_layout = QHBoxLayout(columns)
    columns_layout.setContentsMargins(0, 0, 0, 0)
    columns_layout.setSpacing(page_spacing)

    columns_layout.addWidget(self._build_tensor_metrics_block(), stretch=1)

    right_side = QWidget()
    right_side_layout = QVBoxLayout(right_side)
    right_side_layout.setContentsMargins(0, 0, 0, 0)
    right_side_layout.setSpacing(page_spacing)
    right_side_layout.addWidget(self._build_tensor_metric_params_block(), stretch=1)
    right_side_layout.addWidget(self._build_tensor_actions_block())
    columns_layout.addWidget(right_side, stretch=1)

    layout.addWidget(columns, stretch=1)
    return page


def _build_tensor_metrics_block(self) -> QGroupBox:
    block = QGroupBox("Metrics Selection")
    layout = QVBoxLayout(block)
    layout.setContentsMargins(8, 8, 8, 8)
    grid_spacing = _grid_spacing(self)
    layout.setSpacing(grid_spacing)

    self._tensor_metric_checks = {}
    self._tensor_metric_name_buttons = {}
    self._tensor_metric_indicators = {}
    grouped: dict[str, list[Any]] = {}
    for spec in _tensor_metrics(self):
        grouped.setdefault(spec.group_name, []).append(spec)

    for group_name, metrics in grouped.items():
        title = QLabel(group_name)
        title.setStyleSheet("font-weight: 700;")
        layout.addWidget(title)
        for spec in metrics:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(grid_spacing)
            indicator = QLabel()
            indicator.setFixedSize(12, 12)
            indicator.setToolTip(
                f"{spec.display_name} state: gray=not run, yellow=stale or failed, "
                "green=current params match successful output. Current: gray."
            )
            checkbox = QCheckBox("")
            checkbox.setToolTip("Include this metric in Run Build Tensor.")
            checkbox.stateChanged.connect(self._on_tensor_metric_selection_changed)
            name_button = QPushButton(spec.display_name)
            name_button.setCursor(Qt.PointingHandCursor)
            name_button.setFlat(True)
            name_button.setStyleSheet("text-align: left; padding: 0px; border: none;")
            name_button.setToolTip(
                "Click to edit this metric's parameters. This does not change run selection."
            )
            name_button.clicked.connect(
                lambda _checked=False, metric_key=spec.key: self._set_active_tensor_metric(
                    metric_key
                )
            )
            if not spec.supported:
                checkbox.setEnabled(False)
                checkbox.setToolTip("Pending implementation in next slices.")
                name_button.setEnabled(False)
            row_layout.addWidget(indicator)
            row_layout.addWidget(checkbox)
            row_layout.addWidget(name_button, stretch=1)
            layout.addWidget(row)
            self._tensor_metric_indicators[spec.key] = indicator
            self._tensor_metric_checks[spec.key] = checkbox
            self._tensor_metric_name_buttons[spec.key] = name_button
        layout.addSpacing(4)

    layout.addStretch(1)
    return block


def _build_tensor_bands_block(self) -> QGroupBox:
    block = QGroupBox("Bands")
    layout = QVBoxLayout(block)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(_grid_spacing(self))

    bands_defaults = self._load_tensor_bands_defaults()
    self._tensor_bands_table = QTableWidget(len(bands_defaults), 3)
    self._tensor_bands_table.setHorizontalHeaderLabels(["Band", "Start", "End"])
    self._tensor_bands_table.verticalHeader().setVisible(False)
    self._tensor_bands_table.setToolTip(
        "Reference band definitions used by metrics that require named bands."
    )
    header = self._tensor_bands_table.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.Stretch)
    self._tensor_bands_table.setMinimumHeight(180)
    for row_idx, band in enumerate(bands_defaults):
        self._tensor_bands_table.setItem(
            row_idx, 0, QTableWidgetItem(str(band["name"]))
        )
        self._tensor_bands_table.setItem(
            row_idx, 1, QTableWidgetItem(str(band["start"]))
        )
        self._tensor_bands_table.setItem(row_idx, 2, QTableWidgetItem(str(band["end"])))
    layout.addWidget(self._tensor_bands_table)
    return block


def _build_tensor_metric_params_block(self) -> QGroupBox:
    block = QGroupBox("Metric Parameter Panel")
    layout = QFormLayout(block)
    self._tensor_metric_params_form = layout
    layout.setLabelAlignment(Qt.AlignLeft)
    layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
    block.setContentsMargins(8, 8, 8, 8)

    self._tensor_metric_title_label = QLabel("")
    layout.addRow("Metric", self._tensor_metric_title_label)

    self._tensor_low_freq_edit = QLineEdit()
    self._tensor_low_freq_edit.setToolTip(
        "Lower frequency bound. Must be >= preprocess low."
    )
    self._tensor_high_freq_edit = QLineEdit()
    self._tensor_high_freq_edit.setToolTip(
        "Upper bound. Must be <= min(preprocess high, Nyquist)."
    )
    self._tensor_step_edit = QLineEdit()
    self._tensor_step_edit.setToolTip("Frequency spacing for computation grid.")
    self._tensor_time_resolution_edit = QLineEdit()
    self._tensor_time_resolution_edit.setToolTip(
        "Window duration for spectral estimation."
    )
    self._tensor_hop_edit = QLineEdit()
    self._tensor_hop_edit.setToolTip("Step size between adjacent windows.")
    self._tensor_method_combo = QComboBox()
    for method in self._tensor_supported_methods():
        self._tensor_method_combo.addItem(method, method)
    self._tensor_method_combo.setToolTip("Spectral method (morlet/multitaper)")
    self._tensor_freq_range_edit = QLineEdit()
    self._tensor_freq_range_edit.setPlaceholderText("e.g., 2, 80")
    self._tensor_freq_range_edit.setToolTip(
        "Frequency range passed to periodic/aperiodic decomposition."
    )
    self._tensor_percentile_edit = QLineEdit()
    self._tensor_percentile_edit.setToolTip("Burst threshold percentile.")
    self._tensor_min_cycles_basic_edit = QLineEdit()
    self._tensor_min_cycles_basic_edit.setToolTip(
        "Minimum cycles used for burst detection."
    )
    self._tensor_bands_configure_button = QPushButton("Bands Configure...")
    self._tensor_bands_configure_button.setToolTip(
        "Edit named bands for the active metric."
    )
    self._tensor_bands_configure_button.clicked.connect(self._on_tensor_bands_configure)

    layout.addRow("Low freq (Hz)", self._tensor_low_freq_edit)
    layout.addRow("High freq (Hz)", self._tensor_high_freq_edit)
    layout.addRow("Step (Hz)", self._tensor_step_edit)
    layout.addRow("Time resolution (s)", self._tensor_time_resolution_edit)
    layout.addRow("Hop (s)", self._tensor_hop_edit)
    layout.addRow("Method", self._tensor_method_combo)
    layout.addRow("SpecParam freq range", self._tensor_freq_range_edit)
    layout.addRow("Bands", self._tensor_bands_configure_button)
    layout.addRow("Percentile", self._tensor_percentile_edit)
    layout.addRow("Min cycles", self._tensor_min_cycles_basic_edit)

    self._tensor_channels_button = QPushButton("Select Channels (0/0)")
    self._tensor_channels_button.setToolTip("Choose channels for the active metric.")
    self._tensor_channels_button.clicked.connect(self._on_tensor_channels_select)
    layout.addRow("Channels", self._tensor_channels_button)
    self._tensor_pairs_button = QPushButton("Select Pairs (0/0)")
    self._tensor_pairs_button.setToolTip("Choose channel pairs for the active metric.")
    self._tensor_pairs_button.clicked.connect(self._on_tensor_pairs_select)
    layout.addRow("Pairs", self._tensor_pairs_button)
    self._tensor_advance_button = QPushButton("Advance")
    self._tensor_advance_button.setToolTip(
        "Open advanced settings for the active metric."
    )
    self._tensor_advance_button.clicked.connect(self._on_tensor_metric_advance)
    layout.addRow("Advanced", self._tensor_advance_button)

    self._tensor_metric_notice_label = QLabel("")
    self._tensor_metric_notice_label.setWordWrap(True)
    self._tensor_metric_notice_label.setToolTip(
        "Status of the active metric in the current slice."
    )
    layout.addRow("Status", self._tensor_metric_notice_label)

    self._tensor_basic_param_widgets = {
        "low_freq_hz": self._tensor_low_freq_edit,
        "high_freq_hz": self._tensor_high_freq_edit,
        "freq_step_hz": self._tensor_step_edit,
        "time_resolution_s": self._tensor_time_resolution_edit,
        "hop_s": self._tensor_hop_edit,
        "method": self._tensor_method_combo,
        "freq_range_hz": self._tensor_freq_range_edit,
        "bands": self._tensor_bands_configure_button,
        "percentile": self._tensor_percentile_edit,
        "min_cycles": self._tensor_min_cycles_basic_edit,
    }

    self._ensure_tensor_metric_state_from_defaults(self._record_context())
    self._set_active_tensor_metric(self._tensor_active_metric_key)
    return block


def _build_tensor_actions_block(self) -> QGroupBox:
    block = QGroupBox("Run Build Tensor")
    layout = QVBoxLayout(block)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(4)
    self._tensor_mask_edge_checkbox = QCheckBox("Mask Edge Effects")
    self._tensor_mask_edge_checkbox.setChecked(True)
    self._tensor_mask_edge_checkbox.setToolTip(
        "Mask edge artifacts in the tensor output."
    )
    self._tensor_import_button = QPushButton("Import Configs...")
    self._tensor_import_button.setToolTip("Load tensor configuration.")
    self._tensor_import_button.clicked.connect(self._on_tensor_import_config)
    self._tensor_export_button = QPushButton("Export Configs...")
    self._tensor_export_button.setToolTip("Save tensor configuration.")
    self._tensor_export_button.clicked.connect(self._on_tensor_export_config)
    self._tensor_run_button = QPushButton("Build Tensor")
    self._tensor_run_button.setToolTip("Run tensor building for the selected metrics.")
    self._tensor_run_button.clicked.connect(self._on_tensor_run)
    self._tensor_run_button.setEnabled(False)

    config_row = QWidget()
    config_row_layout = QHBoxLayout(config_row)
    config_row_layout.setContentsMargins(0, 0, 0, 0)
    grid_spacing = _grid_spacing(self)
    config_row_layout.setSpacing(grid_spacing)
    config_row_layout.addWidget(self._tensor_import_button, 1)
    config_row_layout.addWidget(self._tensor_export_button, 1)

    run_row = QWidget()
    run_row_layout = QHBoxLayout(run_row)
    run_row_layout.setContentsMargins(0, 0, 0, 0)
    run_row_layout.setSpacing(grid_spacing)
    run_row_layout.addWidget(self._tensor_mask_edge_checkbox)
    run_row_layout.addStretch(1)
    run_row_layout.addWidget(self._tensor_run_button)

    layout.addWidget(config_row)
    layout.addWidget(run_row)
    return block
