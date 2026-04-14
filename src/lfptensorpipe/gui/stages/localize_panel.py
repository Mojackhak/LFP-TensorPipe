"""Localize panel builder extracted from MainWindow."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)

from lfptensorpipe.gui.stages.indicator_group_box import IndicatorGroupBox

GRID_SPACING = 0


def _grid_spacing(owner) -> int:
    getter = getattr(owner, "_stage_panel_grid_spacing", None)
    return GRID_SPACING if getter is None else int(getter())


def _build_localize_panel(self) -> QGroupBox:
    panel = IndicatorGroupBox("Localize")
    grid = QGridLayout(panel)
    grid_spacing = _grid_spacing(self)
    grid.setHorizontalSpacing(grid_spacing)
    grid.setVerticalSpacing(grid_spacing)

    self._localize_indicator = panel.indicator_label()
    self._localize_indicator.setToolTip(
        "Localize panel state: gray=not run, yellow=failed or draft differs, "
        "green=current draft matches last apply."
    )

    self._localize_match_button = QPushButton("Configure...")
    self._localize_match_button.clicked.connect(self._on_localize_match)
    self._localize_match_button.setToolTip(
        "Map each record channel to Lead-DBS contacts for this record."
    )
    self._localize_match_status_label = QLabel("0/0 mapped")
    self._localize_match_status_label.setToolTip(
        "Mapped channels / total channels. Apply requires full mapping."
    )
    self._localize_matlab_status_label = QLabel("MATLAB: Idle")
    self._localize_matlab_status_label.setToolTip(
        "MATLAB runtime status. Actions auto-connect when needed."
    )
    self._space_value_edit = QLineEdit()
    self._space_value_edit.setReadOnly(True)
    self._space_value_edit.setPlaceholderText("Space inferred from normalization")
    self._localize_elmodel_edit = QLineEdit()
    self._localize_elmodel_edit.setReadOnly(True)
    self._localize_elmodel_edit.setPlaceholderText("Load from reconstruction")

    self._localize_atlas_button = QPushButton("Configure...")
    self._localize_atlas_button.clicked.connect(self._on_localize_atlas_configure)
    self._localize_atlas_button.setToolTip(
        "Choose the atlas and interested regions for this record."
    )
    self._localize_atlas_summary_label = QLabel("0/0 regions selected")
    self._localize_atlas_summary_label.setToolTip(
        "Saved interested-region count for the current Localize atlas config."
    )

    self._localize_apply_button = QPushButton("Apply")
    self._localize_import_button = QPushButton("Import Configs...")
    self._localize_export_button = QPushButton("Export Configs...")
    self._contact_viewer_button = QPushButton("Contact Viewer")
    self._localize_apply_button.setToolTip(
        "Generate representative localize artifacts for the current record."
    )
    self._localize_import_button.setToolTip(
        "Import saved Localize atlas and match config for the current record."
    )
    self._localize_export_button.setToolTip(
        "Export saved Localize atlas and match config for the current record."
    )
    self._contact_viewer_button.setToolTip(
        "Open Contact Viewer with current atlas and representative CSV."
    )
    self._localize_apply_button.clicked.connect(self._on_localize_apply)
    self._localize_import_button.clicked.connect(self._on_localize_import_config)
    self._localize_export_button.clicked.connect(self._on_localize_export_config)
    self._contact_viewer_button.clicked.connect(self._on_contact_viewer)
    self._localize_apply_button.setEnabled(False)
    self._localize_import_button.setEnabled(False)
    self._localize_export_button.setEnabled(False)
    if self._localize_match_button is not None:
        self._localize_match_button.setEnabled(False)
    if self._localize_atlas_button is not None:
        self._localize_atlas_button.setEnabled(False)
    self._contact_viewer_button.setEnabled(False)

    match_row = QWidget()
    match_layout = QHBoxLayout(match_row)
    match_layout.setContentsMargins(0, 0, 0, 0)
    match_layout.setSpacing(grid_spacing)
    match_layout.addWidget(self._localize_match_button)
    match_layout.addWidget(self._localize_match_status_label)
    match_layout.addStretch(1)

    if self._localize_matlab_status_label is not None:
        grid.addWidget(self._localize_matlab_status_label, 0, 0, 1, 2)
    grid.addWidget(QLabel("Match"), 1, 0)
    grid.addWidget(match_row, 1, 1)
    grid.addWidget(QLabel("Atlas"), 2, 0)
    atlas_row = QWidget()
    atlas_row_layout = QHBoxLayout(atlas_row)
    atlas_row_layout.setContentsMargins(0, 0, 0, 0)
    atlas_row_layout.setSpacing(grid_spacing)
    atlas_row_layout.addWidget(self._localize_atlas_button)
    atlas_row_layout.addWidget(self._localize_atlas_summary_label)
    atlas_row_layout.addStretch(1)
    grid.addWidget(atlas_row, 2, 1)

    config_row = QWidget()
    config_row_layout = QHBoxLayout(config_row)
    config_row_layout.setContentsMargins(0, 0, 0, 0)
    config_row_layout.setSpacing(grid_spacing)
    config_row_layout.addWidget(self._localize_import_button)
    config_row_layout.addWidget(self._localize_export_button)
    config_row_layout.addStretch(1)
    grid.addWidget(config_row, 3, 0, 1, 2)

    button_row = QWidget()
    button_row_layout = QHBoxLayout(button_row)
    button_row_layout.setContentsMargins(0, 0, 0, 0)
    button_row_layout.setSpacing(grid_spacing)
    button_row_layout.addWidget(self._localize_apply_button)
    button_row_layout.addWidget(self._contact_viewer_button)

    grid.addWidget(button_row, 4, 0, 1, 2)
    return panel
