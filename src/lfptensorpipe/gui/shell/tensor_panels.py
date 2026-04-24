"""Tensor-related shell and panel assembly MainWindow methods."""

from __future__ import annotations

from lfptensorpipe.gui.shell.common import (
    ALIGNMENT_METHODS,
    Any,
    ECG_METHODS,
    GRID_SPACING,
    PAGE_MARGIN,
    PAGE_SPACING,
    PANEL_SPACING,
    QAbstractItemView,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    Qt,
    RecordContext,
    STAGE_PANEL_MARGIN,
    STAGE_PANEL_SPACING,
    STAGE_SPECS,
    TENSOR_METRICS,
    _stage_alignment_panel,
    _stage_features_panel,
    _stage_localize_panel,
    _stage_preproc_panel,
    _stage_tensor_panel,
)


def _tensor_metric_specs() -> tuple[Any, ...]:
    return tuple(TENSOR_METRICS)


class MainWindowTensorPanelsMixin:
    def _stage_panel_page_margin(self) -> int:
        return PAGE_MARGIN

    def _stage_panel_page_spacing(self) -> int:
        return PAGE_SPACING

    def _stage_panel_grid_spacing(self) -> int:
        return GRID_SPACING

    def _stage_preproc_ecg_methods(self) -> tuple[str, ...]:
        return tuple(ECG_METHODS)

    def _stage_tensor_metric_specs(self) -> tuple[Any, ...]:
        return _tensor_metric_specs()

    def _stage_alignment_methods(self) -> tuple[Any, ...]:
        return tuple(ALIGNMENT_METHODS)

    def _build_left_column(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, PAGE_MARGIN, 0, PAGE_MARGIN)
        layout.setSpacing(PANEL_SPACING)

        dataset_panel = self._build_dataset_panel()
        dataset_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        layout.addWidget(dataset_panel, stretch=1)
        layout.addWidget(self._build_localize_panel(), stretch=0)
        layout.addWidget(self._build_stages_panel(), stretch=0)
        return widget

    def _build_dataset_panel(self) -> QGroupBox:
        panel = QGroupBox("Dataset")
        grid = QGridLayout(panel)
        grid.setHorizontalSpacing(GRID_SPACING)
        grid.setVerticalSpacing(GRID_SPACING)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(2, 1)

        self._project_combo = QComboBox()
        self._project_combo.addItem("Select project", None)
        self._project_combo.currentIndexChanged.connect(self._on_project_changed)
        self._project_combo.setToolTip("Select a project workspace.")

        self._subject_combo = QComboBox()
        self._subject_combo.addItem("Select subject", None)
        self._subject_combo.currentIndexChanged.connect(self._on_subject_changed)
        self._subject_combo.setToolTip("Select a subject in the current project.")

        self._record_list = QListWidget()
        self._record_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._record_list.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._record_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._record_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._record_list.setMaximumHeight(120)
        self._record_list.itemSelectionChanged.connect(self._on_record_changed)
        self._record_list.setToolTip(
            "Select a record for preprocessing and downstream stages."
        )

        self._project_add_button = QPushButton("+")
        self._subject_add_button = QPushButton("+")
        self._record_add_button = QPushButton("+")
        self._record_delete_button = QPushButton("-")
        self._record_rename_button = QPushButton("R")
        self._project_add_button.clicked.connect(self._on_project_add)
        self._subject_add_button.clicked.connect(self._on_subject_add)
        self._record_add_button.clicked.connect(self._on_record_add)
        self._record_delete_button.clicked.connect(self._on_record_delete)
        self._record_rename_button.clicked.connect(self._on_record_rename)
        self._project_add_button.setEnabled(True)
        self._subject_add_button.setEnabled(False)
        self._record_add_button.setEnabled(False)
        self._record_delete_button.setEnabled(False)
        self._record_rename_button.setEnabled(False)
        self._project_add_button.setToolTip(
            "Add an existing project folder to recent projects."
        )
        self._subject_add_button.setToolTip(
            "Create a new subject folder under the current project."
        )
        self._record_add_button.setToolTip(
            "Import a new record into the current subject."
        )
        self._record_delete_button.setToolTip(
            "Delete the selected record and all derived artifacts."
        )
        self._record_rename_button.setToolTip(
            "Rename the selected record and preserve artifacts."
        )

        grid.addWidget(QLabel("Project"), 0, 0)
        grid.addWidget(self._project_combo, 0, 1)
        grid.addWidget(self._project_add_button, 0, 2)

        grid.addWidget(QLabel("Subject"), 1, 0)
        grid.addWidget(self._subject_combo, 1, 1)
        grid.addWidget(self._subject_add_button, 1, 2)

        record_label = QLabel("Record")
        grid.addWidget(record_label, 2, 0, alignment=Qt.AlignTop)
        grid.addWidget(self._record_list, 2, 1)

        record_actions = QWidget()
        record_actions_layout = QVBoxLayout(record_actions)
        record_actions_layout.setContentsMargins(0, 0, 0, 0)
        record_actions_layout.setSpacing(GRID_SPACING)
        record_actions_layout.addWidget(self._record_add_button)
        record_actions_layout.addWidget(self._record_delete_button)
        record_actions_layout.addWidget(self._record_rename_button)
        record_actions_layout.addStretch(1)
        grid.addWidget(record_actions, 2, 2)

        return panel

    def _build_localize_panel(self) -> QGroupBox:
        return _stage_localize_panel._build_localize_panel(self)

    def _build_stages_panel(self) -> QGroupBox:
        panel = QGroupBox("Stages")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(
            STAGE_PANEL_MARGIN,
            STAGE_PANEL_MARGIN,
            STAGE_PANEL_MARGIN,
            STAGE_PANEL_MARGIN,
        )
        layout.setSpacing(STAGE_PANEL_SPACING)

        for spec in STAGE_SPECS:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(STAGE_PANEL_SPACING)

            indicator = QLabel()
            indicator.setFixedSize(12, 12)
            indicator.setToolTip(
                "Stage state: gray=not ready, yellow=stale or blocked, green=ready/current."
            )
            self._stage_indicators[spec.key] = indicator

            button = QPushButton(spec.display_name)
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            button.setToolTip(
                f"Open {spec.display_name}. Enabled only when all upstream stages are green."
            )
            button.clicked.connect(
                lambda checked=False, stage_key=spec.key: self.route_to_stage(stage_key)
            )
            self._stage_buttons[spec.key] = button

            row_layout.addWidget(indicator)
            row_layout.addWidget(button, stretch=1)
            layout.addWidget(row)

        return panel

    def _build_stage_stack(self) -> QStackedWidget:
        stack = QStackedWidget()

        stack.addWidget(self._build_preprocess_page())
        self._stage_page_index["preproc"] = 0

        stack.addWidget(self._build_tensor_page())
        self._stage_page_index["tensor"] = 1

        stack.addWidget(self._build_alignment_page())
        self._stage_page_index["alignment"] = 2

        stack.addWidget(self._build_features_page())
        self._stage_page_index["features"] = 3

        return stack

    def _build_preprocess_page(self) -> QWidget:
        return _stage_preproc_panel._build_preprocess_page(self)

    def _build_tensor_page(self) -> QWidget:
        return _stage_tensor_panel._build_tensor_page(self)

    def _build_tensor_metrics_block(self) -> QGroupBox:
        return _stage_tensor_panel._build_tensor_metrics_block(self)

    def _build_tensor_bands_block(self) -> QGroupBox:
        return _stage_tensor_panel._build_tensor_bands_block(self)

    def _build_tensor_metric_params_block(self) -> QGroupBox:
        return _stage_tensor_panel._build_tensor_metric_params_block(self)

    def _build_tensor_actions_block(self) -> QGroupBox:
        return _stage_tensor_panel._build_tensor_actions_block(self)

    def _build_alignment_page(self) -> QWidget:
        return _stage_alignment_panel._build_alignment_page(self)

    def _build_alignment_paradigm_block(self) -> QGroupBox:
        return _stage_alignment_panel._build_alignment_paradigm_block(self)

    def _build_alignment_method_block(self) -> QGroupBox:
        return _stage_alignment_panel._build_alignment_method_block(self)

    def _build_alignment_epoch_inspector_block(self) -> QGroupBox:
        return _stage_alignment_panel._build_alignment_epoch_inspector_block(self)

    def _build_features_page(self) -> QWidget:
        return _stage_features_panel._build_features_page(self)

    def _build_features_paradigm_block(self) -> QGroupBox:
        return _stage_features_panel._build_features_paradigm_block(self)

    def _build_features_phases_block(self) -> QGroupBox:
        return _stage_features_panel._build_features_phases_block(self)

    def _build_features_run_block(self) -> QGroupBox:
        return _stage_features_panel._build_features_run_block(self)

    def _build_features_available_block(self) -> QGroupBox:
        return _stage_features_panel._build_features_available_block(self)

    def _build_features_subset_block(self) -> QGroupBox:
        return _stage_features_panel._build_features_subset_block(self)

    def _build_features_plot_block(self) -> QGroupBox:
        return _stage_features_panel._build_features_plot_block(self)

    def _record_context(self) -> RecordContext | None:
        if (
            self._current_project is None
            or self._current_subject is None
            or self._current_record is None
        ):
            return None
        return RecordContext(
            project_root=self._current_project,
            subject=self._current_subject,
            record=self._current_record,
        )
