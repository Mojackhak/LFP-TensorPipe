"""Import-sync configuration dialog."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from lfptensorpipe.io.sync import (
    ImportSyncState,
    MarkerPair,
    MarkerPoint,
    PeakDetectConfig,
    SyncFigureData,
    build_sync_summary_figure,
    detect_raw_channel_markers,
    load_external_markers_from_audio,
    load_external_markers_from_csv,
)

from .common import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    Qt,
    build_import_sync_seed,
    build_import_synced_raw,
    estimate_import_sync,
)


def _clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.setParent(None)
            widget.deleteLater()


class SyncDetectAdvanceDialog(QDialog):
    """Edit advanced peak-detection parameters for one sync source."""

    def __init__(
        self,
        *,
        title: str,
        current_config: PeakDetectConfig,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(320, 180)
        self._current_config = current_config
        self._selected_config: PeakDetectConfig | None = None

        root = QVBoxLayout(self)
        form = QFormLayout()
        self._start_edit = QLineEdit()
        self._stop_edit = QLineEdit()
        self._height_edit = QLineEdit()
        self._prominence_edit = QLineEdit()
        self._start_edit.setPlaceholderText("auto")
        self._stop_edit.setPlaceholderText("auto")
        self._height_edit.setPlaceholderText("auto")
        self._prominence_edit.setPlaceholderText("auto")
        if current_config.search_range_s is not None:
            self._start_edit.setText(str(current_config.search_range_s[0]))
            self._stop_edit.setText(str(current_config.search_range_s[1]))
        if current_config.height is not None:
            self._height_edit.setText(str(current_config.height))
        if current_config.prominence is not None:
            self._prominence_edit.setText(str(current_config.prominence))
        form.addRow("Search start (s)", self._start_edit)
        form.addRow("Search stop (s)", self._stop_edit)
        form.addRow("Height", self._height_edit)
        form.addRow("Prominence", self._prominence_edit)
        root.addLayout(form)

        footer = QHBoxLayout()
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        save_button = QPushButton("Save")
        save_button.clicked.connect(self._on_save)
        footer.addWidget(cancel_button)
        footer.addStretch(1)
        footer.addWidget(save_button)
        root.addLayout(footer)

    @property
    def selected_config(self) -> PeakDetectConfig | None:
        return self._selected_config

    @staticmethod
    def _float_or_none(edit: QLineEdit) -> float | None:
        text = edit.text().strip()
        return None if not text else float(text)

    def _on_save(self) -> None:
        try:
            start = self._float_or_none(self._start_edit)
            stop = self._float_or_none(self._stop_edit)
            search_range_s = None
            if start is not None or stop is not None:
                if start is None or stop is None:
                    raise ValueError(
                        "Search range requires both start and stop values."
                    )
                search_range_s = (start, stop)
            self._selected_config = PeakDetectConfig(
                search_range_s=search_range_s,
                min_distance_s=self._current_config.min_distance_s,
                height=self._float_or_none(self._height_edit),
                prominence=self._float_or_none(self._prominence_edit),
                max_peaks=self._current_config.max_peaks,
                use_abs=self._current_config.use_abs,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Sync Advance", str(exc))
            return
        self.accept()


class SyncFigurePreviewDialog(QDialog):
    """Show one sync summary figure in a popup dialog."""

    def __init__(
        self,
        *,
        figure,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Sync Figure Preview")
        self.resize(1000, 480)
        self._figure = figure

        root = QVBoxLayout(self)
        self._canvas = FigureCanvasQTAgg(figure)
        root.addWidget(self._canvas, stretch=1)
        footer = QHBoxLayout()
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        footer.addStretch(1)
        footer.addWidget(close_button)
        root.addLayout(footer)


class ImportSyncDialog(QDialog):
    """Configure import-time synchronization before confirm-import."""

    def __init__(
        self,
        *,
        raw: Any,
        current_state: ImportSyncState | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Sync Import Signal")
        self.resize(980, 760)
        self._raw = raw
        self._state = current_state
        self._lfp_markers: list[MarkerPoint] = []
        self._external_markers: list[MarkerPoint] = []
        self._pairs: list[MarkerPair] = []
        self._estimate = None
        self._lfp_figure_data: SyncFigureData | None = None
        self._external_figure_data: SyncFigureData | None = None
        self._saved_state: ImportSyncState | None = None
        self._lfp_advanced_detect_config = PeakDetectConfig()
        self._external_advanced_detect_config = PeakDetectConfig()

        root = QVBoxLayout(self)
        body = QVBoxLayout()
        self._body_layout = body
        top = QHBoxLayout()
        self._top_layout = top
        body.addLayout(top, stretch=1)

        self._lfp_panel = self._build_side_panel(side="lfp")
        self._external_panel = self._build_side_panel(side="external")
        top.addWidget(self._lfp_panel, stretch=1)
        top.addWidget(self._external_panel, stretch=1)

        pair_box = QWidget()
        self._pair_box = pair_box
        pair_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        pair_layout = QVBoxLayout(pair_box)
        self._pair_layout = pair_layout
        pair_actions = QHBoxLayout()
        self._pair_button = QPushButton("Pair Selected")
        self._pair_button.clicked.connect(self._on_pair_selected)
        self._remove_pair_button = QPushButton("Remove Pair")
        self._remove_pair_button.clicked.connect(self._on_remove_pair)
        self._auto_pair_button = QPushButton("Auto Pair by Order")
        self._auto_pair_button.clicked.connect(self._on_auto_pair)
        self._correct_sfreq_check = QCheckBox("Correct sfreq")
        pair_actions.addWidget(self._pair_button)
        pair_actions.addWidget(self._remove_pair_button)
        pair_actions.addWidget(self._auto_pair_button)
        pair_actions.addStretch(1)
        pair_actions.addWidget(self._correct_sfreq_check)
        pair_layout.addLayout(pair_actions)
        self._pair_table = QTableWidget(0, 6)
        self._pair_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._pair_table.setHorizontalHeaderLabels(
            ["Pair", "LFP idx", "LFP t", "External idx", "External t", "Delta t"]
        )
        pair_layout.addWidget(self._pair_table, stretch=1)
        self._summary_label = QLabel("Summary: no sync estimate yet")
        pair_layout.addWidget(self._summary_label)
        body.addWidget(pair_box, stretch=1)
        root.addLayout(body, stretch=1)

        footer = QHBoxLayout()
        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.clicked.connect(self.reject)
        self._estimate_button = QPushButton("Sync")
        self._estimate_button.clicked.connect(self._on_estimate)
        self._save_button = QPushButton("Save")
        self._save_button.clicked.connect(self._on_save)
        footer.addWidget(self._cancel_button)
        footer.addStretch(1)
        footer.addWidget(self._estimate_button)
        footer.addWidget(self._save_button)
        root.addLayout(footer)

        self._load_initial_state()
        self._update_lfp_channel_enabled_state()
        self._refresh_all()

    @property
    def selected_state(self) -> ImportSyncState | None:
        return self._saved_state

    def _build_side_panel(self, *, side: str) -> QWidget:
        box = QWidget()
        box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QGridLayout(box)
        prefix = "LFP" if side == "lfp" else "External"
        layout.addWidget(QLabel(f"{prefix} Markers"), 0, 0, 1, 2)
        source_combo = QComboBox()
        source_combo.addItems(
            ["Parsed annotations", "Channel peaks"]
            if side == "lfp"
            else ["CSV times", "Audio"]
        )
        setattr(self, f"_{side}_source_combo", source_combo)
        layout.addWidget(QLabel("Source" if side == "lfp" else "Input Type"), 1, 0)
        layout.addWidget(source_combo, 1, 1)
        row = 2
        if side == "lfp":
            channel_label = QLabel("Channel")
            combo = QComboBox()
            combo.addItems([str(name) for name in self._raw.ch_names])
            self._lfp_source_combo.currentTextChanged.connect(
                self._on_lfp_source_changed
            )
            setattr(self, "_lfp_channel_combo", combo)
            setattr(self, "_lfp_channel_label", channel_label)
            layout.addWidget(channel_label, row, 0)
            layout.addWidget(combo, row, 1)
            row += 1
        else:
            file_row = QWidget()
            file_layout = QHBoxLayout(file_row)
            file_layout.setContentsMargins(0, 0, 0, 0)
            edit = QLineEdit()
            browse = QPushButton("Browse")
            browse.clicked.connect(self._on_external_browse)
            file_layout.addWidget(edit, stretch=1)
            file_layout.addWidget(browse)
            self._external_path_edit = edit
            layout.addWidget(QLabel("File Path"), row, 0)
            layout.addWidget(file_row, row, 1)
            row += 1
        distance_row = QWidget()
        distance_layout = QHBoxLayout(distance_row)
        distance_layout.setContentsMargins(0, 0, 0, 0)
        distance_layout.setSpacing(4)
        distance_edit = QLineEdit()
        distance_edit.setText("1.0")
        setattr(self, f"_{side}_distance_edit", distance_edit)
        advance_button = QPushButton("Advance")
        advance_button.clicked.connect(lambda: self._on_detect_advance(side))
        setattr(self, f"_{side}_advance_button", advance_button)
        distance_label = QLabel("Min distance")
        setattr(self, f"_{side}_distance_label", distance_label)
        distance_layout.addWidget(distance_edit, stretch=1)
        distance_layout.addWidget(advance_button)
        layout.addWidget(distance_label, row, 0)
        layout.addWidget(distance_row, row, 1)
        row += 1
        action_row = QHBoxLayout()
        detect_button = QPushButton(
            "Detect / Reload" if side == "lfp" else "Load / Detect"
        )
        detect_button.clicked.connect(
            self._on_lfp_reload if side == "lfp" else self._on_external_reload
        )
        add_button = QPushButton("Add")
        add_button.clicked.connect(lambda: self._on_add_marker(side))
        delete_button = QPushButton("Delete")
        delete_button.clicked.connect(lambda: self._on_delete_marker(side))
        action_row.addWidget(detect_button)
        action_row.addWidget(add_button)
        action_row.addWidget(delete_button)
        layout.addLayout(action_row, row, 0, 1, 2)
        row += 1
        table = QTableWidget(0, 4)
        table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        table.setHorizontalHeaderLabels(["Idx", "Time (s)", "Label", "Source"])
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setSelectionMode(QTableWidget.SingleSelection)
        setattr(self, f"_{side}_table", table)
        setattr(self, f"_{side}_table_row", row)
        layout.addWidget(table, row, 0, 1, 2)
        layout.setRowStretch(row, 1)
        return box

    def _load_initial_state(self) -> None:
        if self._state is not None:
            self._lfp_markers = list(self._state.lfp_markers)
            self._external_markers = list(self._state.external_markers)
            self._pairs = list(self._state.pairs)
            self._estimate = self._state.estimate
            self._lfp_figure_data = self._state.lfp_figure_data
            self._external_figure_data = self._state.external_figure_data
            self._lfp_source_combo.setCurrentText(
                "Channel peaks"
                if self._state.lfp_source_kind == "channel_peaks"
                else "Parsed annotations"
            )
            self._external_source_combo.setCurrentText(
                "Audio" if self._state.external_source_kind == "audio" else "CSV times"
            )
            if self._state.lfp_source_path:
                self._lfp_channel_combo.setCurrentText(self._state.lfp_source_path)
            self._external_path_edit.setText(self._state.external_source_path)
            if self._state.lfp_detect_config is not None:
                self._apply_detect_config("lfp", self._state.lfp_detect_config)
            if self._state.external_detect_config is not None:
                self._apply_detect_config(
                    "external", self._state.external_detect_config
                )
            self._correct_sfreq_check.setChecked(
                bool(self._state.estimate.correct_sfreq)
            )
            return
        self._lfp_markers = list(build_import_sync_seed(self._raw))
        self._external_markers = []

    def _apply_detect_config(self, side: str, config: PeakDetectConfig) -> None:
        getattr(self, f"_{side}_distance_edit").setText(str(config.min_distance_s))
        setattr(
            self,
            f"_{side}_advanced_detect_config",
            PeakDetectConfig(
                search_range_s=config.search_range_s,
                min_distance_s=config.min_distance_s,
                height=config.height,
                prominence=config.prominence,
                max_peaks=config.max_peaks,
                use_abs=config.use_abs,
            ),
        )

    def _float_or_none(self, edit: QLineEdit) -> float | None:
        text = edit.text().strip()
        return None if not text else float(text)

    def _build_detect_config(self, side: str) -> PeakDetectConfig:
        advanced = getattr(self, f"_{side}_advanced_detect_config")
        return PeakDetectConfig(
            search_range_s=advanced.search_range_s,
            min_distance_s=(
                self._float_or_none(getattr(self, f"_{side}_distance_edit")) or 1.0
            ),
            height=advanced.height,
            prominence=advanced.prominence,
            max_peaks=advanced.max_peaks,
            use_abs=advanced.use_abs,
        )

    def _on_detect_advance(self, side: str) -> None:
        dialog = SyncDetectAdvanceDialog(
            title="LFP Advance" if side == "lfp" else "External Advance",
            current_config=self._build_detect_config(side),
            parent=self,
        )
        if dialog.exec() != QDialog.Accepted or dialog.selected_config is None:
            return
        setattr(self, f"_{side}_advanced_detect_config", dialog.selected_config)

    def _update_lfp_channel_enabled_state(self) -> None:
        enabled = self._lfp_source_combo.currentText() == "Channel peaks"
        self._lfp_channel_label.setEnabled(enabled)
        self._lfp_channel_combo.setEnabled(enabled)
        self._lfp_distance_label.setEnabled(enabled)
        self._lfp_distance_edit.setEnabled(enabled)
        self._lfp_advance_button.setEnabled(enabled)

    def _on_lfp_source_changed(self, _text: str) -> None:
        self._update_lfp_channel_enabled_state()

    def _on_external_browse(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select marker file",
            "",
            "All Files (*.*)",
        )
        if selected:
            self._external_path_edit.setText(str(Path(selected).expanduser().resolve()))

    def _reset_estimate(self) -> None:
        self._estimate = None
        self._saved_state = None

    def _reindex_markers(
        self,
        markers: list[MarkerPoint],
        *,
        source: str,
    ) -> list[MarkerPoint]:
        ordered = sorted(markers, key=lambda item: float(item.time_s))
        return [
            MarkerPoint(
                marker_index=index,
                time_s=float(marker.time_s),
                label=str(marker.label).strip() or f"{source}_{index}",
                source=str(marker.source).strip() or source,
            )
            for index, marker in enumerate(ordered)
        ]

    def _next_manual_marker_suffix(self, side: str) -> int:
        markers = self._lfp_markers if side == "lfp" else self._external_markers
        prefix = f"{side}_manual_"
        suffixes = [
            int(str(marker.label)[len(prefix) :])
            for marker in markers
            if str(marker.label).startswith(prefix)
            and str(marker.label)[len(prefix) :].isdigit()
        ]
        return (max(suffixes) + 1) if suffixes else 0

    def _on_lfp_reload(self) -> None:
        self._reset_estimate()
        self._pairs = []
        if self._lfp_source_combo.currentText() == "Parsed annotations":
            self._lfp_markers = list(build_import_sync_seed(self._raw))
            self._lfp_figure_data = None
        else:
            markers, figure_data = detect_raw_channel_markers(
                self._raw,
                self._lfp_channel_combo.currentText().strip(),
                self._build_detect_config("lfp"),
            )
            self._lfp_markers = list(markers)
            self._lfp_figure_data = figure_data
        self._refresh_all()

    def _on_external_reload(self) -> None:
        self._reset_estimate()
        self._pairs = []
        path = self._external_path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "Sync", "External marker file path is required.")
            return
        if self._external_source_combo.currentText() == "Audio":
            markers, figure_data = load_external_markers_from_audio(
                path,
                self._build_detect_config("external"),
            )
            self._external_markers = list(markers)
            self._external_figure_data = figure_data
        else:
            self._external_markers = list(load_external_markers_from_csv(path))
            self._external_figure_data = SyncFigureData(
                kind="events",
                source_label=Path(path).name,
                marker_times_s=tuple(
                    marker.time_s for marker in self._external_markers
                ),
                title=f"External Marker Source ({Path(path).name})",
            )
        self._refresh_all()

    def _on_add_marker(self, side: str) -> None:
        time_s, ok = QInputDialog.getDouble(
            self,
            "Add Marker",
            "Marker time (s)",
            0.0,
            0.0,
            1_000_000_000.0,
            6,
        )
        if not ok:
            return
        markers = self._lfp_markers if side == "lfp" else self._external_markers
        suffix = self._next_manual_marker_suffix(side)
        markers.append(
            MarkerPoint(
                marker_index=len(markers),
                time_s=float(time_s),
                label=f"{side}_manual_{suffix}",
                source=f"{side}_manual",
            )
        )
        self._pairs = []
        if side == "lfp":
            self._lfp_markers = self._reindex_markers(markers, source="lfp")
        else:
            self._external_markers = self._reindex_markers(markers, source="external")
        self._reset_estimate()
        self._refresh_all()

    def _on_delete_marker(self, side: str) -> None:
        table = self._lfp_table if side == "lfp" else self._external_table
        row = table.currentRow()
        if row < 0:
            return
        markers = list(self._lfp_markers if side == "lfp" else self._external_markers)
        markers.pop(row)
        self._pairs = []
        if side == "lfp":
            self._lfp_markers = self._reindex_markers(markers, source="lfp")
        else:
            self._external_markers = self._reindex_markers(markers, source="external")
        self._reset_estimate()
        self._refresh_all()

    def _on_pair_selected(self) -> None:
        lfp_row = self._lfp_table.currentRow()
        ext_row = self._external_table.currentRow()
        if lfp_row < 0 or ext_row < 0:
            return
        lfp_index = self._lfp_markers[lfp_row].marker_index
        ext_index = self._external_markers[ext_row].marker_index
        if any(
            pair.lfp_marker_index == lfp_index
            or pair.external_marker_index == ext_index
            for pair in self._pairs
        ):
            QMessageBox.warning(self, "Sync", "Selected markers are already paired.")
            return
        self._pairs.append(
            MarkerPair(
                pair_id=len(self._pairs),
                lfp_marker_index=lfp_index,
                external_marker_index=ext_index,
            )
        )
        self._reset_estimate()
        self._refresh_all()

    def _on_auto_pair(self) -> None:
        count = min(len(self._lfp_markers), len(self._external_markers))
        self._pairs = [
            MarkerPair(
                pair_id=index,
                lfp_marker_index=self._lfp_markers[index].marker_index,
                external_marker_index=self._external_markers[index].marker_index,
            )
            for index in range(count)
        ]
        self._reset_estimate()
        self._refresh_all()

    def _on_remove_pair(self) -> None:
        row = self._pair_table.currentRow()
        if row < 0:
            return
        self._pairs.pop(row)
        self._pairs = [
            MarkerPair(
                pair_id=index,
                lfp_marker_index=pair.lfp_marker_index,
                external_marker_index=pair.external_marker_index,
            )
            for index, pair in enumerate(self._pairs)
        ]
        self._reset_estimate()
        self._refresh_all()

    def _on_estimate(self) -> None:
        try:
            estimate = estimate_import_sync(
                lfp_markers=self._lfp_markers,
                external_markers=self._external_markers,
                pairs=self._pairs,
                sfreq_before_hz=float(self._raw.info["sfreq"]),
                correct_sfreq=self._correct_sfreq_check.isChecked(),
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Sync", str(exc))
            return
        self._estimate = estimate
        fig = build_sync_summary_figure(
            lfp_markers=tuple(self._lfp_markers),
            external_markers=tuple(self._external_markers),
            pairs=tuple(self._pairs),
            estimate=estimate,
            lfp_figure_data=self._lfp_figure_data,
            external_figure_data=self._external_figure_data,
        )
        self._refresh_summary()
        self._show_preview_dialog(fig)

    def _on_save(self) -> None:
        if self._estimate is None:
            QMessageBox.warning(self, "Sync", "Run Sync before saving.")
            return
        try:
            _ = build_import_synced_raw(self._raw, self._estimate)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Sync", f"Failed to build synced raw: {exc}")
            return
        lfp_source_kind = (
            "channel_peaks"
            if self._lfp_source_combo.currentText() == "Channel peaks"
            else "parsed_annotations"
        )
        external_source_kind = (
            "audio"
            if self._external_source_combo.currentText() == "Audio"
            else "csv_times"
        )
        self._saved_state = ImportSyncState(
            lfp_markers=tuple(self._lfp_markers),
            external_markers=tuple(self._external_markers),
            pairs=tuple(self._pairs),
            estimate=self._estimate,
            lfp_source_kind=lfp_source_kind,
            external_source_kind=external_source_kind,
            lfp_source_path=(
                self._lfp_channel_combo.currentText().strip()
                if lfp_source_kind == "channel_peaks"
                else ""
            ),
            external_source_path=self._external_path_edit.text().strip(),
            lfp_detect_config=(
                self._build_detect_config("lfp")
                if lfp_source_kind == "channel_peaks"
                else None
            ),
            external_detect_config=(
                self._build_detect_config("external")
                if external_source_kind == "audio"
                else None
            ),
            lfp_figure_data=self._lfp_figure_data,
            external_figure_data=self._external_figure_data,
        )
        self.accept()

    def _populate_marker_table(
        self,
        table: QTableWidget,
        markers: list[MarkerPoint],
    ) -> None:
        table.setRowCount(len(markers))
        for row, marker in enumerate(markers):
            for col, value in enumerate(
                (
                    marker.marker_index,
                    f"{marker.time_s:.6f}",
                    marker.label,
                    marker.source,
                )
            ):
                item = QTableWidgetItem(str(value))
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                table.setItem(row, col, item)

    def _refresh_pairs(self) -> None:
        lfp_by_index = {marker.marker_index: marker for marker in self._lfp_markers}
        ext_by_index = {
            marker.marker_index: marker for marker in self._external_markers
        }
        deltas = (
            list(self._estimate.deltas_before_sync_s)
            if self._estimate is not None
            else []
        )
        self._pair_table.setRowCount(len(self._pairs))
        for row, pair in enumerate(self._pairs):
            values = (
                pair.pair_id,
                pair.lfp_marker_index,
                f"{lfp_by_index[pair.lfp_marker_index].time_s:.6f}",
                pair.external_marker_index,
                f"{ext_by_index[pair.external_marker_index].time_s:.6f}",
                (
                    f"{deltas[row]:.6f}"
                    if row < len(deltas)
                    else (
                        f"{lfp_by_index[pair.lfp_marker_index].time_s - ext_by_index[pair.external_marker_index].time_s:.6f}"
                    )
                ),
            )
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self._pair_table.setItem(row, col, item)

    def _show_preview_dialog(self, figure) -> None:
        preview_dialog = SyncFigurePreviewDialog(figure=figure, parent=self)
        preview_dialog.exec()

    def _refresh_summary(self) -> None:
        if self._estimate is None:
            self._summary_label.setText("Summary: no sync estimate yet")
            self._save_button.setEnabled(False)
            return
        summary = (
            f"Summary: pairs={self._estimate.pair_count}, lag={self._estimate.lag_s:.6f} s, "
            f"sfreq={self._estimate.sfreq_after_hz:.6f} Hz"
        )
        if self._estimate.rmse_ms is not None:
            summary += f", rmse={self._estimate.rmse_ms:.3f} ms"
        if self._estimate.r2 is not None:
            summary += f", r2={self._estimate.r2:.4f}"
        self._summary_label.setText(summary)
        self._save_button.setEnabled(True)

    def _refresh_all(self) -> None:
        self._populate_marker_table(self._lfp_table, self._lfp_markers)
        self._populate_marker_table(self._external_table, self._external_markers)
        self._refresh_pairs()
        self._refresh_summary()
