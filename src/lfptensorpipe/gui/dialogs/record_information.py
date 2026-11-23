"""Read-only recording summary and import-time acquisition contact editor."""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lfptensorpipe.app.dataset.record_info import (
    UNKNOWN,
    InputChannel,
    RecordInformation,
)


class RecordInformationDialog(QDialog):
    def __init__(self, info: RecordInformation, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Record Information")
        self.resize(700, 430)
        layout = QVBoxLayout(self)
        summary = QGridLayout()
        summary.setColumnStretch(1, 2)
        summary.setColumnStretch(3, 1)
        layout.addLayout(summary)
        provenance = info.provenance
        source_fields = [
            ("Subject", info.subject),
            (
                "Import type",
                (
                    "SceneRay"
                    if provenance.import_type == "Sceneray"
                    else provenance.import_type
                ),
            ),
            ("Source file", provenance.source_file),
        ]
        for label, value in (
            ("Metadata file", provenance.metadata_file),
            ("Marker file", provenance.marker_file),
            ("Sync source", info.sync_source),
        ):
            if value:
                source_fields.append((label, value))
        rate = info.sampling_rate
        output_count = len(info.output_rows)
        count_text = (
            f"{output_count} output{'s' if output_count != 1 else ''}"
            if info.has_fif or info.output_rows
            else UNKNOWN
        )
        if info.input_rows and (
            len(info.input_rows) != output_count or not info.has_fif
        ):
            count_text = f"{len(info.input_rows)} inputs → {count_text}"
        elif not info.input_rows and info.has_fif:
            count_text = f"Unknown inputs → {count_text}"
        record_fields = [
            ("Record", info.record),
            ("Sampling rate", f"{rate:.7g} Hz" if rate is not None else None),
            (
                "Duration",
                (
                    f"{info.samples / rate:.3f} s"
                    if info.samples is not None and rate
                    else None
                ),
            ),
            ("Channels", count_text),
        ]
        if info.pre_sync_rate is not None:
            record_fields.append(
                ("Sampling rate before Sync", f"{info.pre_sync_rate:.7g} Hz")
            )
        source_rate = provenance.source_sampling_rate_hz
        displayed_rate = info.pre_sync_rate if info.pre_sync_rate is not None else rate
        if source_rate is not None and (
            displayed_rate is None or f"{source_rate:.7g}" != f"{displayed_rate:.7g}"
        ):
            record_fields.append(("Source sampling rate", f"{source_rate:.7g} Hz"))
        for column, fields in ((0, source_fields), (2, record_fields)):
            for row_index, (label, value) in enumerate(fields):
                text = QLabel(str(value) if value is not None else UNKNOWN)
                text.setTextFormat(Qt.PlainText)
                text.setTextInteractionFlags(Qt.TextSelectableByMouse)
                text.setWordWrap(True)
                summary.addWidget(QLabel(label), row_index, column)
                summary.addWidget(text, row_index, column + 1)
        if info.notice:
            note = QLabel(info.notice)
            note.setTextFormat(Qt.PlainText)
            note.setWordWrap(True)
            layout.addWidget(note)
        tabs = QTabWidget()
        for name, rows in [("Output", info.output_rows), ("Input", info.input_rows)]:
            table = QTableWidget(len(rows), 4)
            table.setHorizontalHeaderLabels(
                [name, "Recording contact", "Reference contact", "Recorded signal"]
            )
            table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            table.setSelectionBehavior(QAbstractItemView.SelectRows)
            table.verticalHeader().hide()
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            table.horizontalHeader().setStretchLastSection(True)
            for i, row in enumerate(rows):
                for col, text in enumerate(
                    (
                        row.name,
                        row.recording_contact,
                        row.reference_contact,
                        row.recorded_signal,
                    )
                ):
                    table.setItem(i, col, QTableWidgetItem(text))
            if name == "Input":
                for i, channel in enumerate(provenance.inputs):
                    table.item(i, 0).setToolTip(channel.source_label)
                    table.item(i, 1).setToolTip(
                        channel.recording_contact_source or UNKNOWN
                    )
                    table.item(i, 2).setToolTip(
                        channel.reference_contact_source or UNKNOWN
                    )
            else:
                operations = {op.name: op for op in provenance.outputs}
                for i, row in enumerate(rows):
                    op = operations.get(row.name)
                    if op is not None and op.operation_source:
                        table.item(i, 3).setToolTip(op.operation_source)
            tabs.addTab(table, f"{name} channels")
        layout.addWidget(tabs, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class AcquisitionReferenceDialog(QDialog):
    def __init__(
        self, channels: list[InputChannel], parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Acquisition Reference")
        self.resize(550, 360)
        self._original = channels
        layout = QVBoxLayout(self)
        common = QHBoxLayout()
        self._common_reference = QLineEdit()
        fill = QPushButton("Fill missing")
        fill.setToolTip(
            "Fill missing references in selected rows, or all rows when none are selected."
        )
        fill.clicked.connect(self._fill_missing)
        common.addWidget(QLabel("Common reference"))
        common.addWidget(self._common_reference)
        common.addWidget(fill)
        layout.addLayout(common)
        self._table = QTableWidget(len(channels), 3)
        self._table.setHorizontalHeaderLabels(
            ["Input", "Recording contact", "Reference contact"]
        )
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.verticalHeader().hide()
        for i, channel in enumerate(channels):
            name = QTableWidgetItem(channel.name)
            name.setFlags(name.flags() & ~Qt.ItemIsEditable)
            name.setToolTip(channel.source_label)
            self._table.setItem(i, 0, name)
            self._table.setItem(i, 1, QTableWidgetItem(channel.recording_contact or ""))
            self._table.setItem(i, 2, QTableWidgetItem(channel.reference_contact or ""))
        layout.addWidget(self._table)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _fill_missing(self) -> None:
        ref = self._common_reference.text().strip()
        if not ref:
            return
        rows = [index.row() for index in self._table.selectionModel().selectedRows()]
        for i in rows or range(self._table.rowCount()):
            item = self._table.item(i, 2)
            if not item.text().strip():
                item.setText(ref)

    def channels(self) -> list[InputChannel]:
        result = []
        for i, old in enumerate(self._original):
            row = replace(old)
            for col, field in ((1, "recording_contact"), (2, "reference_contact")):
                value = self._table.item(i, col).text().strip() or None
                if value != getattr(old, field):
                    setattr(row, field, value)
                    setattr(row, field + "_source", "operator")
            result.append(row)
        return result
