"""Edit the channel region assignments for one imported recording."""

import csv
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from lfptensorpipe.app.localize.user_labels import UserLabels, read_labels


class UserLabelsDialog(QDialog):
    def __init__(
        self, context, channels, labels: UserLabels, *, read_only=False, parent=None
    ):
        super().__init__(parent)
        self.setWindowTitle("Channel Region Labels")
        self.resize(570, 420)
        self.channels = channels
        self.regions = labels.regions
        self.read_only = read_only
        self.original = labels.signature(channels)
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(f"Subject: {context.subject}    Record: {context.record}")
        )
        tools = QHBoxLayout()
        self.import_button = QPushButton("Import CSV...")
        self.import_button.clicked.connect(self.import_csv)
        self.add_button = QPushButton("Add region...")
        self.add_button.clicked.connect(self.add_region)
        for button in (self.import_button, self.add_button):
            button.setEnabled(not read_only)
            tools.addWidget(button)
        tools.addStretch()
        layout.addLayout(tools)
        self.table = QTableWidget()
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table.horizontalHeader().setMinimumSectionSize(64)
        self.table.horizontalHeader().setDefaultAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self.table.itemChanged.connect(self._item_changed)
        layout.addWidget(self.table)
        layout.addWidget(QLabel("Assignment method (optional)"))
        self.method = QLineEdit(labels.method)
        self.method.setReadOnly(read_only)
        self.method.textChanged.connect(self._refresh_save)
        layout.addWidget(self.method)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        self.save_button = buttons.button(QDialogButtonBox.StandardButton.Save)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._show_labels(labels)

    def _cell(self, checked, *, enabled=True):
        item = QTableWidgetItem()
        item.setFlags(
            Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable
            if enabled and not self.read_only
            else Qt.ItemFlag.NoItemFlags
        )
        item.setCheckState(
            Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        )
        return item

    def _show_labels(self, labels):
        self.regions = labels.regions
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.channels))
        self.table.setColumnCount(2 + len(self.regions))
        self.table.setHorizontalHeaderLabels(["Channel", "Assessed", *self.regions])
        for i, channel in enumerate(self.channels):
            name = QTableWidgetItem(channel)
            name.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.table.setItem(i, 0, name)
            hits = labels.assessed.get(channel)
            self.table.setItem(i, 1, self._cell(hits is not None))
            for col, region in enumerate(self.regions, 2):
                self.table.setItem(
                    i,
                    col,
                    self._cell(
                        hits is not None and region in hits, enabled=hits is not None
                    ),
                )
        self.table.blockSignals(False)
        self.method.setText(labels.method)
        self._refresh_save()

    def labels(self):
        rows = {}
        for i, channel in enumerate(self.channels):
            if self.table.item(i, 1).checkState() == Qt.CheckState.Checked:
                rows[channel] = {
                    region
                    for col, region in enumerate(self.regions, 2)
                    if self.table.item(i, col).checkState() == Qt.CheckState.Checked
                }
        return UserLabels(self.regions, rows, self.method.text().strip())

    def _refresh_save(self):
        labels = self.labels()
        self.save_button.setEnabled(
            not self.read_only
            and (bool(labels.regions) or not labels.assessed)
            and labels.signature(self.channels) != self.original
        )

    def _item_changed(self, item):
        if item.column() == 1:
            self.table.blockSignals(True)
            for col in range(2, self.table.columnCount()):
                self.table.setItem(
                    item.row(),
                    col,
                    self._cell(
                        False, enabled=item.checkState() == Qt.CheckState.Checked
                    ),
                )
            self.table.blockSignals(False)
        self._refresh_save()

    def add_region(self):
        name, ok = QInputDialog.getText(self, "Add region", "Region name")
        name = name.strip()
        if not ok:
            return
        if not name or name in self.regions:
            QMessageBox.warning(
                self, "Add region", "Enter a unique, nonempty region name."
            )
            return
        labels = self.labels()
        labels.regions = (*labels.regions, name)
        self._show_labels(labels)

    def import_csv(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Import channel labels", "", "CSV files (*.csv)"
        )
        if not filename:
            return
        previous = self.labels()
        try:
            labels = read_labels(Path(filename), self.channels, merge_into=previous)
        except (OSError, ValueError, csv.Error) as exc:
            QMessageBox.warning(self, "Import channel labels", str(exc))
            return
        self._show_labels(labels)
        for row, channel in enumerate(self.channels):
            if previous.assessed.get(channel) != labels.assessed.get(channel):
                item = self.table.item(row, 0)
                font = item.font()
                font.setBold(True)
                item.setFont(font)
                item.setToolTip("Changed by this CSV import.")
