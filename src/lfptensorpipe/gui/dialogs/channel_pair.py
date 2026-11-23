"""Channel-pair selection dialog."""

from __future__ import annotations

from .common import *  # noqa: F403


class ChannelPairDialog(QDialog):
    """Stateful bipolar channel-pair selector."""

    def __init__(
        self,
        *,
        channel_names: list[str],
        current_pairs: tuple[tuple[str, str], ...],
        current_names: tuple[str, ...],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Channel Pair")
        self.resize(560, 380)
        self._channel_names = channel_names
        self._pairs: list[tuple[str, str]] = list(current_pairs)
        self._pair_names: dict[tuple[str, str], str] = {}
        for idx, pair in enumerate(self._pairs):
            if idx < len(current_names):
                token = str(current_names[idx]).strip()
                self._pair_names[pair] = token if token else f"{pair[0]}-{pair[1]}"
            else:
                self._pair_names[pair] = f"{pair[0]}-{pair[1]}"
        self._current_signal: str | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        root.addWidget(QLabel("Configured outputs"))
        self._pair_table = QTableWidget(0, 3)
        self._pair_table.setHorizontalHeaderLabels(
            ["Signal channel", "Reference channel", "Output name"]
        )
        self._pair_table.verticalHeader().setVisible(False)
        self._pair_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._pair_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._pair_table.itemChanged.connect(self._on_pair_name_changed)
        pair_header = self._pair_table.horizontalHeader()
        pair_header.setSectionResizeMode(QHeaderView.Stretch)
        root.addWidget(self._pair_table, stretch=1)
        remove_row = QWidget()
        remove_row_layout = QHBoxLayout(remove_row)
        remove_row_layout.setContentsMargins(0, 0, 0, 0)
        remove_row_layout.addStretch(1)
        remove_button = QPushButton("Remove output")
        remove_button.clicked.connect(self._on_remove_pair)
        remove_row_layout.addWidget(remove_button)
        root.addWidget(remove_row)

        root.addWidget(QLabel("Select the signal channel, then the reference channel"))
        self._channel_list = QListWidget()
        self._channel_list.itemClicked.connect(self._on_channel_clicked)
        root.addWidget(self._channel_list, stretch=1)

        self._status_label = QLabel("Select a signal channel.")
        root.addWidget(self._status_label)

        footer = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        footer.accepted.connect(self._on_accept)
        footer.rejected.connect(self.reject)
        root.addWidget(footer)

        self._render_pairs()
        self._render_channels()

    @property
    def selected_pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._pairs)

    @property
    def selected_names(self) -> tuple[str, ...]:
        self._sync_pair_names_from_table()
        values: list[str] = []
        for pair in self._pairs:
            values.append(self._pair_names.get(pair, f"{pair[0]}-{pair[1]}").strip())
        return tuple(values)

    def _show_warning(self, title: str, message: str) -> int:
        return QMessageBox.warning(self, title, message)

    def _sync_pair_names_from_table(self) -> None:
        for row in range(self._pair_table.rowCount()):
            pair_item = self._pair_table.item(row, 0)
            name_item = self._pair_table.item(row, 2)
            if pair_item is None:
                continue
            pair_token = pair_item.data(Qt.UserRole)
            if not isinstance(pair_token, tuple) or len(pair_token) != 2:
                continue
            pair = (str(pair_token[0]), str(pair_token[1]))
            value = name_item.text().strip() if name_item is not None else ""
            self._pair_names[pair] = value if value else f"{pair[0]}-{pair[1]}"

    def _render_pairs(self) -> None:
        self._pair_table.blockSignals(True)
        self._pair_table.setRowCount(len(self._pairs))
        for row, pair in enumerate(self._pairs):
            pair_item = QTableWidgetItem(pair[0])
            pair_item.setData(Qt.UserRole, pair)
            pair_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            self._pair_table.setItem(row, 0, pair_item)

            reference_item = QTableWidgetItem(pair[1])
            reference_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            self._pair_table.setItem(row, 1, reference_item)

            name = self._pair_names.get(pair, f"{pair[0]}-{pair[1]}")
            name_item = QTableWidgetItem(name)
            self._pair_table.setItem(row, 2, name_item)
        self._pair_table.blockSignals(False)

    def _render_channels(self) -> None:
        self._channel_list.clear()
        for channel in self._channel_names:
            item = QListWidgetItem(channel)
            if channel == self._current_signal:
                item.setForeground(Qt.red)
            self._channel_list.addItem(item)

    def _on_channel_clicked(self, item: QListWidgetItem) -> None:
        channel = item.text()
        if self._current_signal is None:
            self._current_signal = channel
            self._status_label.setText(
                f"Signal channel {channel} selected. Select the reference channel."
            )
            self._render_channels()
            return

        if channel == self._current_signal:
            self._current_signal = None
            self._status_label.setText("Signal channel selection cleared.")
            self._render_channels()
            return

        pair = (self._current_signal, channel)
        if pair not in self._pairs:
            self._pairs.append(pair)
            self._pair_names[pair] = f"{pair[0]}-{pair[1]}"
        self._current_signal = None
        self._status_label.setText(
            "Output added. Select another signal channel or save."
        )
        self._render_pairs()
        self._render_channels()

    def _on_remove_pair(self) -> None:
        row = self._pair_table.currentRow()
        if row < 0:
            return
        pair = self._pairs.pop(row)
        self._pair_names.pop(pair, None)
        self._render_pairs()

    def _on_pair_name_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != 2:
            return
        pair_item = self._pair_table.item(item.row(), 0)
        if pair_item is None:
            return
        pair_token = pair_item.data(Qt.UserRole)
        if not isinstance(pair_token, tuple) or len(pair_token) != 2:
            return
        pair = (str(pair_token[0]), str(pair_token[1]))
        value = item.text().strip()
        self._pair_names[pair] = value if value else f"{pair[0]}-{pair[1]}"

    def _on_accept(self) -> None:
        self._sync_pair_names_from_table()
        names = [
            self._pair_names.get(pair, f"{pair[0]}-{pair[1]}").strip()
            for pair in self._pairs
        ]
        if not self._pairs:
            self._show_warning("Channel Pair", "Configure at least one output.")
            return
        if any(not name for name in names):
            self._show_warning("Channel Pair", "Each output requires a name.")
            return
        if len(set(names)) != len(names):
            self._show_warning("Channel Pair", "Output names must be unique.")
            return
        self.accept()
