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
        self._current_anode: str | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        root.addWidget(QLabel("Bipolar channels"))
        self._pair_table = QTableWidget(0, 2)
        self._pair_table.setHorizontalHeaderLabels(["Pair", "Name"])
        self._pair_table.verticalHeader().setVisible(False)
        self._pair_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._pair_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._pair_table.itemChanged.connect(self._on_pair_name_changed)
        pair_header = self._pair_table.horizontalHeader()
        pair_header.setSectionResizeMode(0, QHeaderView.Stretch)
        pair_header.setSectionResizeMode(1, QHeaderView.Stretch)
        root.addWidget(self._pair_table, stretch=1)
        remove_row = QWidget()
        remove_row_layout = QHBoxLayout(remove_row)
        remove_row_layout.setContentsMargins(0, 0, 0, 0)
        remove_row_layout.addStretch(1)
        remove_button = QPushButton("Remove Pair")
        remove_button.clicked.connect(self._on_remove_pair)
        remove_row_layout.addWidget(remove_button)
        root.addWidget(remove_row)

        root.addWidget(QLabel("Channels (click anode then cathode)"))
        self._channel_list = QListWidget()
        self._channel_list.itemClicked.connect(self._on_channel_clicked)
        root.addWidget(self._channel_list, stretch=1)

        self._status_label = QLabel("No anode selected.")
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
            name_item = self._pair_table.item(row, 1)
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
            pair_item = QTableWidgetItem(f"{pair[0]} -> {pair[1]}")
            pair_item.setData(Qt.UserRole, pair)
            pair_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            self._pair_table.setItem(row, 0, pair_item)

            name = self._pair_names.get(pair, f"{pair[0]}-{pair[1]}")
            name_item = QTableWidgetItem(name)
            self._pair_table.setItem(row, 1, name_item)
        self._pair_table.blockSignals(False)

    def _render_channels(self) -> None:
        self._channel_list.clear()
        for channel in self._channel_names:
            item = QListWidgetItem(channel)
            if channel == self._current_anode:
                item.setForeground(Qt.red)
            self._channel_list.addItem(item)

    def _on_channel_clicked(self, item: QListWidgetItem) -> None:
        channel = item.text()
        if self._current_anode is None:
            self._current_anode = channel
            self._status_label.setText(f"Anode selected: {channel}. Pick cathode.")
            self._render_channels()
            return

        if channel == self._current_anode:
            self._current_anode = None
            self._status_label.setText("Anode selection cancelled.")
            self._render_channels()
            return

        pair = (self._current_anode, channel)
        if pair not in self._pairs:
            self._pairs.append(pair)
            self._pair_names[pair] = f"{pair[0]}-{pair[1]}"
        self._current_anode = None
        self._status_label.setText("Pair added. Select next anode or Save.")
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
        if item.column() != 1:
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
            self._show_warning("Channel Pair", "At least one pair is required.")
            return
        if any(not name for name in names):
            self._show_warning(
                "Channel Pair", "Each bipolar pair requires a non-empty name."
            )
            return
        if len(set(names)) != len(names):
            self._show_warning("Channel Pair", "Bipolar channel names must be unique.")
            return
        self.accept()
