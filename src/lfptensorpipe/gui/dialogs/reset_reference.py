"""Reset-reference dialog."""

from __future__ import annotations

from typing import Callable

from .common import (
    ACTION_PAYLOAD_ROLE,
    QAbstractItemView,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    Qt,
    QVBoxLayout,
    QWidget,
    make_action_table_item,
)
from .dataset_types import ResetReferenceRow


def _display_endpoint(value: str) -> str:
    """Render a reset-reference endpoint for table display."""
    return value if value else "-"


class ResetReferenceDialog(QDialog):
    """Configure reset-reference rows for record import."""

    def __init__(
        self,
        *,
        channel_names: tuple[str, ...],
        current_rows: tuple[ResetReferenceRow, ...],
        default_rows: tuple[ResetReferenceRow, ...] = (),
        set_default_callback: (
            Callable[[tuple[ResetReferenceRow, ...]], None] | None
        ) = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Reset Reference")
        self.resize(460, 300)
        self._channel_names = tuple(channel_names)
        self._set_default_callback = set_default_callback
        self._default_rows = self._normalize_rows(default_rows)
        self._rows: list[ResetReferenceRow] = list(self._normalize_rows(current_rows))

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        top = QWidget()
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(6)
        top_layout.addWidget(QLabel(f"Parsed channels: {len(self._channel_names)}"))
        top_layout.addStretch(1)
        top_layout.addWidget(QLabel("Search"))
        self._search_edit = QLineEdit()
        self._search_edit.textChanged.connect(self._on_search_changed)
        self._search_edit.setToolTip("Filter channels and configured pairs.")
        top_layout.addWidget(self._search_edit)
        root.addWidget(top)

        splitter = QSplitter(Qt.Horizontal)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)
        self._channel_list = QListWidget()
        self._channel_list.itemClicked.connect(self._on_channel_clicked)
        self._channel_list.setToolTip(
            "Click once to set Anode; click a second channel to set Cathode."
        )
        left_layout.addWidget(self._channel_list, stretch=1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)
        self._pair_table = QTableWidget(0, 5)
        self._pair_table.setHorizontalHeaderLabels(
            ["#", "Anode", "Cathode", "Name", "Action"]
        )
        self._pair_table.verticalHeader().setVisible(False)
        self._pair_table.setSelectionMode(QAbstractItemView.NoSelection)
        self._pair_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._pair_table.cellClicked.connect(self._on_pair_table_clicked)
        self._pair_table.setToolTip("Current reset-reference pairs for this import.")
        header = self._pair_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        right_layout.addWidget(self._pair_table, stretch=1)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)
        splitter.setSizes([90, 360])
        root.addWidget(splitter, stretch=1)

        draft = QFrame()
        draft_layout = QGridLayout(draft)
        draft_layout.setContentsMargins(0, 0, 0, 0)
        draft_layout.setHorizontalSpacing(6)
        draft_layout.setVerticalSpacing(4)
        draft_layout.addWidget(QLabel("Anode"), 0, 0)
        self._draft_anode_edit = QLineEdit()
        self._draft_anode_edit.textChanged.connect(self._on_draft_anode_changed)
        self._draft_anode_edit.setToolTip("Draft anode channel.")
        draft_layout.addWidget(self._draft_anode_edit, 0, 1)
        draft_layout.addWidget(QLabel("Cathode"), 1, 0)
        self._draft_cathode_edit = QLineEdit()
        self._draft_cathode_edit.textChanged.connect(self._update_apply_state)
        self._draft_cathode_edit.setToolTip("Draft cathode channel.")
        draft_layout.addWidget(self._draft_cathode_edit, 1, 1)
        draft_layout.addWidget(QLabel("Name"), 2, 0)
        name_row = QWidget()
        name_row_layout = QHBoxLayout(name_row)
        name_row_layout.setContentsMargins(0, 0, 0, 0)
        name_row_layout.setSpacing(6)
        self._draft_name_edit = QLineEdit()
        self._draft_name_edit.textChanged.connect(self._update_apply_state)
        self._draft_name_edit.setToolTip("Output channel name for the new pair.")
        self._draft_apply_button = QPushButton("Apply")
        self._draft_apply_button.clicked.connect(self._on_apply_draft)
        self._draft_apply_button.setToolTip("Add the draft pair to the configuration.")
        name_row_layout.addWidget(self._draft_name_edit, stretch=1)
        name_row_layout.addWidget(self._draft_apply_button)
        draft_layout.addWidget(name_row, 2, 1)
        root.addWidget(draft)

        footer = QWidget()
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(0, 0, 0, 0)
        footer_layout.setSpacing(6)
        self._clear_draft_button = QPushButton("Clear Draft")
        self._clear_all_button = QPushButton("Clear All")
        self._set_default_button = QPushButton("Set as Default")
        self._restore_default_button = QPushButton("Restore Default")
        self._clear_draft_button.clicked.connect(self._on_clear_draft)
        self._clear_all_button.clicked.connect(self._on_clear_all)
        self._set_default_button.clicked.connect(self._on_set_default)
        self._restore_default_button.clicked.connect(self._on_restore_default)
        self._clear_draft_button.setToolTip("Clear the current draft pair.")
        self._clear_all_button.setToolTip("Remove all configured pairs.")
        self._set_default_button.setToolTip(
            "Save current reset-reference pairs as defaults."
        )
        self._restore_default_button.setToolTip(
            "Restore saved reset-reference default pairs."
        )
        cancel_button = QPushButton("Cancel")
        save_button = QPushButton("Save")
        cancel_button.clicked.connect(self.reject)
        save_button.clicked.connect(self._on_save)
        cancel_button.setToolTip("Close without applying pair changes.")
        save_button.setToolTip("Use the current reset-reference pairs for this import.")
        footer_layout.addWidget(self._clear_draft_button)
        footer_layout.addWidget(self._clear_all_button)
        footer_layout.addWidget(self._set_default_button)
        footer_layout.addWidget(self._restore_default_button)
        footer_layout.addStretch(1)
        footer_layout.addWidget(cancel_button)
        footer_layout.addWidget(save_button)
        root.addWidget(footer)

        self._render_channels()
        self._render_pairs()
        self._update_apply_state()
        self._update_footer_states()

    @property
    def selected_rows(self) -> tuple[ResetReferenceRow, ...]:
        return tuple(self._rows)

    def _show_warning(self, title: str, message: str) -> int:
        return QMessageBox.warning(self, title, message)

    def _normalize_rows(
        self,
        rows: tuple[ResetReferenceRow, ...],
    ) -> tuple[ResetReferenceRow, ...]:
        valid_channels = set(self._channel_names)
        seen_pairs: set[tuple[str, str]] = set()
        seen_names: set[str] = set()
        normalized: list[ResetReferenceRow] = []
        for row in rows:
            anode = row.anode.strip()
            cathode = row.cathode.strip()
            name = row.name.strip()
            if not name or (not anode and not cathode):
                continue
            if anode and anode not in valid_channels:
                continue
            if cathode and cathode not in valid_channels:
                continue
            if anode and cathode and anode == cathode:
                continue
            pair_key = (anode, cathode)
            if pair_key in seen_pairs or name in seen_names:
                continue
            seen_pairs.add(pair_key)
            seen_names.add(name)
            normalized.append(
                ResetReferenceRow(
                    anode=anode,
                    cathode=cathode,
                    name=name,
                )
            )
        return tuple(normalized)

    def _on_search_changed(self, _text: str) -> None:
        self._render_channels()
        self._render_pairs()

    def _search_token(self) -> str:
        return self._search_edit.text().strip().lower()

    def _render_channels(self) -> None:
        token = self._search_token()
        draft_anode = self._draft_anode_edit.text().strip()
        self._channel_list.clear()
        for channel in self._channel_names:
            if token and token not in channel.lower():
                continue
            item = QListWidgetItem(channel)
            if channel == draft_anode:
                item.setForeground(Qt.red)
            self._channel_list.addItem(item)

    def _render_pairs(self) -> None:
        token = self._search_token()
        rows: list[ResetReferenceRow] = []
        for row in self._rows:
            if not token:
                rows.append(row)
                continue
            joined = f"{row.anode} {row.cathode} {row.name}".lower()
            if token in joined:
                rows.append(row)

        self._pair_table.setRowCount(len(rows))
        for row_idx, row in enumerate(rows, start=0):
            idx_item = QTableWidgetItem(str(row_idx + 1))
            idx_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self._pair_table.setItem(row_idx, 0, idx_item)
            self._pair_table.setItem(
                row_idx,
                1,
                QTableWidgetItem(_display_endpoint(row.anode)),
            )
            self._pair_table.setItem(
                row_idx,
                2,
                QTableWidgetItem(_display_endpoint(row.cathode)),
            )
            self._pair_table.setItem(row_idx, 3, QTableWidgetItem(row.name))
            self._pair_table.setItem(
                row_idx,
                4,
                make_action_table_item(
                    "Del",
                    (row.anode, row.cathode, row.name),
                    tool_tip="Delete this pair.",
                ),
            )
        self._update_footer_states()

    def _on_draft_anode_changed(self, _text: str) -> None:
        self._render_channels()
        self._update_apply_state()

    def _on_channel_clicked(self, item: QListWidgetItem) -> None:
        channel = item.text().strip()
        draft_anode = self._draft_anode_edit.text().strip()
        if not draft_anode:
            self._draft_anode_edit.setText(channel)
            if not self._draft_name_edit.text().strip():
                self._draft_name_edit.setText(channel)
            return
        if channel == draft_anode:
            self._on_clear_draft()
            return
        self._draft_cathode_edit.setText(channel)
        self._update_apply_state()

    def _update_apply_state(self) -> None:
        anode = self._draft_anode_edit.text().strip()
        cathode = self._draft_cathode_edit.text().strip()
        name = self._draft_name_edit.text().strip()
        self._draft_apply_button.setEnabled(bool((anode or cathode) and name))

    def _update_footer_states(self) -> None:
        has_rows = len(self._rows) > 0
        self._clear_all_button.setEnabled(has_rows)
        self._clear_draft_button.setEnabled(True)
        self._set_default_button.setEnabled(True)
        self._restore_default_button.setEnabled(True)

    def _on_apply_draft(self) -> None:
        anode = self._draft_anode_edit.text().strip()
        cathode = self._draft_cathode_edit.text().strip()
        name = self._draft_name_edit.text().strip()
        if not anode and not cathode:
            self._show_warning(
                "Reset Reference", "At least one of anode or cathode is required."
            )
            return
        if not name:
            self._show_warning("Reset Reference", "Name is required.")
            return
        if anode and anode not in self._channel_names:
            self._show_warning("Reset Reference", f"Unknown anode channel: {anode}")
            return
        if cathode and cathode not in self._channel_names:
            self._show_warning("Reset Reference", f"Unknown cathode channel: {cathode}")
            return
        if cathode and cathode == anode:
            self._show_warning(
                "Reset Reference", "Anode and cathode cannot be identical."
            )
            return
        pair_key = (anode, cathode)
        existing_keys = {(row.anode, row.cathode) for row in self._rows}
        if pair_key in existing_keys:
            self._show_warning("Reset Reference", "Duplicate pair is not allowed.")
            return
        existing_names = {row.name for row in self._rows}
        if name in existing_names:
            self._show_warning(
                "Reset Reference", "Duplicate output channel name is not allowed."
            )
            return
        self._rows.append(ResetReferenceRow(anode=anode, cathode=cathode, name=name))
        self._on_clear_draft()
        self._render_pairs()

    def _on_pair_table_clicked(self, row: int, column: int) -> None:
        if column != 4:
            return
        action_item = self._pair_table.item(row, column)
        if action_item is None:
            return
        payload = action_item.data(ACTION_PAYLOAD_ROLE)
        if not isinstance(payload, (tuple, list)) or len(payload) != 3:
            return
        self._on_remove_row(
            ResetReferenceRow(
                anode=str(payload[0]),
                cathode=str(payload[1]),
                name=str(payload[2]),
            )
        )

    def _on_remove_row(self, row: ResetReferenceRow) -> None:
        self._rows = [current for current in self._rows if current != row]
        self._render_pairs()

    def _on_clear_draft(self) -> None:
        self._draft_anode_edit.clear()
        self._draft_cathode_edit.clear()
        self._draft_name_edit.clear()
        self._render_channels()
        self._update_apply_state()
        self._update_footer_states()

    def _on_clear_all(self) -> None:
        self._rows = []
        self._render_pairs()
        self._update_footer_states()

    def _on_set_default(self) -> None:
        rows = self.selected_rows
        if self._set_default_callback is not None:
            try:
                self._set_default_callback(rows)
            except Exception as exc:  # noqa: BLE001
                self._show_warning(
                    "Reset Reference",
                    f"Failed to save defaults:\n{exc}",
                )
                return
        self._default_rows = rows

    def _on_restore_default(self) -> None:
        self._rows = list(self._default_rows)
        self._on_clear_draft()
        self._render_pairs()

    def _on_save(self) -> None:
        if not self._rows:
            self._show_warning(
                "Reset Reference", "At least one pair is required to save."
            )
            return
        self.accept()
