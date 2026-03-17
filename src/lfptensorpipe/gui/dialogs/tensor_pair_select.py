"""Tensor-pair selection dialog."""

from __future__ import annotations

from .common import *  # noqa: F403


class TensorPairSelectDialog(QDialog):
    """Tensor pair selector using Reset-Reference style draft/table flow."""

    def __init__(
        self,
        *,
        title: str,
        channel_names: tuple[str, ...],
        session_pairs: tuple[tuple[str, str], ...],
        default_pairs: tuple[tuple[str, str], ...],
        directed: bool,
        set_default_callback: (
            Callable[[tuple[tuple[str, str], ...]], None] | None
        ) = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(560, 380)
        self._channel_names = tuple(channel_names)
        self._directed = bool(directed)
        self._pairs: list[tuple[str, str]] = []
        self._default_pairs: tuple[tuple[str, str], ...] = default_pairs
        self._draft_source: str = ""
        self._set_default_callback = set_default_callback
        self._selected_action: str | None = None

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
        self._search_edit.setToolTip("Filter channels and selected pairs.")
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
            "Click once to set Source; click a second channel to set Target."
        )
        left_layout.addWidget(self._channel_list, stretch=1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)
        self._pair_table = QTableWidget(0, 5)
        self._pair_table.setHorizontalHeaderLabels(
            ["#", "Source", "Target", "Pair", "Action"]
        )
        self._pair_table.verticalHeader().setVisible(False)
        self._pair_table.setSelectionMode(QAbstractItemView.NoSelection)
        self._pair_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._pair_table.cellClicked.connect(self._on_pair_table_clicked)
        self._pair_table.setToolTip("Current channel pairs for the active metric.")
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
        splitter.setSizes([120, 420])
        root.addWidget(splitter, stretch=1)

        draft = QFrame()
        draft_layout = QGridLayout(draft)
        draft_layout.setContentsMargins(0, 0, 0, 0)
        draft_layout.setHorizontalSpacing(6)
        draft_layout.setVerticalSpacing(4)
        draft_layout.addWidget(QLabel("Source"), 0, 0)
        self._draft_source_edit = QLineEdit()
        self._draft_source_edit.setReadOnly(True)
        self._draft_source_edit.setToolTip("Draft source channel.")
        draft_layout.addWidget(self._draft_source_edit, 0, 1)
        draft_layout.addWidget(QLabel("Target"), 1, 0)
        self._draft_target_edit = QLineEdit()
        self._draft_target_edit.textChanged.connect(self._update_apply_state)
        self._draft_target_edit.setToolTip("Draft target channel.")
        draft_layout.addWidget(self._draft_target_edit, 1, 1)
        pair_row = QWidget()
        pair_row_layout = QHBoxLayout(pair_row)
        pair_row_layout.setContentsMargins(0, 0, 0, 0)
        pair_row_layout.setSpacing(6)
        self._draft_pair_preview = QLabel("")
        self._draft_pair_preview.setToolTip("Current normalized draft pair preview.")
        self._draft_all_button = QPushButton("All")
        self._draft_all_button.clicked.connect(self._on_add_all_pairs)
        self._draft_all_button.setToolTip("Add all valid pairs.")
        self._draft_apply_button = QPushButton("Apply")
        self._draft_apply_button.clicked.connect(self._on_apply_draft)
        self._draft_apply_button.setToolTip("Add the draft pair.")
        pair_row_layout.addWidget(self._draft_pair_preview, stretch=1)
        pair_row_layout.addWidget(self._draft_all_button)
        pair_row_layout.addWidget(self._draft_apply_button)
        draft_layout.addWidget(pair_row, 2, 1)
        root.addWidget(draft)

        footer = QWidget()
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(0, 0, 0, 0)
        footer_layout.setSpacing(6)
        clear_draft_button = QPushButton("Clear Draft")
        clear_all_button = QPushButton("Clear All")
        restore_button = QPushButton("Restore Defaults")
        set_default_button = QPushButton("Set as Default")
        save_button = QPushButton("Save")
        cancel_button = QPushButton("Cancel")
        clear_draft_button.clicked.connect(self._on_clear_draft)
        clear_all_button.clicked.connect(self._on_clear_all)
        restore_button.clicked.connect(self._on_restore_defaults)
        set_default_button.clicked.connect(lambda: self._accept("set_default"))
        save_button.clicked.connect(lambda: self._accept("save"))
        cancel_button.clicked.connect(self.reject)
        clear_draft_button.setToolTip("Clear the current draft pair.")
        clear_all_button.setToolTip("Remove all configured pairs.")
        restore_button.setToolTip("Restore saved default pair selection.")
        set_default_button.setToolTip(
            "Save the current pair selection as the default."
        )
        save_button.setToolTip("Use the selected pairs for the active metric.")
        cancel_button.setToolTip("Close without changing the selection.")
        footer_layout.addWidget(clear_draft_button)
        footer_layout.addWidget(clear_all_button)
        footer_layout.addWidget(set_default_button)
        footer_layout.addStretch(1)
        footer_layout.addWidget(restore_button)
        footer_layout.addWidget(save_button)
        footer_layout.addWidget(cancel_button)
        root.addWidget(footer)

        self._apply_pairs(session_pairs)
        self._render_channels()
        self._render_pairs()
        self._update_apply_state()

    @property
    def selected_pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._pairs)

    @property
    def selected_action(self) -> str | None:
        return self._selected_action

    def _show_warning(self, title: str, message: str) -> int:
        return QMessageBox.warning(self, title, message)

    def _normalize_pair(self, source: str, target: str) -> tuple[str, str]:
        return _dialog_normalize_pair(source, target, directed=self._directed)

    def _display_pair(self, pair: tuple[str, str]) -> str:
        if self._directed:
            return f"{pair[0]} -> {pair[1]}"
        return f"{pair[0]}-{pair[1]}"

    def _apply_pairs(self, pairs: tuple[tuple[str, str], ...]) -> None:
        deduped: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        allowed = set(self._channel_names)
        for source, target in pairs:
            try:
                normalized = self._normalize_pair(source, target)
            except Exception:
                continue
            if normalized[0] not in allowed or normalized[1] not in allowed:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        self._pairs = deduped
        self._render_pairs()

    def _render_pairs(self) -> None:
        token = self._search_token()
        rows: list[tuple[str, str]] = []
        for pair in self._pairs:
            if not token:
                rows.append(pair)
                continue
            if token in f"{pair[0]} {pair[1]} {self._display_pair(pair)}".lower():
                rows.append(pair)
        self._pair_table.setRowCount(len(rows))
        for row_idx, pair in enumerate(rows):
            self._pair_table.setItem(row_idx, 0, QTableWidgetItem(str(row_idx + 1)))
            self._pair_table.setItem(row_idx, 1, QTableWidgetItem(pair[0]))
            self._pair_table.setItem(row_idx, 2, QTableWidgetItem(pair[1]))
            self._pair_table.setItem(
                row_idx, 3, QTableWidgetItem(self._display_pair(pair))
            )
            self._pair_table.setItem(
                row_idx,
                4,
                make_action_table_item(
                    "Del",
                    pair,
                    tool_tip="Delete this pair.",
                ),
            )

    def _render_channels(self) -> None:
        self._channel_list.clear()
        token = self._search_token()
        for channel in self._channel_names:
            if token and token not in channel.lower():
                continue
            item = QListWidgetItem(channel)
            if channel == self._draft_source:
                item.setForeground(Qt.red)
            self._channel_list.addItem(item)

    def _search_token(self) -> str:
        return self._search_edit.text().strip().lower()

    def _on_search_changed(self, _text: str) -> None:
        self._render_channels()
        self._render_pairs()

    def _on_pair_table_clicked(self, row: int, column: int) -> None:
        if column != 4:
            return
        action_item = self._pair_table.item(row, column)
        if action_item is None:
            return
        payload = action_item.data(ACTION_PAYLOAD_ROLE)
        if not isinstance(payload, (tuple, list)) or len(payload) != 2:
            return
        self._on_remove_pair((str(payload[0]), str(payload[1])))

    def _on_channel_clicked(self, item: QListWidgetItem) -> None:
        channel = item.text().strip()
        if not self._draft_source:
            self._draft_source = channel
            self._draft_source_edit.setText(channel)
            self._render_channels()
            self._update_apply_state()
            return
        if channel == self._draft_source:
            self._on_clear_draft()
            return
        self._draft_target_edit.setText(channel)
        self._update_apply_state()

    def _update_apply_state(self) -> None:
        source = self._draft_source_edit.text().strip()
        target = self._draft_target_edit.text().strip()
        preview = ""
        if source and target and source != target:
            pair = (
                (source, target) if self._directed else tuple(sorted((source, target)))
            )
            preview = self._display_pair(pair)  # type: ignore[arg-type]
        self._draft_pair_preview.setText(preview)
        self._draft_pair_preview.setToolTip(
            "Current normalized draft pair preview. "
            f"Preview: {preview or 'None'}."
        )
        self._draft_apply_button.setEnabled(bool(source and target))

    def _on_apply_draft(self) -> None:
        source = self._draft_source_edit.text().strip()
        target = self._draft_target_edit.text().strip()
        if not source or not target:
            self._show_warning(self.windowTitle(), "Source/Target are required.")
            return
        try:
            pair = self._normalize_pair(source, target)
        except Exception as exc:  # noqa: BLE001
            self._show_warning(self.windowTitle(), str(exc))
            return
        if pair in self._pairs:
            self._show_warning(self.windowTitle(), "Pair already exists.")
            return
        self._pairs.append(pair)
        self._render_pairs()
        self._on_clear_draft()

    def _all_possible_pairs(self) -> list[tuple[str, str]]:
        return _dialog_all_possible_pairs(
            self._channel_names,
            directed=self._directed,
        )

    def _on_add_all_pairs(self) -> None:
        seen = set(self._pairs)
        for pair in self._all_possible_pairs():
            if pair in seen:
                continue
            self._pairs.append(pair)
            seen.add(pair)
        self._render_pairs()

    def _on_clear_draft(self) -> None:
        self._draft_source = ""
        self._draft_source_edit.clear()
        self._draft_target_edit.clear()
        self._draft_pair_preview.clear()
        self._render_channels()
        self._update_apply_state()

    def _on_remove_pair(self, pair: tuple[str, str]) -> None:
        self._pairs = [item for item in self._pairs if item != pair]
        self._render_pairs()

    def _on_clear_all(self) -> None:
        self._pairs = []
        self._render_pairs()

    def _on_restore_defaults(self) -> None:
        self._draft_source = ""
        self._apply_pairs(self._default_pairs)
        self._on_clear_draft()
        self._render_channels()

    def _accept(self, action: str) -> None:
        if not self._pairs:
            self._show_warning(self.windowTitle(), "At least one pair is required.")
            return
        if action == "set_default":
            pairs = tuple(self._pairs)
            if self._set_default_callback is not None:
                try:
                    self._set_default_callback(pairs)
                except Exception as exc:  # noqa: BLE001
                    self._show_warning(
                        self.windowTitle(), f"Set as default failed:\n{exc}"
                    )
                    return
            self._default_pairs = pairs
            self._selected_action = action
            return
        self._selected_action = action
        self.accept()
