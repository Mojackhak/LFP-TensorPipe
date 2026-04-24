"""Feature-axis configuration dialog."""

from __future__ import annotations

from .common import *  # noqa: F403


class FeatureAxisConfigureDialog(QDialog):
    """Configure named intervals for Features bands/times axes."""

    def __init__(
        self,
        *,
        title: str,
        item_label: str,
        current_rows: tuple[dict[str, Any], ...],
        default_rows: tuple[dict[str, Any], ...] = (),
        set_default_callback: (
            Callable[[tuple[dict[str, float | str], ...]], None] | None
        ) = None,
        min_start: float,
        max_end: float | None,
        allow_duplicate_names: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(700, 420)
        self._item_label = item_label
        self._min_start = float(min_start)
        self._max_end = float(max_end) if max_end is not None else None
        self._allow_duplicate_names = bool(allow_duplicate_names)
        self._rows: list[dict[str, float | str]] = []
        self._default_rows: list[dict[str, float | str]] = []
        self._selected_action: str | None = None
        self._set_default_callback = set_default_callback
        token = item_label.strip().lower()
        self._range_unit = "Hz" if token == "band" else "%"
        start_text = f"Start ({self._range_unit})"
        end_text = f"End ({self._range_unit})"

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["#", item_label, start_text, end_text, "Action"]
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionMode(QAbstractItemView.NoSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.cellClicked.connect(self._on_table_clicked)
        self._table.setToolTip(
            f"Configured {self._item_label.lower()} rows for this metric."
        )
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self._table.setMinimumHeight(220)
        root.addWidget(self._table, stretch=1)

        draft_row = QWidget()
        draft_layout = QHBoxLayout(draft_row)
        draft_layout.setContentsMargins(0, 0, 0, 0)
        draft_layout.setSpacing(6)
        self._draft_name_edit = QLineEdit()
        self._draft_name_edit.setPlaceholderText(item_label)
        self._draft_name_edit.setToolTip(f"{item_label} name.")
        self._draft_start_edit = QLineEdit()
        self._draft_start_edit.setPlaceholderText(start_text)
        self._draft_start_edit.setToolTip(f"Start value in {self._range_unit}.")
        self._draft_end_edit = QLineEdit()
        self._draft_end_edit.setPlaceholderText(end_text)
        self._draft_end_edit.setToolTip(f"End value in {self._range_unit}.")
        add_button = QPushButton("Add")
        add_button.clicked.connect(self._on_add)
        add_button.setToolTip(f"Add the draft {self._item_label.lower()} row.")
        draft_layout.addWidget(QLabel(item_label))
        draft_layout.addWidget(self._draft_name_edit, stretch=1)
        draft_layout.addWidget(QLabel(start_text))
        draft_layout.addWidget(self._draft_start_edit)
        draft_layout.addWidget(QLabel(end_text))
        draft_layout.addWidget(self._draft_end_edit)
        draft_layout.addWidget(add_button)
        root.addWidget(draft_row)

        footer = QWidget()
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(0, 0, 0, 0)
        footer_layout.setSpacing(6)
        clear_all_button = QPushButton("Clear All")
        set_default_button = QPushButton("Set as Default")
        restore_default_button = QPushButton("Restore Default")
        cancel_button = QPushButton("Cancel")
        save_button = QPushButton("Save")
        clear_all_button.clicked.connect(self._on_clear_all)
        set_default_button.clicked.connect(lambda: self._on_submit("set_default"))
        restore_default_button.clicked.connect(self._on_restore_default)
        cancel_button.clicked.connect(self.reject)
        save_button.clicked.connect(lambda: self._on_submit("save"))
        clear_all_button.setToolTip(
            f"Remove all configured {self._item_label.lower()} rows."
        )
        set_default_button.setToolTip(
            f"Save current {self._item_label.lower()} rows as defaults."
        )
        restore_default_button.setToolTip(
            f"Restore saved default {self._item_label.lower()} rows."
        )
        cancel_button.setToolTip(
            f"Close without changing the {self._item_label.lower()} rows."
        )
        save_button.setToolTip(
            f"Use the current {self._item_label.lower()} rows for this metric."
        )
        footer_layout.addWidget(clear_all_button)
        footer_layout.addWidget(set_default_button)
        footer_layout.addWidget(restore_default_button)
        footer_layout.addStretch(1)
        footer_layout.addWidget(cancel_button)
        footer_layout.addWidget(save_button)
        root.addWidget(footer)

        self._apply_default_rows(default_rows)
        self._apply_initial_rows(current_rows)

    @property
    def selected_rows(self) -> tuple[dict[str, float | str], ...]:
        return tuple(dict(item) for item in self._rows)

    @property
    def selected_action(self) -> str | None:
        return self._selected_action

    def _apply_default_rows(self, rows: tuple[dict[str, Any], ...]) -> None:
        parsed: list[dict[str, float | str]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name", "")).strip()
            if not name:
                continue
            try:
                start = float(row.get("start"))
                end = float(row.get("end"))
            except Exception:
                continue
            parsed.append({"name": name, "start": start, "end": end})
        valid, normalized, _ = self._validate_rows(parsed)
        self._default_rows = normalized if valid else []

    def _apply_initial_rows(self, rows: tuple[dict[str, Any], ...]) -> None:
        parsed: list[dict[str, float | str]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name", "")).strip()
            if not name:
                continue
            try:
                start = float(row.get("start"))
                end = float(row.get("end"))
            except Exception:
                continue
            parsed.append({"name": name, "start": start, "end": end})
        valid, normalized, _ = self._validate_rows(parsed)
        self._rows = normalized if valid else []
        self._render_table()

    def _render_table(self) -> None:
        self._table.setRowCount(len(self._rows))
        for row_idx, row in enumerate(self._rows):
            self._table.setItem(row_idx, 0, QTableWidgetItem(str(row_idx + 1)))
            self._table.setItem(row_idx, 1, QTableWidgetItem(str(row["name"])))
            self._table.setItem(
                row_idx, 2, QTableWidgetItem(f"{float(row['start']):g}")
            )
            self._table.setItem(row_idx, 3, QTableWidgetItem(f"{float(row['end']):g}"))
            self._table.setItem(
                row_idx,
                4,
                make_action_table_item(
                    "Del",
                    row_idx,
                    tool_tip=f"Delete this {self._item_label.lower()}.",
                ),
            )

    def _validate_rows(
        self,
        rows: list[dict[str, float | str]],
    ) -> tuple[bool, list[dict[str, float | str]], str]:
        cleaned: list[dict[str, float | str]] = []
        seen: set[str] = set()
        for row in rows:
            name = str(row.get("name", "")).strip()
            if not name:
                return False, [], f"{self._item_label} is required."
            if not self._allow_duplicate_names and name in seen:
                return False, [], f"Duplicate {self._item_label.lower()}: {name}"
            try:
                start = float(row.get("start"))
                end = float(row.get("end"))
            except Exception:
                return False, [], "Start/End must be numeric."
            if start < self._min_start:
                return (
                    False,
                    [],
                    f"Start must be >= {self._min_start:g}.",
                )
            if self._max_end is not None and end > self._max_end:
                return (
                    False,
                    [],
                    f"End must be <= {self._max_end:g}.",
                )
            if end <= start:
                return False, [], "Each row must satisfy End > Start."
            seen.add(name)
            cleaned.append({"name": name, "start": start, "end": end})
        cleaned.sort(key=lambda item: float(item["start"]))
        return True, cleaned, ""

    def _on_add(self) -> None:
        name = self._draft_name_edit.text().strip()
        start_text = self._draft_start_edit.text().strip()
        end_text = self._draft_end_edit.text().strip()
        if not name:
            QMessageBox.warning(
                self, self.windowTitle(), f"{self._item_label} is required."
            )
            return
        try:
            start = float(start_text)
            end = float(end_text)
        except Exception:
            QMessageBox.warning(self, self.windowTitle(), "Start/End must be numeric.")
            return
        candidate = [dict(item) for item in self._rows]
        candidate.append({"name": name, "start": start, "end": end})
        valid, normalized, message = self._validate_rows(candidate)
        if not valid:
            QMessageBox.warning(self, self.windowTitle(), message)
            return
        self._rows = normalized
        self._draft_name_edit.clear()
        self._draft_start_edit.clear()
        self._draft_end_edit.clear()
        self._render_table()

    def _on_remove(self, row_idx: int) -> None:
        if row_idx < 0 or row_idx >= len(self._rows):
            return
        self._rows.pop(row_idx)
        self._render_table()

    def _on_table_clicked(self, row: int, column: int) -> None:
        if column != 4:
            return
        action_item = self._table.item(row, column)
        if action_item is None:
            return
        payload = action_item.data(ACTION_PAYLOAD_ROLE)
        row_idx = int(payload) if isinstance(payload, int) else row
        self._on_remove(row_idx)

    def _on_clear_all(self) -> None:
        self._rows = []
        self._render_table()

    def _on_restore_default(self) -> None:
        self._rows = [dict(item) for item in self._default_rows]
        self._render_table()

    def _on_submit(self, action: str) -> None:
        valid, normalized, message = self._validate_rows(
            [dict(item) for item in self._rows]
        )
        if not valid:
            QMessageBox.warning(self, self.windowTitle(), message)
            return
        self._rows = normalized
        if action == "set_default":
            if self._set_default_callback is not None:
                try:
                    self._set_default_callback(tuple(dict(item) for item in self._rows))
                except Exception as exc:  # noqa: BLE001
                    QMessageBox.warning(
                        self, self.windowTitle(), f"Set as default failed:\n{exc}"
                    )
                    return
            self._default_rows = [dict(item) for item in self._rows]
            self._selected_action = action
            return
        self._selected_action = action
        self.accept()
