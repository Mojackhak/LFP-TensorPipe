"""Tensor-band configuration dialog."""

from __future__ import annotations

from .common import *  # noqa: F403


class TensorBandsConfigureDialog(QDialog):
    """Bands configure dialog for PSI/Burst metrics."""

    def __init__(
        self,
        *,
        title: str,
        current_bands: tuple[dict[str, Any], ...],
        default_bands: tuple[dict[str, Any], ...] = (),
        set_default_callback: (
            Callable[[tuple[dict[str, float | str], ...]], None] | None
        ) = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(520, 360)
        self._bands: list[dict[str, float | str]] = []
        self._default_bands: list[dict[str, float | str]] = []
        self._selected_action: str | None = None
        self._set_default_callback = set_default_callback

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["#", "Band", "Start", "End", "Action"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionMode(QAbstractItemView.NoSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.cellClicked.connect(self._on_table_clicked)
        self._table.setToolTip("Configured bands for this metric.")
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        root.addWidget(self._table, stretch=1)

        draft = QFrame()
        draft_layout = QGridLayout(draft)
        draft_layout.setContentsMargins(0, 0, 0, 0)
        draft_layout.setHorizontalSpacing(6)
        draft_layout.setVerticalSpacing(4)
        draft_layout.addWidget(QLabel("Band"), 0, 0)
        self._draft_name_edit = QLineEdit()
        self._draft_name_edit.setToolTip("Band name.")
        draft_layout.addWidget(self._draft_name_edit, 0, 1)
        draft_layout.addWidget(QLabel("Start"), 1, 0)
        self._draft_start_edit = QLineEdit()
        self._draft_start_edit.setToolTip("Band start frequency in Hz.")
        draft_layout.addWidget(self._draft_start_edit, 1, 1)
        draft_layout.addWidget(QLabel("End"), 2, 0)
        end_row = QWidget()
        end_row_layout = QHBoxLayout(end_row)
        end_row_layout.setContentsMargins(0, 0, 0, 0)
        end_row_layout.setSpacing(6)
        self._draft_end_edit = QLineEdit()
        self._draft_end_edit.setToolTip("Band end frequency in Hz.")
        self._apply_button = QPushButton("Apply")
        self._apply_button.clicked.connect(self._on_apply_draft)
        self._apply_button.setToolTip("Add the draft band.")
        end_row_layout.addWidget(self._draft_end_edit, stretch=1)
        end_row_layout.addWidget(self._apply_button)
        draft_layout.addWidget(end_row, 2, 1)
        root.addWidget(draft)

        footer = QWidget()
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(0, 0, 0, 0)
        footer_layout.setSpacing(6)
        clear_draft_button = QPushButton("Clear Draft")
        clear_all_button = QPushButton("Clear All")
        set_default_button = QPushButton("Set as Default")
        restore_button = QPushButton("Restore Defaults")
        save_button = QPushButton("Save")
        cancel_button = QPushButton("Cancel")
        clear_draft_button.clicked.connect(self._on_clear_draft)
        clear_all_button.clicked.connect(self._on_clear_all)
        set_default_button.clicked.connect(lambda: self._on_submit("set_default"))
        restore_button.clicked.connect(self._on_restore_defaults)
        save_button.clicked.connect(lambda: self._on_submit("save"))
        cancel_button.clicked.connect(self.reject)
        clear_draft_button.setToolTip("Clear the current draft band.")
        clear_all_button.setToolTip("Remove all configured bands.")
        set_default_button.setToolTip("Save current bands as the default.")
        restore_button.setToolTip("Restore saved default bands.")
        save_button.setToolTip("Use the current bands for the active metric.")
        cancel_button.setToolTip("Close without changing the band list.")
        footer_layout.addWidget(clear_draft_button)
        footer_layout.addWidget(clear_all_button)
        footer_layout.addWidget(set_default_button)
        footer_layout.addStretch(1)
        footer_layout.addWidget(restore_button)
        footer_layout.addWidget(save_button)
        footer_layout.addWidget(cancel_button)
        root.addWidget(footer)

        self._apply_default_bands(default_bands)
        self._apply_initial_bands(current_bands)

    @property
    def selected_bands(self) -> tuple[dict[str, float | str], ...]:
        return tuple(dict(item) for item in self._bands)

    @property
    def selected_action(self) -> str | None:
        return self._selected_action

    def _apply_default_bands(self, rows: tuple[dict[str, Any], ...]) -> None:
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
        valid, message = self._validate_rows(parsed)
        if valid:
            self._default_bands = sorted(parsed, key=lambda item: float(item["start"]))
            return
        _ = message
        self._default_bands = []

    def _apply_initial_bands(self, rows: tuple[dict[str, Any], ...]) -> None:
        self._bands = []
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
            if start <= 0.0 or end <= start:
                continue
            self._bands.append({"name": name, "start": float(start), "end": float(end)})
        self._bands.sort(key=lambda item: float(item["start"]))
        self._render_table()

    def _render_table(self) -> None:
        self._table.setRowCount(len(self._bands))
        for row_idx, row in enumerate(self._bands):
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
                    str(row["name"]),
                    tool_tip="Delete this band.",
                ),
            )

    def _validate_rows(self, rows: list[dict[str, float | str]]) -> tuple[bool, str]:
        names: set[str] = set()
        ordered = sorted(rows, key=lambda item: float(item["start"]))
        for row in ordered:
            name = str(row["name"]).strip()
            start = float(row["start"])
            end = float(row["end"])
            if not name:
                return False, "Band name cannot be empty."
            if name in names:
                return False, f"Duplicate band name: {name}"
            if start <= 0.0 or end <= start:
                return False, "Each band must satisfy 0 < start < end."
            names.add(name)
        return True, ""

    def _on_apply_draft(self) -> None:
        name = self._draft_name_edit.text().strip()
        start_text = self._draft_start_edit.text().strip()
        end_text = self._draft_end_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Bands Configure", "Band is required.")
            return
        try:
            start = float(start_text)
            end = float(end_text)
        except Exception:
            QMessageBox.warning(self, "Bands Configure", "Start/End must be numeric.")
            return
        candidate = [dict(item) for item in self._bands]
        candidate.append({"name": name, "start": float(start), "end": float(end)})
        valid, message = self._validate_rows(candidate)
        if not valid:
            QMessageBox.warning(self, "Bands Configure", message)
            return
        self._bands = sorted(candidate, key=lambda item: float(item["start"]))
        self._on_clear_draft()
        self._render_table()

    def _on_remove_row(self, row_idx: int) -> None:
        if row_idx < 0 or row_idx >= len(self._bands):
            return
        self._bands.pop(row_idx)
        self._render_table()

    def _on_table_clicked(self, row: int, column: int) -> None:
        if column != 4:
            return
        action_item = self._table.item(row, column)
        if action_item is None:
            return
        payload = action_item.data(ACTION_PAYLOAD_ROLE)
        if not isinstance(payload, str):
            return
        self._bands = [
            current for current in self._bands if str(current["name"]) != payload
        ]
        self._render_table()

    def _on_clear_draft(self) -> None:
        self._draft_name_edit.clear()
        self._draft_start_edit.clear()
        self._draft_end_edit.clear()

    def _on_clear_all(self) -> None:
        self._bands = []
        self._render_table()

    def _on_restore_defaults(self) -> None:
        self._bands = [dict(item) for item in self._default_bands]
        self._render_table()

    def _on_submit(self, action: str) -> None:
        if not self._bands:
            QMessageBox.warning(
                self, "Bands Configure", "At least one band is required."
            )
            return
        valid, message = self._validate_rows([dict(item) for item in self._bands])
        if not valid:
            QMessageBox.warning(self, "Bands Configure", message)
            return
        if action == "set_default":
            if self._set_default_callback is not None:
                try:
                    self._set_default_callback(
                        tuple(dict(item) for item in self._bands)
                    )
                except Exception as exc:  # noqa: BLE001
                    QMessageBox.warning(
                        self, self.windowTitle(), f"Set as default failed:\n{exc}"
                    )
                    return
            self._default_bands = [dict(item) for item in self._bands]
            self._selected_action = action
            return
        self._selected_action = action
        self.accept()
