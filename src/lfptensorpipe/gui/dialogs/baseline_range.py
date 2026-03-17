"""Baseline-range dialog."""

from __future__ import annotations

from .common import *  # noqa: F403


class BaselineRangeConfigureDialog(QDialog):
    """Configure baseline percent ranges for plot normalization."""

    def __init__(
        self,
        *,
        current_ranges: tuple[list[float], ...],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Baseline Configure")
        self.setModal(True)
        self.resize(640, 360)
        self._rows: list[list[float]] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["#", "Start (%)", "End (%)", "Action"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionMode(QAbstractItemView.NoSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.cellClicked.connect(self._on_table_clicked)
        self._table.setToolTip("Configured baseline percent ranges.")
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._table.setMinimumHeight(200)
        root.addWidget(self._table, stretch=1)

        draft_row = QWidget()
        draft_layout = QHBoxLayout(draft_row)
        draft_layout.setContentsMargins(0, 0, 0, 0)
        draft_layout.setSpacing(6)
        self._draft_start_edit = QLineEdit()
        self._draft_start_edit.setPlaceholderText("Start")
        self._draft_start_edit.setToolTip("Baseline start in percent.")
        self._draft_end_edit = QLineEdit()
        self._draft_end_edit.setPlaceholderText("End")
        self._draft_end_edit.setToolTip("Baseline end in percent.")
        add_button = QPushButton("Add")
        add_button.clicked.connect(self._on_add)
        add_button.setToolTip("Add the draft baseline range.")
        draft_layout.addWidget(QLabel("Start"))
        draft_layout.addWidget(self._draft_start_edit)
        draft_layout.addWidget(QLabel("End"))
        draft_layout.addWidget(self._draft_end_edit)
        draft_layout.addWidget(add_button)
        root.addWidget(draft_row)

        footer = QWidget()
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(0, 0, 0, 0)
        footer_layout.setSpacing(6)
        clear_button = QPushButton("Clear All")
        cancel_button = QPushButton("Cancel")
        save_button = QPushButton("Save")
        clear_button.clicked.connect(self._on_clear)
        cancel_button.clicked.connect(self.reject)
        save_button.clicked.connect(self._on_save)
        clear_button.setToolTip("Remove all configured baseline ranges.")
        cancel_button.setToolTip("Close without changing baseline ranges.")
        save_button.setToolTip("Use the current baseline ranges.")
        footer_layout.addWidget(clear_button)
        footer_layout.addStretch(1)
        footer_layout.addWidget(cancel_button)
        footer_layout.addWidget(save_button)
        root.addWidget(footer)

        self._rows = self._normalize_ranges([list(item) for item in current_ranges])
        self._render_table()

    @property
    def selected_ranges(self) -> tuple[list[float], ...]:
        return tuple([float(item[0]), float(item[1])] for item in self._rows)

    @staticmethod
    def _normalize_ranges(value: list[list[float]]) -> list[list[float]]:
        parsed: list[list[float]] = []
        for item in value:
            if not isinstance(item, list) or len(item) != 2:
                continue
            try:
                start = float(item[0])
                end = float(item[1])
            except Exception:
                continue
            if not np.isfinite(start) or not np.isfinite(end):
                continue
            if start < 0.0 or end > 100.0 or end <= start:
                continue
            parsed.append([start, end])
        parsed.sort(key=lambda row: row[0])
        out: list[list[float]] = []
        last_end: float | None = None
        for start, end in parsed:
            if last_end is not None and start < last_end:
                continue
            out.append([float(start), float(end)])
            last_end = float(end)
        return out

    def _render_table(self) -> None:
        self._table.setRowCount(len(self._rows))
        for row_idx, (start, end) in enumerate(self._rows):
            self._table.setItem(row_idx, 0, QTableWidgetItem(str(row_idx + 1)))
            self._table.setItem(row_idx, 1, QTableWidgetItem(f"{float(start):g}"))
            self._table.setItem(row_idx, 2, QTableWidgetItem(f"{float(end):g}"))
            self._table.setItem(
                row_idx,
                3,
                make_action_table_item(
                    "Del",
                    (float(start), float(end)),
                    tool_tip="Delete this baseline range.",
                ),
            )

    def _on_add(self) -> None:
        try:
            start = float(self._draft_start_edit.text().strip())
            end = float(self._draft_end_edit.text().strip())
        except Exception:
            QMessageBox.warning(self, self.windowTitle(), "Start/End must be numeric.")
            return
        candidate = [list(item) for item in self._rows]
        candidate.append([start, end])
        normalized = self._normalize_ranges(candidate)
        if len(normalized) != len(candidate):
            QMessageBox.warning(
                self,
                self.windowTitle(),
                "Invalid or overlapping baseline ranges.",
            )
            return
        self._rows = normalized
        self._draft_start_edit.clear()
        self._draft_end_edit.clear()
        self._render_table()

    def _on_remove(self, row_idx: int) -> None:
        if row_idx < 0 or row_idx >= len(self._rows):
            return
        self._rows.pop(row_idx)
        self._render_table()

    def _on_table_clicked(self, row: int, column: int) -> None:
        if column != 3:
            return
        action_item = self._table.item(row, column)
        if action_item is None:
            return
        payload = action_item.data(ACTION_PAYLOAD_ROLE)
        if not isinstance(payload, (tuple, list)) or len(payload) != 2:
            return
        start = float(payload[0])
        end = float(payload[1])
        self._rows = [
            current
            for current in self._rows
            if not (float(current[0]) == start and float(current[1]) == end)
        ]
        self._render_table()

    def _on_clear(self) -> None:
        self._rows = []
        self._render_table()

    def _on_save(self) -> None:
        self._rows = self._normalize_ranges([list(item) for item in self._rows])
        self.accept()
