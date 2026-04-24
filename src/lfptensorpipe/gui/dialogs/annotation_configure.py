"""Annotation configuration dialog."""

from __future__ import annotations

from .common import *  # noqa: F403


class AnnotationConfigureDialog(QDialog):
    """Configure preprocess annotation rows in a dedicated modal."""

    def __init__(
        self,
        *,
        session_rows: list[dict[str, Any]],
        project_root: Path | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Configure Annotations")
        self.resize(580, 380)
        self.setMinimumSize(580, 380)
        self._project_root = project_root
        self._next_row_id = 1
        self._rows: list[dict[str, Any]] = []
        self._summary_label: QLabel | None = None
        self._search_edit: QLineEdit | None = None
        self._rows_table: QTableWidget | None = None

        for row in session_rows:
            normalized = self._normalize_session_row(row)
            if normalized is None:
                continue
            self._rows.append(normalized)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        top = QWidget()
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(6)
        self._summary_label = QLabel()
        top_layout.addWidget(self._summary_label)
        top_layout.addStretch(1)
        top_layout.addWidget(QLabel("Search"))
        self._search_edit = QLineEdit()
        self._search_edit.textChanged.connect(self._render_rows)
        self._search_edit.setToolTip("Filter annotation rows by description.")
        top_layout.addWidget(self._search_edit)
        root.addWidget(top)

        self._rows_table = QTableWidget(0, 6)
        self._rows_table.setHorizontalHeaderLabels(
            ["#", "Description", "Start", "Duration", "End(optional)", "Action"]
        )
        self._rows_table.verticalHeader().setVisible(False)
        self._rows_table.setSelectionMode(QAbstractItemView.NoSelection)
        self._rows_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._rows_table.cellClicked.connect(self._on_rows_table_clicked)
        self._rows_table.setToolTip("Current annotation rows. Use Del to remove a row.")
        header = self._rows_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        header.setSectionResizeMode(3, QHeaderView.Fixed)
        header.setSectionResizeMode(4, QHeaderView.Fixed)
        header.setSectionResizeMode(5, QHeaderView.Fixed)
        self._rows_table.setColumnWidth(1, 150)
        aligned_width = (
            self._rows_table.fontMetrics().horizontalAdvance("End(optional)") + 24
        )
        for col_idx in (2, 3, 4, 5):
            self._rows_table.setColumnWidth(col_idx, aligned_width)
        root.addWidget(self._rows_table, stretch=1)

        draft = QFrame()
        draft_layout = QGridLayout(draft)
        draft_layout.setContentsMargins(0, 0, 0, 0)
        draft_layout.setHorizontalSpacing(6)
        draft_layout.setVerticalSpacing(4)
        draft_layout.addWidget(QLabel("Description"), 0, 0)
        self._draft_description_edit = QLineEdit()
        self._draft_description_edit.textChanged.connect(self._update_apply_state)
        self._draft_description_edit.setToolTip("Annotation label.")
        draft_layout.addWidget(self._draft_description_edit, 0, 1)
        draft_layout.addWidget(QLabel("Start"), 1, 0)
        self._draft_start_edit = QLineEdit()
        self._draft_start_edit.textChanged.connect(self._update_apply_state)
        self._draft_start_edit.setToolTip("Annotation onset in seconds.")
        draft_layout.addWidget(self._draft_start_edit, 1, 1)
        draft_layout.addWidget(QLabel("Duration"), 2, 0)
        self._draft_duration_edit = QLineEdit()
        self._draft_duration_edit.textChanged.connect(self._update_apply_state)
        self._draft_duration_edit.setToolTip("Annotation duration in seconds.")
        draft_layout.addWidget(self._draft_duration_edit, 2, 1)
        draft_layout.addWidget(QLabel("End(optional)"), 3, 0)
        end_row = QWidget()
        end_row_layout = QHBoxLayout(end_row)
        end_row_layout.setContentsMargins(0, 0, 0, 0)
        end_row_layout.setSpacing(6)
        self._draft_end_edit = QLineEdit()
        self._draft_end_edit.textChanged.connect(self._update_apply_state)
        self._draft_end_edit.setToolTip("Optional end time for the draft row.")
        self._draft_apply_button = QPushButton("Apply")
        self._draft_apply_button.clicked.connect(self._on_apply_draft)
        self._draft_apply_button.setToolTip("Add the draft annotation row.")
        end_row_layout.addWidget(self._draft_end_edit, stretch=1)
        end_row_layout.addWidget(self._draft_apply_button)
        draft_layout.addWidget(end_row, 3, 1)
        root.addWidget(draft)

        footer = QWidget()
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(0, 0, 0, 0)
        footer_layout.setSpacing(6)
        self._clear_draft_button = QPushButton("Clear Draft")
        self._clear_all_button = QPushButton("Clear All")
        self._import_button = QPushButton("Import Annotations")
        cancel_button = QPushButton("Cancel")
        save_button = QPushButton("Save")
        self._clear_draft_button.clicked.connect(self._on_clear_draft)
        self._clear_all_button.clicked.connect(self._on_clear_all)
        self._import_button.clicked.connect(self._on_import_annotations)
        cancel_button.clicked.connect(self.reject)
        save_button.clicked.connect(self.accept)
        self._clear_draft_button.setToolTip("Clear the current draft row.")
        self._clear_all_button.setToolTip("Remove all annotation rows.")
        self._import_button.setToolTip("Import annotation rows from a CSV file.")
        cancel_button.setToolTip("Close without changing annotations.")
        save_button.setToolTip("Use the current annotation rows in Preprocess.")
        footer_layout.addWidget(self._clear_draft_button)
        footer_layout.addWidget(self._clear_all_button)
        footer_layout.addWidget(self._import_button)
        footer_layout.addStretch(1)
        footer_layout.addWidget(cancel_button)
        footer_layout.addWidget(save_button)
        root.addWidget(footer)

        self._render_rows()
        self._update_apply_state()
        self._update_footer_states()

    @property
    def selected_rows(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "description": str(row["description"]),
                "onset": float(row["onset"]),
                "duration": float(row["duration"]),
            }
            for row in self._rows
        )

    def _show_warning(self, title: str, message: str) -> int:
        return QMessageBox.warning(self, title, message)

    def _open_file_name(
        self,
        title: str,
        start_dir: str,
        file_filter: str,
    ) -> tuple[str, str]:
        return QFileDialog.getOpenFileName(self, title, start_dir, file_filter)

    def _load_annotations_csv_rows(
        self,
        path: Path,
    ) -> tuple[bool, list[dict[str, Any]], str]:
        return load_annotations_csv_rows(path)

    def _allocate_row_id(self) -> int:
        row_id = self._next_row_id
        self._next_row_id += 1
        return row_id

    def _normalize_session_row(self, row: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(row, dict):
            return None
        description = str(row.get("description", "")).strip()
        onset_raw = row.get("onset", row.get("start", ""))
        duration_raw = row.get("duration", "")
        try:
            onset = float(onset_raw)
            duration = float(duration_raw)
        except Exception:
            return None
        if not description or onset < 0.0 or duration < 0.0:
            return None
        return {
            "row_id": self._allocate_row_id(),
            "description": description,
            "onset": onset,
            "duration": duration,
        }

    @staticmethod
    def _format_float(value: float) -> str:
        return f"{float(value):g}"

    def _search_token(self) -> str:
        if self._search_edit is None:
            return ""
        return self._search_edit.text().strip().lower()

    def _filtered_rows(self) -> list[dict[str, Any]]:
        token = self._search_token()
        if not token:
            return list(self._rows)
        return [
            row
            for row in self._rows
            if token in str(row.get("description", "")).strip().lower()
        ]

    def _render_rows(self) -> None:
        if self._rows_table is None:
            return
        filtered = self._filtered_rows()
        self._rows_table.setRowCount(len(filtered))
        for row_idx, row in enumerate(filtered):
            onset = float(row["onset"])
            duration = float(row["duration"])
            end = onset + duration

            idx_item = QTableWidgetItem(str(row_idx + 1))
            idx_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self._rows_table.setItem(row_idx, 0, idx_item)
            self._rows_table.setItem(
                row_idx, 1, QTableWidgetItem(str(row["description"]))
            )
            self._rows_table.setItem(
                row_idx, 2, QTableWidgetItem(self._format_float(onset))
            )
            self._rows_table.setItem(
                row_idx, 3, QTableWidgetItem(self._format_float(duration))
            )
            self._rows_table.setItem(
                row_idx, 4, QTableWidgetItem(self._format_float(end))
            )

            row_id = int(row["row_id"])
            self._rows_table.setItem(
                row_idx,
                5,
                make_action_table_item(
                    "Del",
                    row_id,
                    tool_tip="Delete this annotation row.",
                ),
            )

        if self._summary_label is not None:
            self._summary_label.setText(f"Rows: {len(self._rows)}")
        self._update_footer_states()

    def _on_rows_table_clicked(self, row: int, column: int) -> None:
        if column != 5:
            return
        action_item = self._rows_table.item(row, column)
        if action_item is None:
            return
        payload = action_item.data(ACTION_PAYLOAD_ROLE)
        if not isinstance(payload, int):
            return
        self._on_remove_row(payload)

    def _update_apply_state(self) -> None:
        description = self._draft_description_edit.text().strip()
        start = self._draft_start_edit.text().strip()
        has_duration = bool(self._draft_duration_edit.text().strip())
        has_end = bool(self._draft_end_edit.text().strip())
        self._draft_apply_button.setEnabled(
            bool(description and start and (has_duration or has_end))
        )

    def _update_footer_states(self) -> None:
        has_rows = bool(self._rows)
        self._clear_all_button.setEnabled(has_rows)

    def _on_apply_draft(self) -> None:
        description = self._draft_description_edit.text().strip()
        start_text = self._draft_start_edit.text().strip()
        duration_text = self._draft_duration_edit.text().strip()
        end_text = self._draft_end_edit.text().strip()

        if not description:
            self._show_warning("Configure Annotations", "Description is required.")
            return
        try:
            start = float(start_text)
        except Exception:
            self._show_warning("Configure Annotations", "Start must be a valid number.")
            return
        if start < 0.0:
            self._show_warning("Configure Annotations", "Start must be >= 0.")
            return

        duration: float
        if duration_text:
            try:
                duration = float(duration_text)
            except Exception:
                self._show_warning(
                    "Configure Annotations", "Duration must be a valid number."
                )
                return
            if duration < 0.0:
                self._show_warning("Configure Annotations", "Duration must be >= 0.")
                return
        else:
            if not end_text:
                self._show_warning(
                    "Configure Annotations", "Provide Duration or End(optional)."
                )
                return
            try:
                end_value = float(end_text)
            except Exception:
                self._show_warning(
                    "Configure Annotations", "End(optional) must be a valid number."
                )
                return
            if end_value < start:
                self._show_warning(
                    "Configure Annotations", "End(optional) must be >= Start."
                )
                return
            duration = end_value - start

        self._rows.append(
            {
                "row_id": self._allocate_row_id(),
                "description": description,
                "onset": start,
                "duration": duration,
            }
        )
        self._on_clear_draft()
        self._render_rows()

    def _on_remove_row(self, row_id: int) -> None:
        self._rows = [row for row in self._rows if int(row["row_id"]) != int(row_id)]
        self._render_rows()

    def _on_clear_draft(self) -> None:
        self._draft_description_edit.clear()
        self._draft_start_edit.clear()
        self._draft_duration_edit.clear()
        self._draft_end_edit.clear()
        self._update_apply_state()

    def _on_clear_all(self) -> None:
        self._rows = []
        self._render_rows()

    def _on_import_annotations(self) -> None:
        selected_file, _ = self._open_file_name(
            "Import Annotations CSV",
            (
                str(self._project_root)
                if self._project_root is not None
                else str(Path.home())
            ),
            "CSV Files (*.csv)",
        )
        if not selected_file:
            return
        ok, rows, message = self._load_annotations_csv_rows(Path(selected_file))
        if not ok:
            self._show_warning("Import Annotations", message)
            return
        for row in rows:
            normalized = self._normalize_session_row(row)
            if normalized is None:
                continue
            self._rows.append(normalized)
        self._render_rows()
