"""Tensor-channel selection dialog."""

from __future__ import annotations

from .common import *  # noqa: F403


class TensorChannelSelectDialog(QDialog):
    """Tensor-specific channel selector with default actions."""

    def __init__(
        self,
        *,
        title: str,
        channels: tuple[str, ...],
        session_selected: tuple[str, ...],
        default_selected: tuple[str, ...],
        set_default_callback: Callable[[tuple[str, ...]], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(420, 360)
        self._channels = tuple(channels)
        self._session_selected = set(session_selected)
        self._default_selected = set(default_selected)
        self._set_default_callback = set_default_callback
        self._selected_action: str | None = None
        self._list = QListWidget()
        self._list.setToolTip("Select channels to include in the active metric.")

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        root.addWidget(self._list, stretch=1)

        action_row = QWidget()
        action_layout = QHBoxLayout(action_row)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(6)
        select_all_button = QPushButton("Select All")
        clear_button = QPushButton("Clear")
        select_all_button.clicked.connect(self._on_select_all)
        clear_button.clicked.connect(self._on_clear)
        select_all_button.setToolTip("Select every available channel.")
        clear_button.setToolTip("Clear all selected channels.")
        action_layout.addWidget(select_all_button)
        action_layout.addWidget(clear_button)
        action_layout.addStretch(1)
        root.addWidget(action_row)

        footer = QWidget()
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(0, 0, 0, 0)
        footer_layout.setSpacing(6)
        restore_button = QPushButton("Restore Defaults")
        set_default_button = QPushButton("Set as Default")
        save_button = QPushButton("Save")
        cancel_button = QPushButton("Cancel")
        restore_button.clicked.connect(self._on_restore_defaults)
        set_default_button.clicked.connect(lambda: self._accept("set_default"))
        save_button.clicked.connect(lambda: self._accept("save"))
        cancel_button.clicked.connect(self.reject)
        restore_button.setToolTip("Restore saved default channel selection.")
        set_default_button.setToolTip(
            "Save the current channel selection as the default."
        )
        save_button.setToolTip("Use the selected channels for the active metric.")
        cancel_button.setToolTip("Close without changing the selection.")
        footer_layout.addWidget(set_default_button)
        footer_layout.addStretch(1)
        footer_layout.addWidget(restore_button)
        footer_layout.addWidget(save_button)
        footer_layout.addWidget(cancel_button)
        root.addWidget(footer)
        self._render()

    @property
    def selected_channels(self) -> tuple[str, ...]:
        return _dialog_checked_item_texts(self._list)

    @property
    def selected_action(self) -> str | None:
        return self._selected_action

    def _render(self) -> None:
        self._list.clear()
        for channel in self._channels:
            item = QListWidgetItem(channel)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            item.setCheckState(
                Qt.Checked if channel in self._session_selected else Qt.Unchecked
            )
            self._list.addItem(item)

    def _apply_selected(self, selected: set[str]) -> None:
        for row_idx in range(self._list.count()):
            item = self._list.item(row_idx)
            if item is not None:
                item.setCheckState(
                    Qt.Checked if item.text() in selected else Qt.Unchecked
                )

    def _on_select_all(self) -> None:
        _dialog_set_all_check_state(self._list, checked=True)

    def _on_clear(self) -> None:
        _dialog_set_all_check_state(self._list, checked=False)

    def _on_restore_defaults(self) -> None:
        self._apply_selected(self._default_selected)

    def _accept(self, action: str) -> None:
        if action == "set_default":
            selected = tuple(self.selected_channels)
            if self._set_default_callback is not None:
                try:
                    self._set_default_callback(selected)
                except Exception as exc:  # noqa: BLE001
                    QMessageBox.warning(
                        self, self.windowTitle(), f"Set as default failed:\n{exc}"
                    )
                    return
            self._default_selected = set(selected)
            self._selected_action = action
            return
        self._selected_action = action
        self.accept()
