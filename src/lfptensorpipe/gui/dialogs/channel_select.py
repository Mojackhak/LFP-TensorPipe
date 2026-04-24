"""Channel selection dialog."""

from __future__ import annotations

from .common import *  # noqa: F403


class ChannelSelectDialog(QDialog):
    """Reusable channel-selector dialog with checkbox rows."""

    def __init__(
        self,
        *,
        title: str,
        channels: list[str],
        selected_channels: tuple[str, ...],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(420, 340)
        self._channels = channels
        self._selected = set(selected_channels)
        self._list = QListWidget()
        self._list.setToolTip("Select channels to include.")

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
        select_all_button.setToolTip("Select every channel.")
        clear_button.setToolTip("Clear all selected channels.")
        action_layout.addWidget(select_all_button)
        action_layout.addWidget(clear_button)
        action_layout.addStretch(1)
        root.addWidget(action_row)

        footer = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        footer.accepted.connect(self.accept)
        footer.rejected.connect(self.reject)
        save_button = footer.button(QDialogButtonBox.Save)
        cancel_button = footer.button(QDialogButtonBox.Cancel)
        if save_button is not None:
            save_button.setToolTip("Use the selected channels.")
        if cancel_button is not None:
            cancel_button.setToolTip("Close without changing the selection.")
        root.addWidget(footer)
        self._render()

    @property
    def selected_channels(self) -> tuple[str, ...]:
        return _dialog_checked_item_texts(self._list)

    def _render(self) -> None:
        self._list.clear()
        for channel in self._channels:
            item = QListWidgetItem(channel)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            item.setCheckState(
                Qt.Checked if channel in self._selected else Qt.Unchecked
            )
            self._list.addItem(item)

    def _on_select_all(self) -> None:
        _dialog_set_all_check_state(self._list, checked=True)

    def _on_clear(self) -> None:
        _dialog_set_all_check_state(self._list, checked=False)
