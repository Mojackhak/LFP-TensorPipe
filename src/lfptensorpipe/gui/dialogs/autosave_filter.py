"""Plot autosave close filter."""

from __future__ import annotations

from .common import *  # noqa: F403
class _CloseAutosaveFilter(QObject):
    """Event filter that triggers one autosave callback on Qt close."""

    def __init__(
        self, on_close: Callable[[Any], None], parent: QObject | None = None
    ) -> None:
        super().__init__(parent)
        self._on_close = on_close

    def eventFilter(self, watched: QObject, event: QEvent | None) -> bool:
        _ = watched
        if event is not None and event.type() == QEvent.Close:
            self._on_close(event)
        return False


