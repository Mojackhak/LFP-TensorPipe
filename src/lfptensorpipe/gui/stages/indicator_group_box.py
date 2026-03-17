"""QGroupBox helper that anchors an indicator light after the title text."""

from __future__ import annotations

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QLabel, QGroupBox, QStyle, QStyleOptionGroupBox


class IndicatorGroupBox(QGroupBox):
    """Group box with a child QLabel positioned after the painted title text."""

    def __init__(
        self,
        title: str,
        parent=None,
        *,
        indicator_spacing: int = 6,
    ) -> None:
        super().__init__(title, parent)
        self._indicator_spacing = indicator_spacing
        self._indicator = QLabel(self)
        self._indicator.setFixedSize(12, 12)
        self._indicator.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._indicator.show()
        self._indicator.raise_()
        self._sync_indicator_geometry()

    def indicator_label(self) -> QLabel:
        return self._indicator

    def setTitle(self, title: str) -> None:
        super().setTitle(title)
        self._sync_indicator_geometry()

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        self._sync_indicator_geometry()

    def showEvent(self, event) -> None:  # noqa: ANN001
        super().showEvent(event)
        self._sync_indicator_geometry()

    def changeEvent(self, event) -> None:  # noqa: ANN001
        super().changeEvent(event)
        if event.type() in {
            QEvent.FontChange,
            QEvent.LayoutDirectionChange,
            QEvent.PaletteChange,
            QEvent.StyleChange,
        }:
            self._sync_indicator_geometry()

    def _sync_indicator_geometry(self) -> None:
        title = self.title().strip()
        if not title:
            self._indicator.hide()
            return

        option = QStyleOptionGroupBox()
        self.initStyleOption(option)
        title_rect = self.style().subControlRect(
            QStyle.CC_GroupBox,
            option,
            QStyle.SC_GroupBoxLabel,
            self,
        )

        if title_rect.width() <= 0 or title_rect.height() <= 0:
            title_rect = self.rect().adjusted(16, 0, -16, 0)
            title_rect.setWidth(self.fontMetrics().horizontalAdvance(title))
            title_rect.setHeight(self.fontMetrics().height())

        indicator_x = title_rect.x() + title_rect.width() + self._indicator_spacing
        indicator_y = title_rect.y() + max(
            0, (title_rect.height() - self._indicator.height()) // 2
        )
        max_x = max(0, self.width() - self._indicator.width() - 8)
        self._indicator.move(min(indicator_x, max_x), max(0, indicator_y))
        self._indicator.show()
        self._indicator.raise_()
