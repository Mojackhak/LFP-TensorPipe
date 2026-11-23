"""Shared channel and frequency selectors for Epoch Inspector dialogs."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)


class DecimalBox(QDoubleSpinBox):
    def textFromValue(self, value):
        text = super().textFromValue(value)
        separator = self.locale().decimalPoint()
        return text.rstrip("0").rstrip(separator) if separator in text else text


def number(value=0, minimum=-1e9, maximum=1e9):
    box = DecimalBox()
    box.setDecimals(6)
    box.setRange(minimum, maximum)
    box.setValue(value)
    box.setKeyboardTracking(False)
    return box


class Channels(QToolButton):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        menu = QMenu(self)
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.toggle = QPushButton("Select all")
        self.list = QListWidget()
        self.list.setMinimumWidth(190)
        layout.addWidget(self.toggle)
        layout.addWidget(self.list)
        action = QWidgetAction(menu)
        action.setDefaultWidget(panel)
        menu.addAction(action)
        self.setMenu(menu)
        self.setPopupMode(QToolButton.InstantPopup)
        self.toggle.clicked.connect(self.toggle_all)
        self.list.itemChanged.connect(self._changed)
        self.setText("Channels")
        self.setMinimumWidth(75)

    def populate(self, names, selected):
        self.all_mode = selected == "All"
        self.list.blockSignals(True)
        self.list.clear()
        for name in names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(
                Qt.Checked if selected == "All" or name in selected else Qt.Unchecked
            )
            self.list.addItem(item)
        self.list.blockSignals(False)
        self._label()

    def values(self):
        return [
            self.list.item(i).text()
            for i in range(self.list.count())
            if self.list.item(i).checkState() == Qt.Checked
        ]

    def _label(self):
        names = self.values()
        all_selected = len(names) == self.list.count() and bool(names)
        self.setText(
            "All"
            if all_selected
            else (
                ", ".join(names[:2]) + ("…" if len(names) > 2 else "")
                if names
                else "None"
            )
        )
        self.setToolTip(", ".join(names))
        self.toggle.setText("Clear all" if all_selected else "Select all")

    def _changed(self):
        self.all_mode = (
            bool(self.list.count()) and len(self.values()) == self.list.count()
        )
        self._label()
        self.changed.emit()

    def toggle_all(self):
        new_state = (
            Qt.Unchecked if len(self.values()) == self.list.count() else Qt.Checked
        )
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(new_state)
        self.list.blockSignals(False)
        self._changed()


class Frequency(QWidget):
    changed = Signal()

    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.combo = QComboBox()
        self.low, self.high = number(), number()
        for control in (self.combo, self.low, self.high):
            layout.addWidget(control)
        self.combo.currentIndexChanged.connect(self._selection)
        self.low.valueChanged.connect(self.changed)
        self.high.valueChanged.connect(self.changed)

    def populate(self, grid, bands, value="All"):
        self.blockSignals(True)
        self.combo.clear()
        self.combo.addItem("All", "All")
        if grid is not None and grid.numeric:
            for band in bands:
                item = {"label": band["name"], "hz": [band["start"], band["end"]]}
                if grid.frequency_mask(item).any():
                    self.combo.addItem(band["name"], item)
            self.combo.addItem("Custom (Hz)", "custom")
            self.low.setValue(float(grid.freq[0]))
            self.high.setValue(float(grid.freq[-1]))
        elif grid is not None:
            for label in grid.freq:
                item = {"label": str(label)}
                if str(label) in grid.bands:
                    item["segments_hz"] = grid.bands[str(label)]
                self.combo.addItem(str(label), item)
        self.setEnabled(grid is not None)
        index = next(
            (i for i in range(self.combo.count()) if self.combo.itemData(i) == value),
            -1,
        )
        if (
            index < 0
            and isinstance(value, dict)
            and "hz" in value
            and "label" not in value
        ):
            index = self.combo.findData("custom")
            self.low.setValue(value["hz"][0])
            self.high.setValue(value["hz"][1])
        if index < 0:
            self.combo.addItem(
                str(value.get("label", value) if isinstance(value, dict) else value),
                value,
            )
            index = self.combo.count() - 1
        self.combo.setCurrentIndex(index)
        self._selection()
        self.blockSignals(False)

    def _selection(self):
        custom = self.combo.currentData() == "custom"
        self.low.setVisible(custom)
        self.high.setVisible(custom)
        self.changed.emit()

    def value(self):
        value = self.combo.currentData()
        return (
            {"hz": [self.low.value(), self.high.value()]}
            if value == "custom"
            else value
        )
