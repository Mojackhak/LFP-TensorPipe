"""MainWindow layout and sizing helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QGroupBox, QHBoxLayout, QPushButton, QStatusBar, QWidget


def build_ui(
    self,
    *,
    root_margin: int,
    root_spacing: int,
    stage_content_min_width: int,
) -> None:
    root = QWidget()
    root_layout = QHBoxLayout(root)
    root_layout.setContentsMargins(root_margin, root_margin, root_margin, root_margin)
    root_layout.setSpacing(root_spacing)

    self._left_column_widget = self._build_left_column()
    root_layout.addWidget(self._left_column_widget)

    self._stage_stack = self._build_stage_stack()
    self._stage_stack.setMinimumWidth(stage_content_min_width)
    root_layout.addWidget(self._stage_stack, stretch=1)

    self.setCentralWidget(root)
    self.setStatusBar(QStatusBar(self))
    self._build_menu_bar()
    self._apply_panel_title_style()
    self._enforce_button_text_fit()
    self._update_left_column_width()


def build_menu_bar(self) -> None:
    menu_bar = self.menuBar()
    settings_menu = menu_bar.addMenu("Settings")
    configs_action = QAction("Configs", self)
    configs_action.triggered.connect(self._on_settings_configs)
    settings_menu.addAction(configs_action)
    self._settings_configs_action = configs_action


def compute_left_column_width(
    window_width: int,
    *,
    left_width_ratio: float,
    left_width_min: int,
    left_width_max: int,
) -> int:
    target = int(round(window_width * left_width_ratio))
    return max(left_width_min, min(left_width_max, target))


def update_left_column_width(
    self,
    *,
    left_width_ratio: float,
    left_width_min: int,
    left_width_max: int,
) -> None:
    if self._left_column_widget is None:
        return
    width = compute_left_column_width(
        self.width(),
        left_width_ratio=left_width_ratio,
        left_width_min=left_width_min,
        left_width_max=left_width_max,
    )
    self._left_column_widget.setFixedWidth(width)


def apply_panel_title_style(
    self, *, button_cls: type[QPushButton] = QPushButton
) -> None:
    probe_button = button_cls("Apply", self)
    probe_font = probe_button.font()
    point_size = probe_font.pointSizeF()
    probe_button.deleteLater()
    if point_size <= 0:
        point_size = self.font().pointSizeF()
    if point_size <= 0:
        point_size = 10.0

    self.setStyleSheet(
        "QGroupBox::title {" f"font-size: {point_size:.1f}pt;" "font-weight: 700;" "}"
    )


def enforce_button_text_fit(self, *, button_text_horizontal_padding: int) -> None:
    for button in self.findChildren(QPushButton):
        text = button.text().replace("&", "")
        required_width = (
            button.fontMetrics().horizontalAdvance(text)
            + button_text_horizontal_padding
        )
        if not button.icon().isNull():
            required_width += button.iconSize().width() + 6
        button.setMinimumWidth(max(button.minimumWidth(), required_width))


def normalize_feature_axis_rows(
    value: Any,
    *,
    min_start: float,
    max_end: float | None = None,
    allow_duplicate_names: bool = False,
) -> list[dict[str, float | str]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, float | str]] = []
    names: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        if not allow_duplicate_names and name in names:
            continue
        try:
            start = float(item.get("start"))
            end = float(item.get("end"))
        except Exception:
            continue
        if not np.isfinite(start) or not np.isfinite(end):
            continue
        if start < float(min_start) or end <= start:
            continue
        if max_end is not None and end > float(max_end):
            continue
        if not allow_duplicate_names:
            names.add(name)
        out.append({"name": name, "start": start, "end": end})
    return sorted(out, key=lambda item: float(item["start"]))


def placeholder_block(title: str) -> QGroupBox:
    raise NotImplementedError(title)
