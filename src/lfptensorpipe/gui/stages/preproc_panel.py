"""Preprocess page and widget builders extracted from MainWindow."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QGroupBox, QWidget

from lfptensorpipe.gui.stages import preproc_panel_annotations as _annotations
from lfptensorpipe.gui.stages import preproc_panel_builders as _builders

PAGE_MARGIN = 0
PAGE_SPACING = 0
GRID_SPACING = 0
ECG_METHODS: tuple[str, ...] = ()


def _page_margin(owner) -> int:
    getter = getattr(owner, "_stage_panel_page_margin", None)
    return PAGE_MARGIN if getter is None else int(getter())


def _page_spacing(owner) -> int:
    getter = getattr(owner, "_stage_panel_page_spacing", None)
    return PAGE_SPACING if getter is None else int(getter())


def _grid_spacing(owner) -> int:
    getter = getattr(owner, "_stage_panel_grid_spacing", None)
    return GRID_SPACING if getter is None else int(getter())


def _ecg_methods(owner) -> tuple[str, ...]:
    getter = getattr(owner, "_stage_preproc_ecg_methods", None)
    return ECG_METHODS if getter is None else tuple(getter())


def _build_preprocess_page(self) -> QWidget:
    return _builders.build_preprocess_page(
        self,
        page_margin=_page_margin(self),
        page_spacing=_page_spacing(self),
    )


def _build_preproc_raw_block(self) -> QGroupBox:
    return _builders.build_preproc_raw_block(self, grid_spacing=_grid_spacing(self))


def _build_preproc_filter_block(self) -> QGroupBox:
    return _builders.build_preproc_filter_block(self, grid_spacing=_grid_spacing(self))


def _build_preproc_annotations_block(self) -> QGroupBox:
    return _annotations.build_preproc_annotations_block(self, grid_spacing=_grid_spacing(self))


def _set_annotations_editable(self, editable: bool) -> None:
    _annotations.set_annotations_editable(self, editable)


def _annotations_table_rows(self):
    return _annotations.annotations_table_rows(self)


def _highlight_annotation_rows(self, invalid_rows: list[int]) -> None:
    _annotations.highlight_annotation_rows(self, invalid_rows)


def _append_annotation_rows(self, rows):
    _annotations.append_annotation_rows(self, rows)


def _reset_annotations_table(self) -> None:
    _annotations.reset_annotations_table(self)


def _build_preproc_finish_block(self) -> QGroupBox:
    return _builders.build_preproc_finish_block(self, grid_spacing=_grid_spacing(self))


def _build_preproc_bad_segment_block(self) -> QGroupBox:
    return _builders.build_preproc_bad_segment_block(self, grid_spacing=_grid_spacing(self))


def _build_preproc_ecg_block(self) -> QGroupBox:
    return _builders.build_preproc_ecg_block(
        self,
        grid_spacing=_grid_spacing(self),
        ecg_methods=_ecg_methods(self),
    )


def _build_preproc_visualization_block(self) -> QGroupBox:
    return _builders.build_preproc_visualization_block(self, grid_spacing=_grid_spacing(self))


def _register_preproc_indicator(
    self, step: str, indicator: QLabel | None = None
) -> QLabel:
    if indicator is None:
        indicator = self._make_indicator_label("gray")
    else:
        self._set_indicator_color(indicator, "gray")
    self._preproc_step_indicators[step] = indicator
    return indicator


def _build_preproc_status_row(self, step: str) -> QWidget:
    return _builders.build_preproc_status_row(self, step, grid_spacing=_grid_spacing(self))
