"""Stage routing and indicator helpers for MainWindow."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QFrame, QGroupBox, QLabel, QVBoxLayout


def _all_upstream_entry_states_green(
    window: Any,
    stage_key: str,
    *,
    stage_specs: tuple[Any, ...],
) -> bool:
    spec_by_key = {spec.key: spec for spec in stage_specs}
    current = spec_by_key.get(stage_key)
    upstream_key = current.prerequisite_key if current is not None else None
    while isinstance(upstream_key, str):
        if window._stage_states.get(upstream_key) != "green":
            return False
        upstream = spec_by_key.get(upstream_key)
        upstream_key = upstream.prerequisite_key if upstream is not None else None
    return True


def set_indicator_color(
    indicator: QLabel,
    state: str,
    *,
    indicator_colors: dict[str, str],
) -> None:
    color = indicator_colors.get(state, indicator_colors["gray"])
    indicator.setStyleSheet(
        "border-radius: 6px;" f"background-color: {color};" "border: 1px solid #8A8A8A;"
    )


def refresh_stage_controls(
    window: Any,
    *,
    stage_specs: tuple[Any, ...],
    indicator_colors: dict[str, str],
) -> None:
    state_meaning = {
        "gray": "not ready",
        "yellow": "stale or blocked",
        "green": "ready/current",
    }
    for spec in stage_specs:
        state = window._stage_states[spec.key]
        set_indicator_color(
            window._stage_indicators[spec.key],
            state,
            indicator_colors=indicator_colors,
        )
        window._stage_indicators[spec.key].setToolTip(
            "Stage state: gray=not ready, yellow=stale or blocked, "
            f"green=ready/current. Current: {state_meaning.get(state, state)}."
        )

        button = window._stage_buttons[spec.key]
        enabled = _all_upstream_entry_states_green(
            window,
            spec.key,
            stage_specs=stage_specs,
        )
        button.setEnabled(enabled)
        button.setChecked(spec.key == window._active_stage_key)
        tooltip = f"Open {spec.display_name}. Enabled only when all upstream stages are green."
        if not enabled:
            tooltip = f"{tooltip} Unavailable until previous stages are green."
        button.setToolTip(tooltip)


def route_to_stage(
    window: Any, stage_key: str, *, stage_specs: tuple[Any, ...]
) -> None:
    if stage_key not in window._stage_page_index:
        return
    button = window._stage_buttons.get(stage_key)
    if button is not None and not button.isEnabled():
        return

    window._active_stage_key = stage_key
    page_index = window._stage_page_index[stage_key]
    window._stage_stack.setCurrentIndex(page_index)
    for key, stage_button in window._stage_buttons.items():
        stage_button.setChecked(key == stage_key)
    route_key = next(spec.route_key for spec in stage_specs if spec.key == stage_key)
    window.statusBar().showMessage(f"Route: {route_key}")


def page_title(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet("font-weight: 600; font-size: 15px;")
    return label


def make_indicator_label(
    state: str,
    *,
    indicator_colors: dict[str, str],
) -> QLabel:
    label = QLabel()
    label.setFixedSize(12, 12)
    set_indicator_color(label, state, indicator_colors=indicator_colors)
    return label


def placeholder_block(title: str) -> QGroupBox:
    block = QGroupBox(title)
    block_layout = QVBoxLayout(block)
    block_layout.setContentsMargins(8, 8, 8, 8)
    block_layout.setSpacing(3)

    body = QLabel("Skeleton placeholder")
    body.setStyleSheet("color: #666666;")
    body.setFrameShape(QFrame.NoFrame)
    block_layout.addWidget(body)
    return block


__all__ = [
    "set_indicator_color",
    "refresh_stage_controls",
    "route_to_stage",
    "page_title",
    "make_indicator_label",
    "placeholder_block",
]
