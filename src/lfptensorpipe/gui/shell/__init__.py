"""Main-window shell assembly helpers.

`gui.shell` owns MainWindow orchestration and state reconciliation. Widget
builders belong in `gui.stages`, and dialog classes belong in `gui.dialogs`.
"""

from __future__ import annotations

from .alignment_logic import MainWindowAlignmentMixin
from .actions import (
    persist_record_params_snapshot,
    post_step_action_sync,
    sync_record_params_from_logs,
)
from .busy_state import (
    on_busy_tick,
    render_busy_message,
    run_with_busy,
    set_busy_ui_lock,
    start_busy,
    stop_busy,
)
from .routing import (
    make_indicator_label,
    page_title,
    placeholder_block,
    refresh_stage_controls,
    route_to_stage,
    set_indicator_color,
)
from .dataset_localize_logic import MainWindowDatasetLocalizeMixin
from .features_logic import MainWindowFeaturesMixin
from .preproc_logic import MainWindowPreprocMixin
from .stage_state import MainWindowStageStateMixin
from .tensor_logic import MainWindowTensorMixin

__all__ = [
    "MainWindowAlignmentMixin",
    "MainWindowDatasetLocalizeMixin",
    "MainWindowFeaturesMixin",
    "MainWindowPreprocMixin",
    "MainWindowStageStateMixin",
    "MainWindowTensorMixin",
    "persist_record_params_snapshot",
    "post_step_action_sync",
    "sync_record_params_from_logs",
    "on_busy_tick",
    "render_busy_message",
    "run_with_busy",
    "set_busy_ui_lock",
    "start_busy",
    "stop_busy",
    "make_indicator_label",
    "page_title",
    "placeholder_block",
    "refresh_stage_controls",
    "route_to_stage",
    "set_indicator_color",
]
