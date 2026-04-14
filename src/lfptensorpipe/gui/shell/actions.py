"""Record snapshot action helpers for MainWindow."""

from __future__ import annotations

from typing import Any


def sync_record_params_from_logs(
    window: Any,
    *,
    include_master: bool,
    clear_dirty: bool,
) -> None:
    context = window._record_context()
    if context is None:
        return
    if clear_dirty:
        window._record_param_dirty_keys.clear()
    snapshot = window._build_log_priority_snapshot(
        context,
        include_master=include_master,
    )
    skipped = window._apply_record_params_snapshot(context, snapshot)
    if skipped > 0:
        window.statusBar().showMessage("部分字段未覆盖（存在未保存改动）")


def persist_record_params_snapshot(window: Any, *, reason: str) -> bool:
    context = window._record_context()
    if context is None:
        return False
    snapshot = window._collect_record_params_snapshot()
    ok = window._write_record_params_payload(context, params=snapshot, reason=reason)
    if ok:
        window._record_param_dirty_keys.clear()
    return ok


def post_step_action_sync(window: Any, *, reason: str) -> None:
    context = window._record_context()
    if context is None:
        return
    sync_record_params_from_logs(
        window,
        include_master=False,
        clear_dirty=True,
    )
    persist_record_params_snapshot(window, reason=reason)


__all__ = [
    "sync_record_params_from_logs",
    "persist_record_params_snapshot",
    "post_step_action_sync",
]
