"""Record-parameter shell assembly mixin."""

from __future__ import annotations

from . import actions as _shell_actions
from lfptensorpipe.gui.shell.record_params_apply import MainWindowRecordParamsApplyMixin
from lfptensorpipe.gui.shell.record_params_dirty import MainWindowRecordParamsDirtyMixin
from lfptensorpipe.gui.shell.record_params_snapshot import (
    MainWindowRecordParamsSnapshotMixin,
)
from lfptensorpipe.gui.shell.record_params_store import MainWindowRecordParamsStoreMixin


class MainWindowRecordParamsMixin(
    MainWindowRecordParamsDirtyMixin,
    MainWindowRecordParamsStoreMixin,
    MainWindowRecordParamsSnapshotMixin,
    MainWindowRecordParamsApplyMixin,
):
    """Assembly mixin for record-parameter state, IO, snapshot, and replay."""

    def _sync_record_params_from_logs(
        self, *, include_master: bool, clear_dirty: bool
    ) -> None:
        _shell_actions.sync_record_params_from_logs(
            self,
            include_master=include_master,
            clear_dirty=clear_dirty,
        )

    def _persist_record_params_snapshot(self, *, reason: str) -> bool:
        return _shell_actions.persist_record_params_snapshot(self, reason=reason)

    def _persist_record_params_snapshot_on_close(self) -> None:
        try:
            self._persist_record_params_snapshot(reason="app_close")
        except Exception as exc:  # noqa: BLE001
            try:
                self.statusBar().showMessage(f"Close autosave skipped: {exc}")
            except Exception:
                pass

    def _post_step_action_sync(self, *, reason: str) -> None:
        _shell_actions.post_step_action_sync(self, reason=reason)
