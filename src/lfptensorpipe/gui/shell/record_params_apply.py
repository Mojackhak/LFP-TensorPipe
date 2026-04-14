"""Record-parameter snapshot apply assembly mixin."""

from __future__ import annotations

from typing import Any

from lfptensorpipe.app import RecordContext

from lfptensorpipe.gui.shell.record_params_apply_alignment import (
    MainWindowRecordParamsApplyAlignmentMixin,
)
from lfptensorpipe.gui.shell.record_params_apply_features import (
    MainWindowRecordParamsApplyFeaturesMixin,
)
from lfptensorpipe.gui.shell.record_params_apply_localize import (
    MainWindowRecordParamsApplyLocalizeMixin,
)
from lfptensorpipe.gui.shell.record_params_apply_preproc import (
    MainWindowRecordParamsApplyPreprocMixin,
)
from lfptensorpipe.gui.shell.record_params_apply_tensor import (
    MainWindowRecordParamsApplyTensorMixin,
)


class MainWindowRecordParamsApplyMixin(
    MainWindowRecordParamsApplyPreprocMixin,
    MainWindowRecordParamsApplyTensorMixin,
    MainWindowRecordParamsApplyAlignmentMixin,
    MainWindowRecordParamsApplyFeaturesMixin,
    MainWindowRecordParamsApplyLocalizeMixin,
):
    """Assembly mixin for record-parameter snapshot replay."""

    def _apply_record_params_snapshot(
        self, context: RecordContext, snapshot: dict[str, Any]
    ) -> int:
        skipped = 0
        self._record_param_syncing = True
        try:
            skipped += self._apply_record_params_preproc_snapshot(snapshot)
            skipped += self._apply_record_params_tensor_snapshot(context, snapshot)
            skipped += self._apply_record_params_alignment_snapshot(snapshot)
            skipped += self._apply_record_params_features_snapshot(snapshot)
            skipped += self._apply_record_params_localize_snapshot(snapshot)
        finally:
            self._record_param_syncing = False

        self._refresh_preproc_controls()
        self._refresh_tensor_controls()
        self._refresh_alignment_controls()
        self._refresh_features_controls()
        self._refresh_localize_match_status()
        self._refresh_localize_atlas_summary()
        self._refresh_localize_action_state()
        return skipped
