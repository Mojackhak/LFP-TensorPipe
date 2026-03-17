"""Record-parameter snapshot assembly mixin."""

from __future__ import annotations

from lfptensorpipe.gui.shell.record_params_snapshot_collect import (
    MainWindowRecordParamsSnapshotCollectMixin,
)
from lfptensorpipe.gui.shell.record_params_snapshot_logs import (
    MainWindowRecordParamsSnapshotLogsMixin,
)


class MainWindowRecordParamsSnapshotMixin(
    MainWindowRecordParamsSnapshotCollectMixin,
    MainWindowRecordParamsSnapshotLogsMixin,
):
    """Assembly mixin for record-parameter collection and log-priority replay."""
