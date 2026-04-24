"""Dataset-dialog data containers."""

from __future__ import annotations

from .common import *  # noqa: F403
from lfptensorpipe.io.sync import ImportSyncState, SyncEstimate, SyncFigureData


@dataclass(frozen=True)
class ResetReferenceRow:
    anode: str
    cathode: str
    name: str


@dataclass
class ParsedImportPreview:
    base_raw: Any
    report: dict[str, str]
    source_path: Path
    is_fif_input: bool
    import_type: str
    synced_raw: Any | None = None
    sync_estimate: SyncEstimate | None = None
    sync_figure_data: SyncFigureData | None = None
    sync_state: ImportSyncState | None = None

    @property
    def raw(self) -> Any:
        return self.synced_raw if self.synced_raw is not None else self.base_raw
