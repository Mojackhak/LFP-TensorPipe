"""Dataset-dialog data containers."""

from __future__ import annotations

from .common import *  # noqa: F403

@dataclass(frozen=True)
class ResetReferenceRow:
    anode: str
    cathode: str
    name: str


@dataclass
class ParsedImportPreview:
    raw: Any
    report: dict[str, str]
    source_path: Path
    is_fif_input: bool
    import_type: str
