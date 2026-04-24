"""Record-scoped path resolution for stage artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PREPROC_STEPS = {
    "raw",
    "filter",
    "annotations",
    "bad_segment_removal",
    "ecg_artifact_removal",
    "finish",
}


@dataclass(frozen=True)
class RecordContext:
    """Identity of one project/subject/record run scope."""

    project_root: Path
    subject: str
    record: str


class PathResolver:
    """Generate deterministic paths for all record-scoped artifacts."""

    def __init__(self, context: RecordContext) -> None:
        self.context = context

    @property
    def lfp_root(self) -> Path:
        return (
            self.context.project_root
            / "derivatives"
            / "lfptensorpipe"
            / self.context.subject
            / self.context.record
        )

    @property
    def preproc_root(self) -> Path:
        return self.lfp_root / "preproc"

    @property
    def import_root(self) -> Path:
        return self.lfp_root / "import"

    @property
    def tensor_root(self) -> Path:
        return self.lfp_root / "tensor"

    @property
    def alignment_root(self) -> Path:
        return self.lfp_root / "alignment"

    @property
    def features_root(self) -> Path:
        return self.lfp_root / "features"

    def record_ui_state_path(self, create: bool = False) -> Path:
        """Resolve record-scoped shared UI-state path."""
        if create:
            self.lfp_root.mkdir(parents=True, exist_ok=True)
        return self.lfp_root / "lfptensorpipe_ui_state.json"

    def record_params_log_path(self, create: bool = False) -> Path:
        """Deprecated alias for `record_ui_state_path`."""
        return self.record_ui_state_path(create=create)

    def preproc_step_dir(self, step_key: str, create: bool = False) -> Path:
        """Resolve one preprocess step directory."""
        if step_key not in PREPROC_STEPS:
            raise ValueError(f"Unknown preprocess step: {step_key}")
        out = self.preproc_root / step_key
        if create:
            out.mkdir(parents=True, exist_ok=True)
        return out

    def tensor_metric_dir(self, metric_key: str, create: bool = False) -> Path:
        """Resolve one tensor metric directory."""
        sanitized = metric_key.strip().replace(" ", "_")
        if not sanitized:
            raise ValueError("Metric key cannot be empty.")
        out = self.tensor_root / sanitized
        if create:
            out.mkdir(parents=True, exist_ok=True)
        return out

    def import_sync_dir(self, create: bool = False) -> Path:
        """Resolve record-scoped import sync artifact directory."""
        out = self.import_root / "sync"
        if create:
            out.mkdir(parents=True, exist_ok=True)
        return out

    def alignment_paradigm_dir(self, paradigm_slug: str, create: bool = False) -> Path:
        """Resolve one alignment paradigm directory."""
        sanitized = paradigm_slug.strip().replace(" ", "_")
        if not sanitized:
            raise ValueError("Trial slug cannot be empty.")
        out = self.alignment_root / sanitized
        if create:
            out.mkdir(parents=True, exist_ok=True)
        return out

    def ensure_record_roots(self, *, include_tensor: bool = False) -> None:
        """Create stage roots for the current record context."""
        self.preproc_root.mkdir(parents=True, exist_ok=True)
        if include_tensor:
            self.tensor_root.mkdir(parents=True, exist_ok=True)
        self.alignment_root.mkdir(parents=True, exist_ok=True)
        self.features_root.mkdir(parents=True, exist_ok=True)
