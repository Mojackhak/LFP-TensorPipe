"""Recording identities captured while parsing a vendor export."""

from dataclasses import dataclass, field


@dataclass
class SourceMetadata:
    """Original source labels keyed by parsed channel name, and sidecars used."""

    labels: dict[str, str] = field(default_factory=dict)
    metadata_file: str | None = None
    marker_file: str | None = None
    sampling_rate_hz: float | None = None
