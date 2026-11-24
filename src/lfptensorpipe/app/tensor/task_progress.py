"""Run-local desktop progress events, independent of result logs and lineage."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

PROGRESS_DIR_ENV = "LFPTENSORPIPE_TENSOR_PROGRESS_DIR"
logger = logging.getLogger(__name__)


def enabled() -> bool:
    return bool(os.environ.get(PROGRESS_DIR_ENV))


def emit(event: str, metrics=(), fraction: float = 0.0) -> None:
    directory = os.environ.get(PROGRESS_DIR_ENV)
    if not directory:
        return
    line = json.dumps([event, list(metrics), fraction], separators=(",", ":")) + "\n"
    # A process owns its stream. One append per batch, including in pool callbacks.
    try:
        with (Path(directory) / f"{os.getpid()}.jsonl").open("ab", buffering=0) as f:
            f.write(line.encode("ascii"))
    except OSError as exc:
        # Display telemetry must never turn an accepted computation into a failure.
        logger.warning("Could not report Tensor progress: %s", exc)


def fit_batch_finished(spectra: int, total: int) -> None:
    emit("advance", ("periodic_aperiodic",), spectra / total)


def publish_metric(metric: str, writer, *args, **kwargs):
    emit("computed", (metric,))
    result = writer(*args, **kwargs)
    emit("saved", (metric,))
    return result


class RunProgress:
    """Consume complete event lines once; keep the trailing partial line for the next poll."""

    def __init__(self, metrics):
        self.fractions = dict.fromkeys(metrics, 0.0)
        self.computed: set[str] = set()
        self.saved: set[str] = set()
        self.offsets: dict[Path, int] = {}
        self.started = False

    def read(self, directory: Path) -> None:
        for path in directory.glob("*.jsonl"):
            with path.open("rb") as f:
                f.seek(self.offsets.get(path, 0))
                for line in f:
                    if not line.endswith(b"\n"):
                        break
                    event, metrics, fraction = json.loads(line)
                    self.started = True
                    for key in metrics:
                        if key not in self.fractions:
                            continue
                        if event == "advance":
                            self.fractions[key] = min(
                                1.0, self.fractions[key] + fraction
                            )
                        elif event in ("computed", "saved"):
                            self.computed.add(key)
                            self.fractions[key] = 1.0
                            if event == "saved":
                                self.saved.add(key)
                    self.offsets[path] = f.tell()
