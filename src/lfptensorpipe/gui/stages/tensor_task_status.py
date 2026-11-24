"""Desktop task summary, details and measured progress for Build Tensor."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import logging
import math
from pathlib import Path
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QToolButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lfptensorpipe.app.path_resolver import PathResolver
from lfptensorpipe.app.runlog_store import read_run_log
from lfptensorpipe.app.tensor.cpu_budget import normalize_tensor_cpu_percent
from lfptensorpipe.app.tensor.params import (
    TENSOR_DIRECTED_SELECTOR_KEYS,
    TENSOR_UNDIRECTED_SELECTOR_KEYS,
    TENSOR_METRICS_BY_KEY,
)
from lfptensorpipe.app.tensor.paths import (
    tensor_metric_log_path,
    tensor_metric_tensor_path,
    tensor_output_metric_keys,
)
from lfptensorpipe.app.tensor.task_summary import build_task_summary, selection_summary
from lfptensorpipe.app.preproc_service import preproc_step_raw_path


def _elapsed(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes:02d}:{seconds:02d}"


def _size(nbytes):
    if nbytes is None:
        return "—"
    mb = nbytes / 1_000_000
    return f"~{mb:,.1f} MB" if mb >= 1 else f"~{mb:.3f} MB"


def _weight_text(weight):
    if weight is None:
        return "—"
    return "<0.1%" if 0 < weight < 0.1 else f"{weight:.1f}%"


class _SummaryLabel(QLabel):
    """Keep the full task line accessible when the status bar narrows."""

    def set_line(self, text):
        self._full_text = text
        self.setText(
            self.fontMetrics().elidedText(
                text, Qt.TextElideMode.ElideRight, self.width()
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.set_line(getattr(self, "_full_text", ""))


class TaskDetails(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Build Tensor — Task Details")
        self.resize(780, 390)
        layout = QVBoxLayout(self)
        self.record = QLabel()
        layout.addWidget(self.record)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ("Metric", "Selection", "Output grid", "Planned work", "Weight", "State")
        )
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        layout.addWidget(self.table)
        self.totals = QLabel()
        self.totals.setWordWrap(True)
        layout.addWidget(self.totals)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

    def display(self, summary, states, message):
        record = (
            "—"
            if summary.duration is None
            else f"{summary.duration:g} s · {summary.sfreq:g} Hz · {summary.input_channels} input channels"
        )
        budget = (
            "—"
            if summary.slots is None
            else f"{summary.cpu_percent:g}% · {summary.slots} M6 reference slots"
        )
        self.record.setText(f"Record   {record}\nCPU setting   {budget}")
        self.table.setRowCount(len(summary.metrics))
        for i, (key, row) in enumerate(summary.metrics.items()):
            name = TENSOR_METRICS_BY_KEY[key].display_name
            if row.point and row.point.profile.mode:
                name += f" ({row.point.profile.mode})"
            if (
                key in TENSOR_DIRECTED_SELECTOR_KEYS
                or key in TENSOR_UNDIRECTED_SELECTOR_KEYS
            ):
                kind = (
                    "directed" if key in TENSOR_DIRECTED_SELECTOR_KEYS else "undirected"
                )
                selection = f"{len(row.pairs)} {kind} pairs"
            else:
                selection = f"{len(row.channels)} channels"
            cells = (
                name,
                selection,
                row.output_grid,
                row.planned_work,
                _weight_text(row.weight),
                row.error or states.get(key, "—"),
            )
            for j, value in enumerate(cells):
                item = self.table.item(i, j)
                if item is None:
                    item = QTableWidgetItem()
                    self.table.setItem(i, j, item)
                if item.text() != value:
                    item.setText(value)
                    item.setToolTip(value)
                if j == 2 and row.array_bytes is not None:
                    item.setToolTip(
                        f"{value} · Estimated numeric arrays {_size(row.array_bytes)}"
                    )
                elif j == 3 and row.point:
                    item.setToolTip(row.work_details or value)
                elif j == 4 and row.units is not None:
                    item.setToolTip(f"{row.units:,.2f} M6 reference work units")
        if summary.runtime is not None:
            seconds = (
                f"{summary.runtime:.2g}"
                if summary.runtime < 1
                else f"{summary.runtime:.0f}"
            )
            runtime = f"~{seconds} s (Estimated)"
        elif not summary.metrics:
            runtime = "Select metrics"
        elif (
            summary.error
            or summary.cpu_percent is None
            or any(m.error for m in summary.metrics.values())
        ):
            runtime = "Check settings"
        else:
            runtime = "Preparing…"
        self.totals.setText(
            f"Estimated result data    {_size(summary.array_bytes)}\nM6 reference runtime    {runtime}"
            + (f"\n{message}" if message else "")
        )


class TensorTaskStatus(QWidget):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.label = _SummaryLabel()
        self.label.setMinimumWidth(0)
        self.label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.bar = QProgressBar()
        self.bar.setMaximumWidth(180)
        self.bar.setMaximumHeight(16)
        self.bar.setTextVisible(False)
        self.count = QLabel()
        self.details = QToolButton()
        self.details.setText("Details…")
        self.details.setAutoRaise(True)
        self.details.setToolTip(
            "Show the selected work, output size and M6 reference runtime."
        )
        self.details.clicked.connect(self.show_details)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 0, 2, 0)
        layout.addWidget(self.label, 1)
        layout.addWidget(self.bar, 1)
        layout.addWidget(self.count)
        layout.addWidget(self.details)
        self.setMinimumWidth(0)
        self._dialog = None
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tensor-preview"
        )
        self._future = None
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(120)
        self._debounce.timeout.connect(self.refresh)
        self._ready = QTimer(self)
        self._ready.setInterval(100)
        self._ready.timeout.connect(self._receive)
        self._key = None
        self.summary = selection_summary({}, None)
        self.phase = ""
        self.message = ""
        self.elapsed = 0.0
        self.progress = None
        self.states = {}
        self.hide()
        self.render()

    def shutdown(self):
        self._ready.stop()
        self._debounce.stop()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def route(self, tensor):
        self.setVisible(tensor)
        if tensor:
            self.queue_preview()
        elif self._dialog:
            self._dialog.close()

    def queue_preview(self):
        if not self.isHidden() and not self.window._tensor_run_is_active():
            self._debounce.start()

    def _snapshot(self):
        win = self.window
        ctx = win._record_context()
        keys = win._selected_tensor_metrics()
        params = win._tensor_preview_params(keys)
        try:
            cpu = normalize_tensor_cpu_percent(win._tensor_cpu_percent_edit.text())
        except ValueError:
            cpu = None
        generation = None
        if ctx and keys:
            path = preproc_step_raw_path(PathResolver(ctx), "finish", create=False)
            try:
                generation = path.stat().st_mtime_ns
            except OSError:
                pass
        return (
            ctx,
            params,
            win._tensor_mask_edge_checkbox.isChecked(),
            cpu,
            generation,
        )

    def _submit(self, ctx, params, mask, cpu):
        if self._future is not None:
            self._future.cancel()
            self._future = None
        self.summary = selection_summary(params, cpu)
        if ctx and params:
            self._future = self._executor.submit(
                build_task_summary, ctx, deepcopy(params), mask, cpu
            )
            self._ready.start()
        else:
            self._ready.stop()

    def refresh(self):
        if self.isHidden() or self.window._tensor_run_is_active():
            return
        snapshot = self._snapshot()
        if self._key == snapshot:
            return
        self._key = snapshot
        self.phase = self.message = ""
        self.progress = None
        self.states = {}
        self._submit(*snapshot[:4])
        self.render()

    def _receive(self):
        if self._debounce.isActive() or self._future is None or not self._future.done():
            return
        try:
            self.summary = self._future.result()
        except Exception as exc:
            # A preview is advisory. Expose the cause in Details without changing Run.
            self.summary.error = str(exc)
            logging.getLogger(__name__).debug(
                "Tensor task preview unavailable", exc_info=True
            )
        self._future = None
        self._ready.stop()
        self.render()

    def trgc_group_count(self):
        if self._future is None and self._key == self._snapshot():
            row = self.summary.metrics.get("trgc")
            return row.frequency_groups if row else None
        return None

    def started(self, state):
        self._debounce.stop()
        snapshot = self._snapshot()
        if (
            snapshot != self._key
            or self.summary.error
            or any(row.error for row in self.summary.metrics.values())
        ):
            self._key = snapshot
            self._submit(
                state["context"],
                state["metric_params_map"],
                state["mask_edge_effects"],
                state["cpu_percent"],
            )
        self.phase = "Preparing"
        self.progress = state["task_progress"]
        self.elapsed = 0.0
        self.states = {}
        self.message = ""
        self.render()

    def poll(self, state):
        if not state.get("progress_read_error"):
            try:
                self.progress.read(Path(state["progress_files"].name))
            except (OSError, ValueError) as exc:
                state["progress_read_error"] = f"Live progress unavailable: {exc}"
                self.message = state["progress_read_error"]
        self.elapsed = time.monotonic() - state["started_at"]
        self.phase = "Running" if self.progress.started else "Preparing"
        if self.progress.started and len(self.progress.computed) == len(
            self.progress.fractions
        ):
            self.phase = "Saving"
        if state.get("stop_requested"):
            self.phase = "Stopping"
        if state.get("recovery_error"):
            self.message = state["recovery_error"]
        for key, fraction in self.progress.fractions.items():
            if key in self.progress.saved:
                self.states[key] = "Saved"
            elif key in self.progress.computed:
                self.states[key] = "Saving"
            else:
                self.states[key] = f"{fraction:.0%}"
        self.render()

    def finished(self, state, ok, cancelled, message):
        self.poll(state)
        # Reconcile against recovered, current-run receipts and every required file.
        resolver = PathResolver(state["context"])
        saved = set()
        for k in state["selected_metrics"]:
            try:
                log = read_run_log(tensor_metric_log_path(resolver, k)) or {}
            except (OSError, ValueError) as exc:
                self.states[k] = f"Could not verify saved output: {exc}"
                continue
            current = log.get("params", {}).get("run_id") == state["run_id"]
            if (
                current
                and log.get("completed")
                and all(
                    tensor_metric_tensor_path(resolver, out).is_file()
                    for out in tensor_output_metric_keys([k])
                )
            ):
                saved.add(k)
                self.progress.fractions[k] = 1.0
                self.states[k] = "Saved"
            else:
                self.states[k] = (
                    str(log.get("message", "Failed")) if current else "Not saved"
                )
        self.progress.saved = saved
        self.phase = (
            "Stopped"
            if cancelled
            else (
                "Complete"
                if ok and len(saved) == len(state["selected_metrics"])
                else "Failed"
            )
        )
        self.message = message
        self.render()

    def show_details(self):
        if self._dialog is None:
            self._dialog = TaskDetails(self)
        self._dialog.show()
        self._dialog.raise_()
        self.render()

    def render(self):
        summary = self.summary
        active = bool(self.phase)
        self.bar.setVisible(active)
        self.count.setVisible(active)
        if active:
            n = len(self.progress.fractions)
            saved = len(self.progress.saved)
            known = bool(n) and all(
                k in summary.metrics and summary.metrics[k].weight is not None
                for k in self.progress.fractions
            )
            percent = None
            if self.phase == "Complete":
                percent = 100
            elif known and self.phase != "Preparing":
                percent = min(
                    99,
                    math.floor(
                        sum(
                            summary.metrics[k].weight * f
                            for k, f in self.progress.fractions.items()
                        )
                        + 1e-9
                    ),
                )
            self.bar.setRange(
                0,
                (
                    0
                    if percent is None and self.phase not in ("Failed", "Stopped")
                    else 100
                ),
            )
            self.bar.setValue(percent or 0)
            line = f"Build Tensor · {self.phase}"
            self.count.setText(
                (f"{percent}% · " if percent is not None else "")
                + f"{saved}/{n} saved · Elapsed {_elapsed(self.elapsed)}"
            )
        else:
            channels, undirected, directed = summary.counts()
            parts = [f"{len(summary.metrics)} metrics"]
            if undirected:
                parts.append(f"{undirected} undirected pairs")
            if directed:
                parts.append(f"{directed} directed pairs")
            if summary.metrics:
                parts.extend(
                    (f"{channels} channels", f"Est. data {_size(summary.array_bytes)}")
                )
            line = " · ".join(parts)
        self.label.set_line(line)
        self.label.setToolTip(self.message or summary.error or line)
        if self._dialog and self._dialog.isVisible():
            self._dialog.display(summary, self.states, self.message or summary.error)
