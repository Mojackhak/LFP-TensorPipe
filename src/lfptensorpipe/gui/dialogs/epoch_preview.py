"""Four-panel epoch comparison and a native-signal viewer."""

from copy import deepcopy

import numpy as np
from matplotlib import colormaps, colors
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure
from matplotlib.ticker import NullLocator
from matplotlib.transforms import nonsingular
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QSizePolicy,
    QStyledItemDelegate,
    QTableWidget,
    QVBoxLayout,
)

from lfptensorpipe.app.alignment.inspector_config import SIGNAL
from lfptensorpipe.app.alignment.inspector_preview import prepare_epoch_preview
from lfptensorpipe.app.alignment.inspector_qc import (
    cell_bounds,
    interpolate,
    log_frequency_cells,
    signal_blocks,
)
from lfptensorpipe.utils.pair_keys import parse_pair_token
from .epoch_review_controls import Channels, Frequency


class DeviationRows(QStyledItemDelegate):
    def __init__(self, parent):
        super().__init__(parent)
        self.colors = None

    def paint(self, painter: QPainter, option, index):
        if self.colors is None:
            return
        rgba = self.colors[index.row()]
        image = QImage(rgba.data, len(rgba), 1, rgba.nbytes, QImage.Format_RGBA8888)
        painter.drawImage(option.rect.adjusted(0, 1, 0, -1), image)


class EpochPreview(QDialog):
    def __init__(
        self,
        parent,
        data,
        view,
        prefs,
        bands,
        persist,
        open_wave,
        *,
        style_for_metric=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Epoch Inspector · Preview")
        self.resize(850, 620)
        self.data, self.view, self.prefs = data, set(view), deepcopy(prefs)
        self.bands, self.persist = bands, persist
        self.style_for_metric = style_for_metric
        self._open_wave = open_wave
        self.metric_cmap = colormaps["RdBu_r"].with_extremes(bad="#ededed")
        self.signal = None
        layout = QGridLayout(self)
        toolbar = QHBoxLayout()
        self.metric_combo = QComboBox()
        for metric in data.metrics:
            self.metric_combo.addItem(
                "Preprocessed Signal" if metric == SIGNAL else metric, metric
            )
        self.channels, self.frequency = Channels(), Frequency()
        for label, w in (
            ("Metric", self.metric_combo),
            ("Channels", self.channels),
            ("Frequency band", self.frequency),
        ):
            toolbar.addWidget(QLabel(label))
            toolbar.addWidget(w)
        toolbar.addStretch()
        layout.addLayout(toolbar, 0, 0, 1, 3)
        self.canvases = []
        for i, title in enumerate(
            ("View unchecked", "View checked", "Unchecked − checked")
        ):
            box = QGroupBox(title)
            inner = QVBoxLayout(box)
            canvas = FigureCanvasQTAgg(Figure(figsize=(3, 2), layout="constrained"))
            canvas.setMinimumSize(50, 50)
            canvas.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
            inner.addWidget(canvas)
            self.canvases.append(canvas)
            layout.addWidget(box, 1 + i // 2, i % 2)
        box = QGroupBox("Epoch deviation")
        inner = QVBoxLayout(box)
        self.rows = QTableWidget(len(data.epochs), 1)
        self.rows.setAccessibleName("Epoch deviation")
        self.rows.horizontalHeader().hide()
        self.rows.horizontalHeader().setStretchLastSection(True)
        self.rows.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.rows.verticalHeader().setMinimumSectionSize(24)
        self.rows.verticalHeader().setDefaultSectionSize(24)
        self.rows.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.rows.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.rows.setToolTip(
            "Double-click an epoch or its number to inspect the selected metric."
        )
        self.rows.cellDoubleClicked.connect(lambda row, _: self._open_epoch(row))
        self.rows.verticalHeader().sectionDoubleClicked.connect(self._open_epoch)
        self.delegate = DeviationRows(self.rows)
        self.rows.setItemDelegate(self.delegate)
        inner.addWidget(self.rows)
        self.axis_note = QLabel()
        inner.addWidget(self.axis_note)
        layout.addWidget(box, 2, 1)
        self.group_bar, self.deviation_bar = (
            FigureCanvasQTAgg(Figure(figsize=(1.1, 2), layout="constrained"))
            for _ in range(2)
        )
        for row, legend in enumerate((self.group_bar, self.deviation_bar), start=1):
            legend.setFixedWidth(110)
            layout.addWidget(legend, row, 2)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(1, 1)
        layout.setRowStretch(2, 1)
        self.metric_combo.setCurrentIndex(
            max(0, self.metric_combo.findData(prefs.get("metric", SIGNAL)))
        )
        self.metric_combo.currentIndexChanged.connect(self._select_metric)
        self.channels.changed.connect(self._draw)
        self.frequency.changed.connect(self._draw)
        self._select_metric()

    def _select_metric(self):
        key = self.metric_combo.currentData()
        self.plot_style = (
            self.style_for_metric(key)
            if self.style_for_metric and key != SIGNAL
            else {}
        )
        previous = self.prefs.get("metrics", {}).get(key, {})
        names = self.data.channels(key)
        selection = previous.get("channels", "All")
        old_index = self.prefs.get("channel_index")
        if (
            not self.prefs.get("metrics")
            and key == self.prefs.get("metric")
            and type(old_index) is int
            and 0 <= old_index < len(names)
        ):
            selection = [names[old_index]]
        self.channels.populate(names, selection)
        grid = None if key == SIGNAL else self.data.grid(key)
        self.frequency.populate(grid, self.bands, previous.get("frequency", "All"))
        self._draw()

    def _draw(self):
        try:
            self._draw_selection()
        except (ValueError, OSError) as exc:
            self.axis_note.setText(str(exc))

    def _draw_selection(self):
        key = self.metric_combo.currentData()
        selected = self.channels.values()
        frequency = self.frequency.value()
        self.prefs["metric"] = key
        self.prefs.setdefault("metrics", {})[key] = {
            "channels": "All" if self.channels.all_mode else selected,
            "frequency": frequency,
        }
        names = self.data.channels(key)
        cs = [names.index(c) for c in selected]
        self.prefs["channel_index"] = cs[0] if len(cs) == 1 else None
        self.persist(deepcopy(self.prefs))
        bounds = self.data.bounds
        self.frequency_axis = None
        if key == SIGNAL:
            if self.signal is None:
                self.signal = self.data.read_signal()
            n = max(2, max(sum(len(span.time) for span in ep) for ep in self.signal))
            times = np.linspace(*bounds, n, endpoint=self.data.axis == "percent")
            values = np.empty((len(self.signal), len(cs), 1, n))
            for e, spans in enumerate(self.signal):
                for pos, c in enumerate(cs):
                    values[e, pos, 0] = interpolate(signal_blocks(spans, c), times)
            freqs = np.array([0.0])
        else:
            grid = self.data.grid(key)
            times = grid.time
            if grid.numeric:
                mask, hz_edges = log_frequency_cells(grid, frequency)
            else:
                mask = grid.frequency_mask(frequency)
            inds = np.flatnonzero(mask)
            rows = slice(inds[0], inds[-1] + 1) if inds.size else slice(0, 1)
            if grid.numeric and inds.size > 1:
                # Limits cover the requested band within the measured frequency range.
                finite_edges = hz_edges[np.isfinite(hz_edges)]
                self.frequency_axis = (
                    hz_edges[inds[0] : inds[-1] + 2],
                    finite_edges[0],
                    finite_edges[-1],
                )
            values = grid.values[:, cs, rows, :].copy()
            values[:, :, ~mask[rows], :] = np.nan
            freqs = grid.freq[rows]
        self.epoch_planes, maps = prepare_epoch_preview(key, values, self.view)
        self.times, self.freqs = times, freqs
        group_values = np.concatenate([plane[np.isfinite(plane)] for plane in maps[:2]])
        group_norm = self._value_norm(group_values)
        difference = maps[2]
        diff_limit = np.max(
            np.abs(difference), where=np.isfinite(difference), initial=0.0
        )
        diff_norm = (
            colors.Normalize(-diff_limit, diff_limit)
            if diff_limit
            else colors.Normalize(-1, 1)
        )
        trace = values.shape[2] == 1
        self.value_unit = {
            "raw_power": "dB",
            "periodic": "dB",
            "burst": "log10",
            "aperiodic": "z",
            SIGNAL: "µV",
        }.get(key, "")
        for index, (canvas, plane) in enumerate(zip(self.canvases, maps[:3])):
            is_difference = index == 2
            norm = diff_norm if is_difference else group_norm
            self._plot_plane(canvas, plane, norm, colorbar=is_difference)
        deviation = maps[3]
        deviation_max = np.max(deviation, where=np.isfinite(deviation), initial=0.0)
        deviation_norm = colors.Normalize(0, deviation_max or 1.0)
        deviation_cmap = colormaps["viridis"].with_extremes(bad="#ededed")
        legends = (
            (self.group_bar, group_norm, self.metric_cmap, group_values.size > 0),
            (
                self.deviation_bar,
                deviation_norm,
                deviation_cmap,
                np.isfinite(deviation).any(),
            ),
        )
        for canvas, norm, palette, has_data in legends:
            canvas.figure.clear()
            visible = not trace and bool(has_data)
            canvas.setVisible(visible)
            if visible:
                mapping = ScalarMappable(norm=norm, cmap=palette)
                bar = self._add_colorbar(canvas.figure, mapping, self.value_unit)
                if canvas is self.deviation_bar and deviation_max == 0:
                    bar.set_ticks([0])
                canvas.draw_idle()
        self.delegate.colors = np.ascontiguousarray(
            deviation_cmap(deviation_norm(deviation), bytes=True)
        )
        self.rows.viewport().update()
        self.axis_note.setText(
            f"{bounds[0]:g} – {bounds[1]:g} {self.data.unit} · double-click an epoch"
        )

    @staticmethod
    def _value_norm(finite):
        limits = (
            nonsingular(finite.min(), finite.max(), expander=0.05)
            if finite.size
            else (-1.0, 1.0)
        )
        return colors.Normalize(*limits)

    def _plot_plane(self, canvas, plane, norm, *, colorbar):
        canvas.figure.clear()
        ax = canvas.figure.subplots()
        ax.set_xlabel(f"Epoch time ({self.data.unit})")
        ax.set_xlim(*self.data.bounds)
        trace = plane.shape[0] == 1
        if trace:
            margin = 0.05 * (norm.vmax - norm.vmin)
            ax.set_ylim(norm.vmin - margin, norm.vmax + margin)
            value_label = (
                self.plot_style.get("colorbar_label")
                or self.value_unit
                or self.metric_combo.currentText()
            )
            if (
                self.value_unit
                and self.value_unit.casefold() not in value_label.casefold()
            ):
                value_label = f"{value_label} ({self.value_unit})"
            ax.set_ylabel(value_label)
        elif self.frequency_axis is not None:
            y_edges, low, high = self.frequency_axis
            ticks = [low, *(f for f in (4, 8, 13, 20, 35, 60) if low < f < high), high]
            ax.set_yscale("log", base=2)
            ax.set_yticks(ticks, labels=[f"{f:g}" for f in ticks])
            ax.yaxis.set_minor_locator(NullLocator())
            ax.set_ylim(low, high)
            ax.set_ylabel("Hz")
        else:
            positions = np.arange(len(self.freqs))
            y_edges = cell_bounds(positions)
            ax.set_yticks(positions, labels=list(map(str, self.freqs)))
        if not np.isfinite(plane).any():
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
        elif trace:
            ax.plot(self.times, plane[0], linewidth=0.8)
        else:
            time_edges = cell_bounds(
                self.times,
                support=self.data.bounds,
                left_edges=self.data.axis != "percent",
            )
            mesh = ax.pcolormesh(
                time_edges,
                y_edges,
                plane,
                cmap=self.metric_cmap,
                norm=norm,
                shading="flat",
            )
            if self.frequency_axis is not None:
                for start, end in self.data.grid(
                    self.metric_combo.currentData()
                ).notches:
                    start, end = max(start, low), min(end, high)
                    if start < end:
                        ax.axhspan(
                            start, end, color=self.metric_cmap.get_bad(), linewidth=0
                        )
            if colorbar:
                self._add_colorbar(canvas.figure, mesh, self.value_unit, plot=ax)
        if self.plot_style:
            ax.tick_params(labelsize=self.plot_style["tick_label_size"])
            ax.xaxis.label.set_fontsize(self.plot_style["font_size"])
            ax.yaxis.label.set_fontsize(self.plot_style["font_size"])
        canvas.draw_idle()

    def _open_epoch(self, row):
        channels = self.channels.values()
        metric = self.metric_combo.currentData()
        if metric == SIGNAL and channels:
            try:
                self._open_wave(row)
            except (ValueError, OSError) as exc:
                self.axis_note.setText(str(exc))
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Epoch {row+1} · {self.metric_combo.currentText()}")
        dialog.resize(720, 460)
        layout = QVBoxLayout(dialog)
        channel_label = ", ".join(channels) or "No channels selected"
        scope = [f"Mean of {channel_label}" if len(channels) > 1 else channel_label]
        band = self.frequency.value()
        if isinstance(band, dict) and "hz" in band:
            lo, hi = band["hz"]
            scope.append(f"{lo:g}–{hi:g} Hz")
        elif isinstance(band, dict):
            scope.append(band["label"])
        elif metric != SIGNAL:
            scope.append(
                "All frequencies"
                if np.issubdtype(self.freqs.dtype, np.number)
                else "All bins"
            )
        label = QLabel(" · ".join(scope))
        label.setWordWrap(True)
        layout.addWidget(label)
        canvas = FigureCanvasQTAgg(Figure(layout="constrained"))
        layout.addWidget(canvas, stretch=1)
        plane = self.epoch_planes[row]
        self._plot_plane(
            canvas, plane, self._value_norm(plane[np.isfinite(plane)]), colorbar=True
        )
        dialog.exec()
        dialog.deleteLater()

    def _add_colorbar(self, figure, mapping, unit, *, plot=None):
        if plot is None:
            scale_axis = figure.subplots()
            scale_axis.set_box_aspect(12)
            bar = figure.colorbar(mapping, cax=scale_axis)
        else:
            bar = figure.colorbar(mapping, ax=plot, pad=0.035, fraction=0.065)
        bar.ax.set_title(unit, fontsize=9)
        if self.plot_style:
            if figure is self.group_bar.figure:
                bar.set_label(
                    self.plot_style["colorbar_label"],
                    fontsize=self.plot_style["font_size"],
                )
            bar.ax.tick_params(labelsize=self.plot_style["tick_label_size"])
        return bar


def show_waveform(parent, data, row, result, *, channels=None):
    spans = data.read_signal([row])[0]
    dialog = QDialog(parent)
    epoch = data.epochs[row]
    dialog.setWindowTitle(f"Preprocessed Signal · {row+1} · {epoch.label}")
    dialog.resize(780, 520)
    layout = QVBoxLayout(dialog)
    fig = Figure(layout="constrained")
    layout.addWidget(FigureCanvasQTAgg(fig))
    names = data.signal_channels if channels is None else channels
    axes = fig.subplots(len(names), 1, sharex=True, squeeze=False)[:, 0]
    hits = [e for r in (result or {}).get("rules", {}).values() for e in r["evidence"]]
    for name, ax in zip(names, axes):
        for x, _, values in signal_blocks(spans, data.signal_channels.index(name)):
            ax.plot(x, values, linewidth=0.7)
        for s in data.mappings[row]:
            ax.axvline(s.target_start, color=".45", linestyle=":", linewidth=0.8)
        for item in hits:
            if item["channel"] == name or name in (
                parse_pair_token(item["channel"]) or ()
            ):
                intervals = (
                    item.get("missing_time", [])
                    if "coverage_percent" in item
                    else [item["time"]]
                )
                for interval in intervals:
                    ax.axvspan(*interval, color="tab:red", alpha=0.15)
        ax.set_ylabel(name + " (µV)")
    for label, source_time in (getattr(epoch, "events_t", {}) or {}).items():
        if label in {"start", "end"}:
            continue
        for seg in data.mappings[row]:
            if seg.source_start <= source_time <= seg.source_end:
                position = seg.target_start + (source_time - seg.source_start) * (
                    seg.target_end - seg.target_start
                ) / (seg.source_end - seg.source_start)
                axes[0].annotate(
                    str(label),
                    (position, 1),
                    xycoords=("data", "axes fraction"),
                    xytext=(2, -2),
                    textcoords="offset points",
                    va="top",
                    fontsize=8,
                )
                break
    axes[-1].set(xlabel=f"Epoch time ({data.unit})", xlim=data.bounds)
    dialog.exec()
    dialog.deleteLater()
