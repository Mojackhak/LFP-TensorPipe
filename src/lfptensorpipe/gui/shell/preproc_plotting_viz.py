"""PSD/TFR advance and plotting helpers for preprocess visualization."""

from __future__ import annotations

from lfptensorpipe.lfp.mask.annotations import (
    annotation_sample_support_by_channel,
    valid_segments_from_annotation_support,
)
from lfptensorpipe.gui.shell.common import (
    Any,
    QDialog,
    normalize_preproc_viz_psd_params,
    normalize_preproc_viz_tfr_params,
)

_TFR_BAD_EDGE_PREFIXES = ("bad", "edge")
_TFR_BAD_EDGE_SHADOW_COLOR = "#F2000E"
_TFR_BAD_EDGE_SHADOW_ALPHA = 0.2


def _tfr_bad_edge_shadow_intervals(
    raw: Any,
    *,
    picks: list[str],
    display_start: float,
    display_stop: float,
) -> list[tuple[float, float]]:
    """Return visible BAD/EDGE intervals affecting any plotted channel."""
    selected_channels = tuple(str(name) for name in picks)
    if not selected_channels:
        return []

    support_by_channel, _, _ = annotation_sample_support_by_channel(
        raw,
        channels=selected_channels,
        keep=_TFR_BAD_EDGE_PREFIXES,
        mode="prefix",
    )
    combined_support = support_by_channel.any(axis=0)
    sfreq = float(raw.info["sfreq"])
    intervals: list[tuple[float, float]] = []
    shadow_runs = valid_segments_from_annotation_support(~combined_support)
    for start_index, stop_index in shadow_runs:
        start = max(float(display_start), start_index / sfreq)
        stop = min(float(display_stop), stop_index / sfreq)
        if stop > start:
            intervals.append((start, stop))
    return intervals


def _on_preproc_viz_psd_advance(self) -> None:
    source = self._current_preproc_viz_source()
    if source is None:
        self._show_warning(
            "Visualization PSD",
            "Select a valid visualization step first.",
        )
        return

    def _save_psd_defaults(params: dict[str, Any]) -> None:
        self._save_preproc_viz_psd_defaults(params)
        self.statusBar().showMessage("Visualization PSD defaults saved to app storage.")
        self._persist_record_params_snapshot(reason="preproc_viz_psd_default")

    dialog = self._create_qc_advance_dialog(
        mode="psd",
        session_params=self._preproc_viz_psd_params,
        default_params=self._load_preproc_viz_psd_defaults(),
        set_default_callback=_save_psd_defaults,
        parent=self,
    )
    if dialog.exec() != QDialog.Accepted or dialog.selected_params is None:
        return
    self._preproc_viz_psd_params = dict(dialog.selected_params)
    self._mark_record_param_dirty("preproc.viz")
    self._refresh_preproc_visualization_controls(self._record_context())
    self.statusBar().showMessage("Visualization PSD session parameters updated.")
    self._persist_record_params_snapshot(reason="preproc_viz_psd_save")


def _on_preproc_viz_tfr_advance(self) -> None:
    source = self._current_preproc_viz_source()
    if source is None:
        self._show_warning(
            "Visualization TFR",
            "Select a valid visualization step first.",
        )
        return

    def _save_tfr_defaults(params: dict[str, Any]) -> None:
        self._save_preproc_viz_tfr_defaults(params)
        self.statusBar().showMessage("Visualization TFR defaults saved to app storage.")
        self._persist_record_params_snapshot(reason="preproc_viz_tfr_default")

    dialog = self._create_qc_advance_dialog(
        mode="tfr",
        session_params=self._preproc_viz_tfr_params,
        default_params=self._load_preproc_viz_tfr_defaults(),
        set_default_callback=_save_tfr_defaults,
        parent=self,
    )
    if dialog.exec() != QDialog.Accepted or dialog.selected_params is None:
        return
    self._preproc_viz_tfr_params = dict(dialog.selected_params)
    self._mark_record_param_dirty("preproc.viz")
    self._refresh_preproc_visualization_controls(self._record_context())
    self.statusBar().showMessage("Visualization TFR session parameters updated.")
    self._persist_record_params_snapshot(reason="preproc_viz_tfr_save")


def _on_preproc_viz_psd_plot(self) -> None:
    _plot_spectral_qc(self, "psd")


def _on_preproc_viz_tfr_plot(self) -> None:
    _plot_spectral_qc(self, "tfr")


def _plot_spectral_qc(self, mode):
    import numpy as np
    from matplotlib.colors import LogNorm
    from lfptensorpipe.preproc.spectral_qc import compute_spectral_qc, finite_mean

    normalize = (
        normalize_preproc_viz_psd_params
        if mode == "psd"
        else normalize_preproc_viz_tfr_params
    )
    valid, params, message = normalize(getattr(self, f"_preproc_viz_{mode}_params"))
    title = f"Visualization {mode.upper()}"
    if not valid:
        self._show_warning(title, f"Invalid parameters: {message}")
        return
    source = self._current_preproc_viz_source()
    if source is None:
        self._show_warning(title, "No valid visualization source is available.")
        return
    picks = list(self._preproc_viz_selected_channels)
    if not picks:
        self._show_warning(title, "Select at least one channel.")
        return
    if not self._enable_plots:
        return
    raw = None
    try:
        _, raw_path = source
        raw = self._read_raw_fif(raw_path, preload=False, verbose="ERROR")
        result = compute_spectral_qc(raw, picks, params, mode)
        freqs, power = result["frequencies"], result["power"] * 1e12
        labels = picks
        if params["average"]:
            power = finite_mean(power, axis=0)[None]
            labels = ["Channel average"]
        method = params["method"]
        if mode == "psd":
            fig, ax = self._create_matplotlib_subplots()
            with np.errstate(divide="ignore", invalid="ignore"):
                display = 10 * np.log10(power)
            for label, values in zip(labels, display):
                ax.plot(freqs, values, label=label)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel(
                "PSD (dB re 1 µV²/Hz)"
                if result["density"]
                else "Mean wavelet power (dB re 1 µV²)"
            )
            ax.set_title(f"{mode.upper()} | {method} | {raw_path.parent.name}")
            ax.legend()
            if not np.isfinite(display).any():
                ax.text(
                    0.5,
                    0.5,
                    "No estimable finite power",
                    transform=ax.transAxes,
                    ha="center",
                )
            self._track_plot_figure(fig)
            fig.tight_layout()
            fig.show()
        else:
            times = result["times"]
            # Edges follow the actual log or linear centers, including one time cell.
            transformed = np.log(freqs) if params["spacing"] == "log" else freqs
            edges = np.r_[
                transformed[0] - (transformed[1] - transformed[0]) / 2,
                (transformed[:-1] + transformed[1:]) / 2,
                transformed[-1] + (transformed[-1] - transformed[-2]) / 2,
            ]
            if params["spacing"] == "log":
                edges = np.exp(edges)
            dt = params["decim"] / float(raw.info["sfreq"])
            time_edges = np.r_[times - dt / 2, times[-1] + dt / 2]
            time_edges[0] = max(params["tmin"] or 0, time_edges[0])
            time_edges[-1] = min(
                params["tmax"] or raw.n_times / raw.info["sfreq"],
                raw.n_times / raw.info["sfreq"],
                time_edges[-1],
            )
            for label, values in zip(labels, power):
                fig, ax = self._create_matplotlib_subplots()
                positive = values[np.isfinite(values) & (values > 0)]
                if positive.size:
                    vmin, vmax = positive.min(), positive.max()
                    image = ax.pcolormesh(
                        time_edges,
                        edges,
                        np.ma.masked_where(
                            ~np.isfinite(values) | (values <= 0), values
                        ),
                        shading="flat",
                        cmap="viridis",
                        norm=LogNorm(vmin, max(vmax, vmin * (1 + 1e-6))),
                    )
                    fig.colorbar(image, ax=ax, label="Power (µV², log scale)")
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No estimable finite power",
                        transform=ax.transAxes,
                        ha="center",
                    )
                ax.set_xlim(time_edges[0], time_edges[-1])
                ax.set_ylim(edges[0], edges[-1])
                if params["spacing"] == "log":
                    ax.set_yscale("log")
                shadow_picks = picks if params["average"] else [label]
                for left, right in _tfr_bad_edge_shadow_intervals(
                    raw,
                    picks=shadow_picks,
                    display_start=float(times[0]),
                    display_stop=float(time_edges[-1]),
                ):
                    ax.axvspan(
                        left,
                        right,
                        color=_TFR_BAD_EDGE_SHADOW_COLOR,
                        alpha=_TFR_BAD_EDGE_SHADOW_ALPHA,
                        linewidth=0,
                    )
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Frequency (Hz)")
                ax.set_title(f"TFR | {method} | {label} | {raw_path.parent.name}")
                self._track_plot_figure(fig)
                fig.tight_layout()
                fig.show()
        self.statusBar().showMessage(
            f"{title}: {len(result['dropped'])} segments had insufficient support; missing values remain NaN."
        )
    except Exception as exc:
        self._show_warning(title, f"Plot failed: {exc}")
    finally:
        if raw is not None:
            raw.close()
