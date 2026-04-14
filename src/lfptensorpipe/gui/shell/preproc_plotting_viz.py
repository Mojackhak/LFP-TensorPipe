"""PSD/TFR advance and plotting helpers for preprocess visualization."""

from __future__ import annotations

from lfptensorpipe.gui.shell.common import (
    Any,
    QDialog,
)


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
        self.statusBar().showMessage(
            "Visualization PSD defaults saved to app storage."
        )
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
        self.statusBar().showMessage(
            "Visualization TFR defaults saved to app storage."
        )
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
    self.statusBar().showMessage("Visualization TFR session parameters updated.")
    self._persist_record_params_snapshot(reason="preproc_viz_tfr_save")


def _on_preproc_viz_psd_plot(self) -> None:
    source = self._current_preproc_viz_source()
    if source is None:
        self._show_warning(
            "Visualization PSD",
            "No valid visualization source is available.",
        )
        return
    _, raw_path = source
    picks = list(self._preproc_viz_selected_channels)
    if not picks:
        self._show_warning("Visualization PSD", "Select at least one channel.")
        return
    if not self._enable_plots:
        return
    try:
        raw = self._read_raw_fif(raw_path, preload=False, verbose="ERROR")
        spectrum = raw.compute_psd(
            method="welch",
            fmin=float(self._preproc_viz_psd_params["fmin"]),
            fmax=float(self._preproc_viz_psd_params["fmax"]),
            n_fft=int(self._preproc_viz_psd_params["n_fft"]),
            picks=picks,
            verbose="ERROR",
        )
        spectrum.plot(average=bool(self._preproc_viz_psd_params["average"]))
        if hasattr(raw, "close"):
            raw.close()
    except Exception as exc:
        self._show_warning("Visualization PSD", f"PSD plot failed:\n{exc}")


def _on_preproc_viz_tfr_plot(self) -> None:
    source = self._current_preproc_viz_source()
    if source is None:
        self._show_warning(
            "Visualization TFR",
            "No valid visualization source is available.",
        )
        return
    _, raw_path = source
    picks = list(self._preproc_viz_selected_channels)
    if not picks:
        self._show_warning("Visualization TFR", "Select at least one channel.")
        return
    if not self._enable_plots:
        return
    try:
        import numpy as np
        from matplotlib.colors import LogNorm

        raw = self._read_raw_fif(raw_path, preload=True, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        stop = min(raw.n_times, int(max(1.0, 20.0) * sfreq))
        data = raw.get_data(picks=picks, start=0, stop=stop)
        if data.shape[0] == 0 or data.shape[1] == 0:
            raise ValueError("No samples available for selected channels.")
        fmin = float(self._preproc_viz_tfr_params["fmin"])
        fmax = float(self._preproc_viz_tfr_params["fmax"])
        n_freqs = int(self._preproc_viz_tfr_params["n_freqs"])
        decim = int(self._preproc_viz_tfr_params["decim"])
        freqs = np.logspace(np.log10(fmin), np.log10(fmax), n_freqs, dtype=float)
        n_cycles = np.maximum(2.0, freqs / 4.0)
        power = self._compute_tfr_array_morlet(
            data[np.newaxis, :, :],
            sfreq=sfreq,
            freqs=freqs,
            n_cycles=n_cycles,
            output="power",
            decim=decim,
        )
        mean_power = power.mean(axis=1).squeeze(0)
        time_axis = np.arange(mean_power.shape[1], dtype=float) * (decim / sfreq)
        positive_power = mean_power[np.isfinite(mean_power) & (mean_power > 0.0)]
        if positive_power.size == 0:
            raise ValueError(
                "TFR power contains no positive values for log color scale."
            )
        vmin = float(positive_power.min())
        vmax = float(positive_power.max())
        if vmax <= vmin:
            vmax = vmin * (1.0 + 1e-6)
        plot_power = np.maximum(mean_power, vmin)
        fig, ax = self._create_matplotlib_subplots()
        image = ax.imshow(
            plot_power,
            aspect="auto",
            origin="lower",
            extent=[time_axis[0], time_axis[-1], freqs[0], freqs[-1]],
            cmap="viridis",
            norm=LogNorm(vmin=vmin, vmax=vmax),
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_yscale("log")
        ax.set_title(f"TFR ({raw_path.parent.name})")
        fig.colorbar(image, ax=ax, label="Power (log scale)")
        fig.tight_layout()
        fig.show()
        if hasattr(raw, "close"):
            raw.close()
    except Exception as exc:
        self._show_warning("Visualization TFR", f"TFR plot failed:\n{exc}")
