"""Preprocess channel and visualization-source MainWindow methods."""

from __future__ import annotations

from lfptensorpipe.gui.shell.common import (
    PREPROC_VIZ_STEP_LABELS,
    PREPROC_VIZ_STEP_ORDER,
    Path,
    PathResolver,
    QDialog,
    RecordContext,
    preproc_step_raw_path,
)


class MainWindowPreprocChannelsMixin:
    @staticmethod
    def _preproc_step_path_for_viz(resolver: PathResolver, step: str) -> Path:
        if step == "finish":
            return resolver.preproc_root / "finish" / "raw.fif"
        return preproc_step_raw_path(resolver, step)

    @staticmethod
    def _read_channel_names_from_raw(raw_path: Path) -> list[str]:
        try:
            import mne

            raw = mne.io.read_raw_fif(str(raw_path), preload=False, verbose="ERROR")
            channel_names = list(raw.ch_names)
            if hasattr(raw, "close"):
                raw.close()
            return channel_names
        except Exception:
            return []

    @staticmethod
    def _intersect_channels(
        available: tuple[str, ...],
        selected: tuple[str, ...],
        *,
        preserve_empty: bool = False,
    ) -> tuple[str, ...]:
        selected_set = set(selected)
        kept = tuple(channel for channel in available if channel in selected_set)
        if kept or preserve_empty:
            return kept
        return available

    @staticmethod
    def _format_channel_button_text(
        label: str,
        selected: tuple[str, ...],
        available: tuple[str, ...],
    ) -> str:
        return f"{label} ({len(selected)}/{len(available)})"

    def _run_channel_selector(
        self,
        *,
        title: str,
        available: tuple[str, ...],
        selected: tuple[str, ...],
        allow_empty: bool = False,
    ) -> tuple[str, ...] | None:
        dialog = self._create_channel_select_dialog(
            title=title,
            channels=list(available),
            selected_channels=selected,
            parent=self,
        )
        if dialog.exec() != QDialog.Accepted:
            return None
        chosen = dialog.selected_channels
        if not chosen and not allow_empty:
            self._show_warning(title, "At least one channel must be selected.")
            return None
        return chosen

    def _available_preproc_viz_steps(self, context: RecordContext) -> list[str]:
        resolver = PathResolver(context)
        available: list[str] = []
        for step in PREPROC_VIZ_STEP_ORDER:
            if self._preproc_step_path_for_viz(resolver, step).exists():
                available.append(step)
        return available

    def _current_preproc_viz_step(self) -> str | None:
        if self._preproc_viz_step_combo is None:
            return None
        step = self._preproc_viz_step_combo.currentData()
        if not isinstance(step, str):
            return None
        return step

    def _refresh_preproc_ecg_channel_state(self, context: RecordContext | None) -> None:
        if context is None:
            self._preproc_ecg_available_channels = ()
            self._preproc_ecg_selected_channels = ()
            if self._preproc_ecg_channels_button is not None:
                self._preproc_ecg_channels_button.setText("Select Channels (0/0)")
                self._preproc_ecg_channels_button.setEnabled(False)
            return
        resolver = PathResolver(context)
        raw_path = preproc_step_raw_path(resolver, "bad_segment_removal")
        if not raw_path.exists():
            self._preproc_ecg_available_channels = ()
            self._preproc_ecg_selected_channels = ()
            if self._preproc_ecg_channels_button is not None:
                self._preproc_ecg_channels_button.setText("Select Channels (0/0)")
                self._preproc_ecg_channels_button.setEnabled(False)
            return
        channels = tuple(self._read_channel_names_from_raw(raw_path))
        self._preproc_ecg_available_channels = channels
        self._preproc_ecg_selected_channels = self._intersect_channels(
            channels,
            self._preproc_ecg_selected_channels,
            preserve_empty=True,
        )
        if self._preproc_ecg_channels_button is not None:
            self._preproc_ecg_channels_button.setEnabled(bool(channels))
            self._preproc_ecg_channels_button.setText(
                self._format_channel_button_text(
                    "Select Channels",
                    self._preproc_ecg_selected_channels,
                    channels,
                )
            )

    def _on_preproc_ecg_channels_select(self) -> None:
        if not self._preproc_ecg_available_channels:
            self._show_warning(
                "ECG Channels",
                "No channels available for selection.",
            )
            return
        chosen = self._run_channel_selector(
            title="ECG Channels",
            available=self._preproc_ecg_available_channels,
            selected=self._preproc_ecg_selected_channels,
            allow_empty=True,
        )
        if chosen is None:
            return
        self._preproc_ecg_selected_channels = chosen
        self._mark_record_param_dirty("preproc.ecg")
        if self._preproc_ecg_channels_button is not None:
            self._preproc_ecg_channels_button.setText(
                self._format_channel_button_text(
                    "Select Channels",
                    self._preproc_ecg_selected_channels,
                    self._preproc_ecg_available_channels,
                )
            )
        self._refresh_preproc_controls()

    def _refresh_preproc_visualization_controls(
        self, context: RecordContext | None
    ) -> None:
        combo = self._preproc_viz_step_combo
        if combo is None:
            return
        if context is None:
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("No step output", None)
            combo.setCurrentIndex(0)
            combo.blockSignals(False)
            self._preproc_viz_available_channels = ()
            self._preproc_viz_selected_channels = ()
            if self._preproc_viz_channels_button is not None:
                self._preproc_viz_channels_button.setText("Select Channels (0/0)")
                self._preproc_viz_channels_button.setEnabled(False)
            if self._preproc_viz_psd_advance_button is not None:
                self._preproc_viz_psd_advance_button.setEnabled(False)
            if self._preproc_viz_psd_plot_button is not None:
                self._preproc_viz_psd_plot_button.setEnabled(False)
            if self._preproc_viz_tfr_advance_button is not None:
                self._preproc_viz_tfr_advance_button.setEnabled(False)
            if self._preproc_viz_tfr_plot_button is not None:
                self._preproc_viz_tfr_plot_button.setEnabled(False)
            return

        available_steps = self._available_preproc_viz_steps(context)
        previous_step = self._current_preproc_viz_step()
        target_step = previous_step if previous_step in available_steps else None
        if target_step is None and self._preproc_viz_last_step in available_steps:
            target_step = self._preproc_viz_last_step
        if target_step is None and available_steps:
            target_step = available_steps[0]

        combo.blockSignals(True)
        combo.clear()
        if not available_steps:
            combo.addItem("No step output", None)
            combo.setCurrentIndex(0)
        else:
            for step in available_steps:
                combo.addItem(PREPROC_VIZ_STEP_LABELS[step], step)
            target_idx = combo.findData(target_step)
            combo.setCurrentIndex(target_idx if target_idx >= 0 else 0)
        combo.blockSignals(False)
        self._on_preproc_viz_step_changed(combo.currentIndex())

    def _on_preproc_viz_step_changed(self, index: int) -> None:
        _ = index
        context = self._record_context()
        step = self._current_preproc_viz_step()
        if (
            context is None
            or step is None
            or self._preproc_viz_step_combo is None
            or self._preproc_viz_step_combo.currentData() is None
        ):
            self._preproc_viz_available_channels = ()
            self._preproc_viz_selected_channels = ()
            if self._preproc_viz_channels_button is not None:
                self._preproc_viz_channels_button.setText("Select Channels (0/0)")
                self._preproc_viz_channels_button.setEnabled(False)
            if self._preproc_viz_psd_advance_button is not None:
                self._preproc_viz_psd_advance_button.setEnabled(False)
            if self._preproc_viz_psd_plot_button is not None:
                self._preproc_viz_psd_plot_button.setEnabled(False)
            if self._preproc_viz_tfr_advance_button is not None:
                self._preproc_viz_tfr_advance_button.setEnabled(False)
            if self._preproc_viz_tfr_plot_button is not None:
                self._preproc_viz_tfr_plot_button.setEnabled(False)
            return

        resolver = PathResolver(context)
        raw_path = self._preproc_step_path_for_viz(resolver, step)
        if not raw_path.exists():
            available_steps = self._available_preproc_viz_steps(context)
            fallback_step = (
                self._preproc_viz_last_step
                if self._preproc_viz_last_step in available_steps
                else None
            )
            if fallback_step is None and available_steps:
                fallback_step = available_steps[0]
            if fallback_step is not None:
                fallback_index = self._preproc_viz_step_combo.findData(fallback_step)
                if fallback_index >= 0:
                    self._show_warning(
                        "Visualization",
                        (
                            f"Step '{PREPROC_VIZ_STEP_LABELS.get(step, step)}' is unavailable. "
                            f"Falling back to '{PREPROC_VIZ_STEP_LABELS.get(fallback_step, fallback_step)}'."
                        ),
                    )
                    self._preproc_viz_step_combo.setCurrentIndex(fallback_index)
                    return

        self._preproc_viz_last_step = step
        channels = tuple(self._read_channel_names_from_raw(raw_path))
        self._preproc_viz_available_channels = channels
        self._preproc_viz_selected_channels = self._intersect_channels(
            channels,
            self._preproc_viz_selected_channels,
        )
        has_source = bool(channels and self._preproc_viz_selected_channels)
        if self._preproc_viz_channels_button is not None:
            self._preproc_viz_channels_button.setEnabled(bool(channels))
            self._preproc_viz_channels_button.setText(
                self._format_channel_button_text(
                    "Select Channels",
                    self._preproc_viz_selected_channels,
                    channels,
                )
            )
        if self._preproc_viz_psd_advance_button is not None:
            self._preproc_viz_psd_advance_button.setEnabled(has_source)
        if self._preproc_viz_psd_plot_button is not None:
            self._preproc_viz_psd_plot_button.setEnabled(has_source)
        if self._preproc_viz_tfr_advance_button is not None:
            self._preproc_viz_tfr_advance_button.setEnabled(has_source)
        if self._preproc_viz_tfr_plot_button is not None:
            self._preproc_viz_tfr_plot_button.setEnabled(has_source)

    def _on_preproc_viz_channels_select(self) -> None:
        if not self._preproc_viz_available_channels:
            self._show_warning(
                "Visualization Channels",
                "No channels available for selection.",
            )
            return
        chosen = self._run_channel_selector(
            title="Visualization Channels",
            available=self._preproc_viz_available_channels,
            selected=self._preproc_viz_selected_channels,
        )
        if chosen is None:
            return
        self._preproc_viz_selected_channels = chosen
        self._mark_record_param_dirty("preproc.viz")
        if self._preproc_viz_channels_button is not None:
            self._preproc_viz_channels_button.setText(
                self._format_channel_button_text(
                    "Select Channels",
                    self._preproc_viz_selected_channels,
                    self._preproc_viz_available_channels,
                )
            )
        has_source = bool(self._preproc_viz_selected_channels)
        if self._preproc_viz_psd_advance_button is not None:
            self._preproc_viz_psd_advance_button.setEnabled(has_source)
        if self._preproc_viz_psd_plot_button is not None:
            self._preproc_viz_psd_plot_button.setEnabled(has_source)
        if self._preproc_viz_tfr_advance_button is not None:
            self._preproc_viz_tfr_advance_button.setEnabled(has_source)
        if self._preproc_viz_tfr_plot_button is not None:
            self._preproc_viz_tfr_plot_button.setEnabled(has_source)

    def _current_preproc_viz_source(self) -> tuple[str, Path] | None:
        context = self._record_context()
        step = self._current_preproc_viz_step()
        if context is None or step is None:
            return None
        resolver = PathResolver(context)
        raw_path = self._preproc_step_path_for_viz(resolver, step)
        if raw_path.exists():
            return step, raw_path
        return None
