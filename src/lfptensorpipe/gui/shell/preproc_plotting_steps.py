"""Step-specific preprocess plot entrypoints."""

from __future__ import annotations

from lfptensorpipe.gui.shell.common import (
    PathResolver,
    indicator_from_log,
    preproc_step_log_path,
    preproc_step_raw_path,
    rawdata_input_fif_path,
)


def _on_preproc_raw_plot(self) -> None:
    context = self._record_context()
    if context is None:
        self.statusBar().showMessage(
            "Raw Plot unavailable: select project/subject/record."
        )
        return
    resolver = PathResolver(context)
    raw_path = preproc_step_raw_path(resolver, "raw")
    raw_log_path = preproc_step_log_path(resolver, "raw")
    if raw_path.exists():
        if indicator_from_log(raw_log_path) != "green":
            src_path = rawdata_input_fif_path(context)
            self._mark_preproc_step_runtime(
                resolver=resolver,
                step="raw",
                completed=True,
                input_path=str(src_path if src_path.exists() else raw_path),
                output_path=str(raw_path),
                message="Opened existing preprocess raw artifact.",
            )
        self._refresh_stage_states_from_context()
        self._refresh_preproc_controls()
        self.statusBar().showMessage(
            "Raw step opened from existing preprocess artifact."
        )
        self._open_mne_raw_plot(raw_path, title_prefix="Raw", autosave_step="raw")
        return
    ok, message = self._run_with_busy(
        "Raw Plot",
        lambda: self._bootstrap_raw_step_from_rawdata_runtime(context),
    )
    self._refresh_preproc_controls()
    if not ok:
        self.statusBar().showMessage(f"Raw Plot failed: {message}")
        return
    self.statusBar().showMessage(message)
    self._open_mne_raw_plot(raw_path, title_prefix="Raw", autosave_step="raw")


def _open_step_plot(
    self,
    *,
    step: str,
    missing_message: str,
    title_prefix: str,
    autosave_step: str | None = None,
) -> None:
    context = self._record_context()
    if context is None:
        self.statusBar().showMessage(
            f"{title_prefix} Plot unavailable: select project/subject/record."
        )
        return
    raw_path = preproc_step_raw_path(PathResolver(context), step)
    if not raw_path.exists():
        self.statusBar().showMessage(missing_message)
        self._refresh_preproc_controls()
        return
    self._open_mne_raw_plot(
        raw_path, title_prefix=title_prefix, autosave_step=autosave_step
    )


def _on_preproc_filter_plot(self) -> None:
    _open_step_plot(
        self,
        step="filter",
        missing_message="Filter Plot unavailable: filter/raw.fif is missing.",
        title_prefix="Filter",
        autosave_step="filter",
    )


def _on_preproc_annotations_plot(self) -> None:
    _open_step_plot(
        self,
        step="annotations",
        missing_message="Annotations Plot unavailable: annotations/raw.fif is missing.",
        title_prefix="Annotations",
        autosave_step="annotations",
    )


def _on_preproc_bad_segment_plot(self) -> None:
    _open_step_plot(
        self,
        step="bad_segment_removal",
        missing_message="Bad Segment Plot unavailable: bad_segment_removal/raw.fif is missing.",
        title_prefix="Bad Segment",
        autosave_step="bad_segment_removal",
    )


def _on_preproc_ecg_plot(self) -> None:
    _open_step_plot(
        self,
        step="ecg_artifact_removal",
        missing_message="ECG Plot unavailable: ecg_artifact_removal/raw.fif is missing.",
        title_prefix="ECG",
        autosave_step="ecg_artifact_removal",
    )


def _on_preproc_finish_plot(self) -> None:
    context = self._record_context()
    if context is None:
        self.statusBar().showMessage(
            "Finish Plot unavailable: select project/subject/record."
        )
        return
    finish_path = PathResolver(context).preproc_root / "finish" / "raw.fif"
    if not finish_path.exists():
        self.statusBar().showMessage(
            "Finish Plot unavailable: finish/raw.fif is missing."
        )
        self._refresh_preproc_controls()
        return
    self._open_mne_raw_plot(finish_path, title_prefix="Finish")
