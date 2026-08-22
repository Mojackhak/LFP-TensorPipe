"""Step-specific preprocess plot entrypoints."""

from __future__ import annotations

from lfptensorpipe.gui.shell.common import (
    PathResolver,
    preproc_step_indicator_state,
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
    if preproc_step_indicator_state(resolver, "raw") == "green":
        self._refresh_stage_states_from_context()
        self._refresh_preproc_controls()
        self.statusBar().showMessage(
            "Current Raw opened from the accepted preprocess artifact."
        )
        self._open_mne_raw_plot(raw_path, title_prefix="Raw")
        return

    canonical_path = rawdata_input_fif_path(context)
    if canonical_path.exists():
        self._refresh_stage_states_from_context()
        self._refresh_preproc_controls()
        self.statusBar().showMessage(
            "Canonical rawdata opened for review; close the plot normally to "
            "accept it as Raw."
        )
        self._open_mne_raw_plot(
            canonical_path,
            title_prefix="Raw",
            stale_raw_review=True,
        )
        return

    self._refresh_stage_states_from_context()
    self._refresh_preproc_controls()
    if raw_path.exists():
        self.statusBar().showMessage(
            "Retained Raw opened read-only; canonical rawdata is unavailable, "
            "so closing cannot accept it."
        )
        self._open_mne_raw_plot(
            raw_path,
            title_prefix="Raw",
            stale_raw_review=True,
        )
        return
    self.statusBar().showMessage(
        "Raw Plot unavailable: canonical rawdata and retained Raw are missing."
    )


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
    context = self._record_context()
    if context is None:
        self.statusBar().showMessage(
            "Filter Plot unavailable: select project/subject/record."
        )
        return
    resolver = PathResolver(context)
    if self._preproc_filter_review_required_runtime(resolver):
        raw_path = self._preproc_filter_preview_raw_path_runtime(resolver)
    else:
        raw_path = preproc_step_raw_path(resolver, "filter")
    if not raw_path.exists():
        self.statusBar().showMessage(
            "Filter Plot unavailable: no review Preview or accepted Filter result exists."
        )
        self._refresh_preproc_controls()
        return
    self._open_mne_raw_plot(
        raw_path,
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
