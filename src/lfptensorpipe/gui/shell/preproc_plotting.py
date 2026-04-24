"""Preprocess plotting mixin facade."""

from __future__ import annotations

from .preproc_plotting_backend import (
    _attach_plot_autosave as _attach_plot_autosave_impl,
    _defer_close_for_active_mne_browsers as _defer_close_for_active_mne_browsers_impl,
    _finalize_app_shutdown as _finalize_app_shutdown_impl,
    _open_mne_raw_plot as _open_mne_raw_plot_impl,
    _request_close_all_mne_browsers as _request_close_all_mne_browsers_impl,
)
from .preproc_plotting_steps import (
    _on_preproc_annotations_plot as _on_preproc_annotations_plot_impl,
    _on_preproc_bad_segment_plot as _on_preproc_bad_segment_plot_impl,
    _on_preproc_ecg_plot as _on_preproc_ecg_plot_impl,
    _on_preproc_filter_plot as _on_preproc_filter_plot_impl,
    _on_preproc_finish_plot as _on_preproc_finish_plot_impl,
    _on_preproc_raw_plot as _on_preproc_raw_plot_impl,
)
from .preproc_plotting_viz import (
    _on_preproc_viz_psd_advance as _on_preproc_viz_psd_advance_impl,
    _on_preproc_viz_psd_plot as _on_preproc_viz_psd_plot_impl,
    _on_preproc_viz_tfr_advance as _on_preproc_viz_tfr_advance_impl,
    _on_preproc_viz_tfr_plot as _on_preproc_viz_tfr_plot_impl,
)


class MainWindowPreprocPlottingMixin:
    def _defer_close_for_active_mne_browsers(self, event) -> bool:
        return _defer_close_for_active_mne_browsers_impl(self, event)

    def _request_close_all_mne_browsers(self) -> None:
        _request_close_all_mne_browsers_impl(self)

    def _finalize_app_shutdown(self) -> None:
        _finalize_app_shutdown_impl(self)

    def _on_preproc_viz_psd_advance(self) -> None:
        _on_preproc_viz_psd_advance_impl(self)

    def _on_preproc_viz_tfr_advance(self) -> None:
        _on_preproc_viz_tfr_advance_impl(self)

    def _on_preproc_viz_psd_plot(self) -> None:
        _on_preproc_viz_psd_plot_impl(self)

    def _on_preproc_viz_tfr_plot(self) -> None:
        _on_preproc_viz_tfr_plot_impl(self)

    def _on_preproc_raw_plot(self) -> None:
        _on_preproc_raw_plot_impl(self)

    def _on_preproc_filter_plot(self) -> None:
        _on_preproc_filter_plot_impl(self)

    def _on_preproc_annotations_plot(self) -> None:
        _on_preproc_annotations_plot_impl(self)

    def _on_preproc_bad_segment_plot(self) -> None:
        _on_preproc_bad_segment_plot_impl(self)

    def _on_preproc_ecg_plot(self) -> None:
        _on_preproc_ecg_plot_impl(self)

    def _on_preproc_finish_plot(self) -> None:
        _on_preproc_finish_plot_impl(self)

    def _attach_plot_autosave(
        self,
        *,
        browser,
        raw,
        raw_path,
        step: str,
        title_prefix: str,
    ) -> None:
        _attach_plot_autosave_impl(
            self,
            browser=browser,
            raw=raw,
            raw_path=raw_path,
            step=step,
            title_prefix=title_prefix,
        )

    def _open_mne_raw_plot(
        self,
        raw_path,
        title_prefix: str,
        *,
        autosave_step: str | None = None,
    ) -> None:
        _open_mne_raw_plot_impl(
            self,
            raw_path=raw_path,
            title_prefix=title_prefix,
            autosave_step=autosave_step,
        )
