"""Explicit MainWindow dependency seams for dialogs, I/O, and runtime helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lfptensorpipe.app import (
    alignment_epoch_inspector_state,
    alignment_metric_tensor_warped_path,
    alignment_method_panel_state,
    alignment_trial_stage_state,
    apply_annotations_step,
    apply_bad_segment_step,
    apply_ecg_step,
    apply_filter_step,
    apply_finish_step,
    apply_reset_reference,
    bootstrap_raw_step_from_rawdata,
    can_open_contact_viewer,
    create_alignment_paradigm,
    create_subject,
    default_alignment_method_params,
    delete_alignment_paradigm,
    delete_record,
    discover_atlases,
    discover_records,
    discover_subjects,
    extract_features_indicator_state,
    features_panel_state,
    finish_alignment_epochs,
    has_reconstruction_mat,
    import_record_from_raw,
    infer_subject_space,
    load_alignment_annotation_labels,
    persist_alignment_epoch_picks,
    load_alignment_epoch_rows,
    load_alignment_paradigms,
    load_annotations_csv_rows,
    load_burst_baseline_annotation_labels,
    load_derive_defaults,
    load_localize_paths,
    load_reconstruction_contacts,
    localize_indicator_state,
    localize_panel_state,
    mark_preproc_step,
    preproc_annotations_panel_state,
    preproc_ecg_panel_state,
    preproc_filter_panel_state,
    rename_record,
    upgrade_record_run_logs,
    run_align_epochs,
    run_extract_features,
    run_localize_apply,
    tensor_metric_panel_state,
    update_alignment_paradigm,
    validate_alignment_method_params,
    validate_tensor_frequency_params,
)
from lfptensorpipe.gui.dialogs.alignment_method_params import (
    AlignmentMethodParamsDialog,
)
from lfptensorpipe.gui.dialogs.annotation_configure import AnnotationConfigureDialog
from lfptensorpipe.gui.dialogs.autosave_filter import _CloseAutosaveFilter
from lfptensorpipe.gui.dialogs.channel_select import ChannelSelectDialog
from lfptensorpipe.gui.dialogs.feature_axis import FeatureAxisConfigureDialog
from lfptensorpipe.gui.dialogs.features_plot_advance import FeaturesPlotAdvanceDialog
from lfptensorpipe.gui.dialogs.filter_advance import FilterAdvanceDialog
from lfptensorpipe.gui.dialogs.localize_atlas import LocalizeAtlasDialog
from lfptensorpipe.gui.dialogs.localize_match import LocalizeMatchDialog
from lfptensorpipe.gui.dialogs.paths_config import PathsConfigDialog
from lfptensorpipe.gui.dialogs.qc_advance import QcAdvanceDialog
from lfptensorpipe.gui.dialogs.record_import import RecordImportDialog
from lfptensorpipe.gui.dialogs.tensor_channel_select import TensorChannelSelectDialog
from lfptensorpipe.gui.dialogs.tensor_metric_advance import TensorMetricAdvanceDialog
from lfptensorpipe.gui.dialogs.tensor_pair_select import TensorPairSelectDialog
from lfptensorpipe.io.pkl_io import load_pkl, save_pkl
from lfptensorpipe.anat.lead_config import discover_regions
from lfptensorpipe.stats.preproc.normalize import normalize_df_by_baseline
from lfptensorpipe.stats.preproc.transform import transform_df
from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox


class MainWindowRuntimeDependenciesMixin:
    def _show_warning(self, title: str, message: str) -> int:
        return QMessageBox.warning(self, title, message)

    def _show_information(self, title: str, message: str) -> int:
        return QMessageBox.information(self, title, message)

    def _ask_question(
        self,
        title: str,
        message: str,
        *,
        buttons: QMessageBox.StandardButton = QMessageBox.Yes | QMessageBox.No,
        default_button: QMessageBox.StandardButton = QMessageBox.No,
    ) -> QMessageBox.StandardButton:
        return QMessageBox.question(
            self,
            title,
            message,
            buttons,
            default_button,
        )

    def _select_existing_directory(
        self,
        title: str,
        start_dir: str,
    ) -> str:
        return QFileDialog.getExistingDirectory(self, title, start_dir)

    def _open_file_name(
        self,
        title: str,
        start_dir: str,
        file_filter: str,
    ) -> tuple[str, str]:
        return QFileDialog.getOpenFileName(self, title, start_dir, file_filter)

    def _save_file_name(
        self,
        title: str,
        start_path: str,
        file_filter: str,
    ) -> tuple[str, str]:
        return QFileDialog.getSaveFileName(self, title, start_path, file_filter)

    def _prompt_text(
        self,
        title: str,
        label: str,
        *,
        text: str = "",
    ) -> tuple[str, bool]:
        return QInputDialog.getText(self, title, label, text=text)

    def _create_record_import_dialog(self, **kwargs: Any) -> RecordImportDialog:
        return RecordImportDialog(**kwargs)

    def _create_paths_config_dialog(self, **kwargs: Any) -> PathsConfigDialog:
        return PathsConfigDialog(**kwargs)

    def _create_qc_advance_dialog(self, **kwargs: Any) -> QcAdvanceDialog:
        return QcAdvanceDialog(**kwargs)

    def _create_channel_select_dialog(self, **kwargs: Any) -> ChannelSelectDialog:
        return ChannelSelectDialog(**kwargs)

    def _create_localize_match_dialog(self, **kwargs: Any) -> LocalizeMatchDialog:
        return LocalizeMatchDialog(**kwargs)

    def _create_filter_advance_dialog(self, **kwargs: Any) -> FilterAdvanceDialog:
        return FilterAdvanceDialog(**kwargs)

    def _create_annotation_configure_dialog(
        self,
        **kwargs: Any,
    ) -> AnnotationConfigureDialog:
        return AnnotationConfigureDialog(**kwargs)

    def _create_feature_axis_configure_dialog(
        self,
        **kwargs: Any,
    ) -> FeatureAxisConfigureDialog:
        return FeatureAxisConfigureDialog(**kwargs)

    def _create_features_plot_advance_dialog(
        self,
        **kwargs: Any,
    ) -> FeaturesPlotAdvanceDialog:
        return FeaturesPlotAdvanceDialog(**kwargs)

    def _create_alignment_method_params_dialog(
        self,
        **kwargs: Any,
    ) -> AlignmentMethodParamsDialog:
        return AlignmentMethodParamsDialog(**kwargs)

    def _create_localize_atlas_dialog(self, **kwargs: Any) -> LocalizeAtlasDialog:
        return LocalizeAtlasDialog(**kwargs)

    def _create_tensor_channel_select_dialog(
        self,
        **kwargs: Any,
    ) -> TensorChannelSelectDialog:
        return TensorChannelSelectDialog(**kwargs)

    def _create_tensor_pair_select_dialog(
        self,
        **kwargs: Any,
    ) -> TensorPairSelectDialog:
        return TensorPairSelectDialog(**kwargs)

    def _create_tensor_metric_advance_dialog(
        self,
        **kwargs: Any,
    ) -> TensorMetricAdvanceDialog:
        return TensorMetricAdvanceDialog(**kwargs)

    def _close_autosave_filter_class(self) -> type[_CloseAutosaveFilter]:
        return _CloseAutosaveFilter

    def _create_subject_runtime(
        self,
        project_root: Path,
        subject: str,
    ) -> tuple[bool, str]:
        return create_subject(project_root, subject)

    def _discover_subjects_runtime(self, project_root: Path) -> list[str]:
        return discover_subjects(project_root)

    def _discover_records_runtime(
        self,
        project_root: Path,
        subject: str,
    ) -> list[str]:
        return discover_records(project_root, subject)

    def _upgrade_record_run_logs_runtime(
        self,
        project_root: Path,
        subject: str,
        record: str,
    ) -> Any:
        return upgrade_record_run_logs(project_root, subject, record)

    def _apply_reset_reference_runtime(
        self,
        raw: Any,
        rows: tuple[tuple[str, str, str], ...],
    ) -> Any:
        return apply_reset_reference(raw, rows)

    def _import_record_from_raw_runtime(self, **kwargs: Any) -> Any:
        return import_record_from_raw(**kwargs)

    def _delete_record_runtime(self, **kwargs: Any) -> Any:
        return delete_record(**kwargs)

    def _rename_record_runtime(self, **kwargs: Any) -> Any:
        return rename_record(**kwargs)

    def _load_localize_paths_runtime(self, config_store: Any) -> dict[str, str]:
        return load_localize_paths(config_store)

    def _discover_atlases_runtime(self, *args: Any, **kwargs: Any) -> list[str]:
        return discover_atlases(*args, **kwargs)

    def _discover_localize_regions_runtime(
        self, *args: Any, **kwargs: Any
    ) -> list[str]:
        return discover_regions(*args, **kwargs)

    def _infer_subject_space_runtime(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[str | None, str]:
        return infer_subject_space(*args, **kwargs)

    def _localize_indicator_state_runtime(self, *args: Any, **kwargs: Any) -> str:
        return localize_indicator_state(*args, **kwargs)

    def _localize_panel_state_runtime(self, *args: Any, **kwargs: Any) -> str:
        return localize_panel_state(*args, **kwargs)

    def _has_reconstruction_mat_runtime(self, *args: Any, **kwargs: Any) -> bool:
        return has_reconstruction_mat(*args, **kwargs)

    def _can_open_contact_viewer_runtime(self, *args: Any, **kwargs: Any) -> bool:
        return can_open_contact_viewer(*args, **kwargs)

    def _run_localize_apply_runtime(self, **kwargs: Any) -> Any:
        return run_localize_apply(**kwargs)

    def _launch_contact_viewer_runtime(self, **kwargs: Any) -> Any:
        from lfptensorpipe.app import launch_contact_viewer

        return launch_contact_viewer(**kwargs)

    def _load_reconstruction_contacts_runtime(self, *args: Any) -> Any:
        return load_reconstruction_contacts(*args)

    def _load_derive_defaults_runtime(self, config_store: Any) -> dict[str, Any]:
        return load_derive_defaults(config_store)

    def _extract_features_indicator_state_runtime(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        return extract_features_indicator_state(*args, **kwargs)

    def _features_panel_state_runtime(self, *args: Any, **kwargs: Any) -> str:
        return features_panel_state(*args, **kwargs)

    def _run_extract_features_runtime(self, *args: Any, **kwargs: Any) -> Any:
        return run_extract_features(*args, **kwargs)

    def _create_alignment_paradigm_runtime(
        self,
        config_store: Any,
        **kwargs: Any,
    ) -> Any:
        return create_alignment_paradigm(config_store, **kwargs)

    def _delete_alignment_paradigm_runtime(
        self,
        config_store: Any,
        **kwargs: Any,
    ) -> Any:
        return delete_alignment_paradigm(config_store, **kwargs)

    def _load_alignment_paradigms_runtime(
        self,
        config_store: Any,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        return load_alignment_paradigms(config_store, **kwargs)

    def _load_alignment_epoch_rows_runtime(
        self,
        context: Any,
        *,
        paradigm_slug: str,
    ) -> list[dict[str, Any]]:
        return load_alignment_epoch_rows(context, paradigm_slug=paradigm_slug)

    def _persist_alignment_epoch_picks_runtime(
        self,
        context: Any,
        *,
        paradigm_slug: str,
        picked_epoch_indices: list[int] | None,
    ) -> bool:
        return persist_alignment_epoch_picks(
            context,
            paradigm_slug=paradigm_slug,
            picked_epoch_indices=picked_epoch_indices,
        )

    def _default_alignment_method_params_runtime(
        self, method_key: str
    ) -> dict[str, Any]:
        return default_alignment_method_params(method_key)

    def _load_alignment_annotation_labels_runtime(self, context: Any) -> list[str]:
        return load_alignment_annotation_labels(context)

    def _validate_alignment_method_params_runtime(
        self,
        method_key: str,
        params: dict[str, Any],
        *,
        annotation_labels: list[str],
    ) -> tuple[bool, dict[str, Any], str]:
        return validate_alignment_method_params(
            method_key,
            params,
            annotation_labels=annotation_labels,
        )

    def _update_alignment_paradigm_runtime(
        self, config_store: Any, **kwargs: Any
    ) -> Any:
        return update_alignment_paradigm(config_store, **kwargs)

    def _run_align_epochs_runtime(self, context: Any, **kwargs: Any) -> Any:
        return run_align_epochs(context, **kwargs)

    def _alignment_metric_tensor_warped_path_runtime(
        self,
        resolver: Any,
        paradigm_slug: str,
        metric_key: str,
    ) -> Path:
        return alignment_metric_tensor_warped_path(resolver, paradigm_slug, metric_key)

    def _alignment_method_panel_state_runtime(
        self,
        resolver: Any,
        *,
        paradigm: dict[str, Any] | None,
    ) -> str:
        return alignment_method_panel_state(resolver, paradigm=paradigm)

    def _alignment_epoch_inspector_state_runtime(
        self,
        resolver: Any,
        *,
        paradigm: dict[str, Any] | None,
        picked_epoch_indices: list[int] | None,
    ) -> str:
        return alignment_epoch_inspector_state(
            resolver,
            paradigm=paradigm,
            picked_epoch_indices=picked_epoch_indices,
        )

    def _alignment_trial_stage_state_runtime(
        self,
        resolver: Any,
        *,
        paradigm_slug: str,
    ) -> str:
        return alignment_trial_stage_state(resolver, paradigm_slug=paradigm_slug)

    def _finish_alignment_epochs_runtime(self, context: Any, **kwargs: Any) -> Any:
        return finish_alignment_epochs(context, **kwargs)

    def _bootstrap_raw_step_from_rawdata_runtime(self, context: Any) -> Any:
        return bootstrap_raw_step_from_rawdata(context)

    def _apply_filter_step_runtime(self, context: Any, **kwargs: Any) -> Any:
        return apply_filter_step(context, **kwargs)

    def _apply_annotations_step_runtime(self, context: Any, **kwargs: Any) -> Any:
        return apply_annotations_step(context, **kwargs)

    def _apply_bad_segment_step_runtime(self, context: Any) -> Any:
        return apply_bad_segment_step(context)

    def _apply_ecg_step_runtime(self, context: Any, **kwargs: Any) -> Any:
        return apply_ecg_step(context, **kwargs)

    def _apply_finish_step_runtime(self, context: Any) -> Any:
        return apply_finish_step(context)

    def _mark_preproc_step_runtime(self, **kwargs: Any) -> Any:
        return mark_preproc_step(**kwargs)

    def _preproc_filter_panel_state_runtime(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        return preproc_filter_panel_state(*args, **kwargs)

    def _preproc_annotations_panel_state_runtime(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        return preproc_annotations_panel_state(*args, **kwargs)

    def _preproc_ecg_panel_state_runtime(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        return preproc_ecg_panel_state(*args, **kwargs)

    def _load_annotations_csv_rows_runtime(
        self,
        path: Path,
    ) -> tuple[bool, list[dict[str, Any]], str]:
        return load_annotations_csv_rows(path)

    def _validate_tensor_frequency_params_runtime(
        self,
        context: Any,
        *,
        low_freq: float,
        high_freq: float,
        step_hz: float,
    ) -> tuple[bool, str, dict[str, Any]]:
        return validate_tensor_frequency_params(
            context,
            low_freq=low_freq,
            high_freq=high_freq,
            step_hz=step_hz,
        )

    def _tensor_metric_panel_state_runtime(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        return tensor_metric_panel_state(*args, **kwargs)

    def _load_burst_baseline_annotation_labels_runtime(self, context: Any) -> list[str]:
        return load_burst_baseline_annotation_labels(context)

    def _load_pickle(self, path: Path) -> Any:
        return load_pkl(path)

    def _save_pickle(self, payload: Any, path: Path) -> None:
        save_pkl(payload, path)

    def _transform_dataframe(
        self,
        payload: Any,
        **kwargs: Any,
    ) -> Any:
        return transform_df(payload, **kwargs)

    def _normalize_dataframe_by_baseline(
        self,
        payload: Any,
        **kwargs: Any,
    ) -> Any:
        return normalize_df_by_baseline(payload, **kwargs)

    def _read_raw_fif(
        self,
        path: Path,
        *,
        preload: bool,
        verbose: str = "ERROR",
    ) -> Any:
        import mne

        return mne.io.read_raw_fif(str(path), preload=preload, verbose=verbose)

    def _compute_tfr_array_morlet(
        self,
        data: Any,
        *,
        sfreq: float,
        freqs: Any,
        n_cycles: Any,
        output: str,
        decim: int,
    ) -> Any:
        from mne.time_frequency import tfr_array_morlet

        return tfr_array_morlet(
            data,
            sfreq=sfreq,
            freqs=freqs,
            n_cycles=n_cycles,
            output=output,
            decim=decim,
        )

    def _create_matplotlib_subplots(self) -> tuple[Any, Any]:
        import matplotlib.pyplot as plt

        return plt.subplots()

    def _load_cmcrameri_vik(self) -> Any:
        import cmcrameri.cm as cmc

        return cmc.vik

    def _plot_single_effect_series(self, *args: Any, **kwargs: Any) -> Any:
        from lfptensorpipe.viz.visualdf import plot_single_effect_series

        return plot_single_effect_series(*args, **kwargs)

    def _plot_single_effect_df(self, *args: Any, **kwargs: Any) -> Any:
        from lfptensorpipe.viz.visualdf import plot_single_effect_df

        return plot_single_effect_df(*args, **kwargs)

    def _plot_single_effect_scalar(self, *args: Any, **kwargs: Any) -> Any:
        from lfptensorpipe.viz.visualdf import plot_single_effect_scalar

        return plot_single_effect_scalar(*args, **kwargs)
