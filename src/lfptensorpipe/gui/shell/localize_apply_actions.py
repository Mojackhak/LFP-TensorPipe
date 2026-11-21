"""Localize apply and Contact Viewer MainWindow methods."""

from __future__ import annotations

from lfptensorpipe.app.localize.paths import (
    LEAD_DBS,
    localize_indicator_state,
    localize_log_path,
    source_results,
)
from lfptensorpipe.app.runlog_store import read_run_log


class MainWindowLocalizeApplyActionsMixin:
    def _on_localize_apply(self) -> None:
        if (
            self._current_project is None
            or self._current_subject is None
            or self._current_record is None
        ):
            return
        space = self._localize_inferred_space
        atlas = self._localize_selected_atlas
        selected_regions = self._localize_selected_regions
        combo = getattr(self, "_localize_source_combo", None)
        source = combo.currentData() if combo is not None else LEAD_DBS
        if source is None or (
            source == LEAD_DBS
            and (
                not isinstance(space, str)
                or not isinstance(atlas, str)
                or not selected_regions
            )
        ):
            return

        record = (self._current_project, self._current_subject, self._current_record)
        channel_names = getattr(self, "_localize_label_channels", ())
        was_ready = localize_indicator_state(*record) == "green"
        ok, message = self._run_with_busy(
            "Localize Apply",
            lambda: self._run_localize_apply_runtime(
                project_root=record[0],
                subject=record[1],
                record=record[2],
                space=space or "",
                atlas=atlas or "",
                selected_regions=selected_regions,
                source=source,
                channels=channel_names,
                paths=self._localize_paths,
                read_only_project_root=self._demo_data_source_readonly,
            ),
        )
        if ok or (was_ready and localize_indicator_state(*record) != "green"):
            self._refresh_stage_states_from_context()
        self._refresh_localize_action_state()
        prefix = "Localize OK" if ok else "Localize Failed"
        self.statusBar().showMessage(f"{prefix}: {message}")
        self._post_step_action_sync(reason="localize_apply")

    def _on_contact_viewer(self) -> None:
        if (
            self._current_project is None
            or self._current_subject is None
            or self._current_record is None
        ):
            return
        payload = read_run_log(
            localize_log_path(
                self._current_project, self._current_subject, self._current_record
            )
        )
        atlas = source_results(payload).get(LEAD_DBS, {}).get("atlas")
        if not isinstance(atlas, str):
            self.statusBar().showMessage(
                "Contact Viewer unavailable: save atlas config first."
            )
            return
        ok, message = self._run_with_busy(
            "Contact Viewer",
            lambda: self._launch_contact_viewer_runtime(
                project_root=self._current_project,
                subject=self._current_subject,
                record=self._current_record,
                atlas=atlas,
                paths=self._localize_paths,
            ),
        )
        if not ok:
            self.statusBar().showMessage(f"Contact Viewer unavailable: {message}")
            return
        self.statusBar().showMessage(message)
