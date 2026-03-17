"""Localize apply and Contact Viewer MainWindow methods."""

from __future__ import annotations


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
        if (
            not isinstance(space, str)
            or not isinstance(atlas, str)
            or not selected_regions
        ):
            return
        if not self._ensure_matlab_ready_for_action("Localize Apply"):
            return

        ok, message = self._run_with_busy(
            "Localize Apply",
            lambda: self._run_localize_apply_runtime(
                project_root=self._current_project,
                subject=self._current_subject,
                record=self._current_record,
                space=space,
                atlas=atlas,
                selected_regions=selected_regions,
                paths=self._localize_paths,
                read_only_project_root=self._demo_data_source_readonly,
            ),
        )
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
        atlas = self._localize_selected_atlas
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
