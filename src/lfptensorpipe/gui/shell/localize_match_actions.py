"""Localize match MainWindow methods."""

from __future__ import annotations

from lfptensorpipe.gui.shell.common import (
    QDialog,
    rawdata_input_fif_path,
)


class MainWindowLocalizeMatchActionsMixin:
    def _load_record_channels_for_match(self) -> tuple[bool, str, tuple[str, ...]]:
        context = self._record_context()
        if context is None:
            return False, "Select project/subject/record first.", ()
        fif_path = rawdata_input_fif_path(context)
        if not fif_path.exists():
            return False, f"Missing raw.fif: {fif_path}", ()
        try:
            import mne

            raw = mne.io.read_raw_fif(str(fif_path), preload=False, verbose="ERROR")
            channels = tuple(str(name) for name in raw.ch_names)
            if hasattr(raw, "close"):
                raw.close()
        except Exception as exc:  # noqa: BLE001
            return False, f"Failed to load record channels: {exc}", ()
        if not channels:
            return False, "No channels found in raw.fif.", ()
        return True, "", channels

    def _on_localize_match(self) -> None:
        if (
            self._current_project is None
            or self._current_subject is None
            or self._current_record is None
        ):
            self.statusBar().showMessage(
                "Match unavailable: select project/subject/record."
            )
            return
        if not self._localize_reconstruction_exists:
            self.statusBar().showMessage(
                "Match unavailable: reconstruction.mat is missing."
            )
            return
        if not self._ensure_matlab_ready_for_action("Match"):
            return
        project_root = self._current_project
        subject = self._current_subject
        if project_root is None or subject is None:
            return
        active_context = (
            str(project_root),
            subject,
            self._current_record,
        )
        ok_summary, message_summary, summary = self._run_with_busy(
            "Localize Match",
            lambda: self._load_reconstruction_contacts_runtime(
                project_root,
                subject,
                self._localize_paths,
            ),
        )
        if (
            self._current_project is None
            or self._current_subject is None
            or self._current_record is None
        ):
            return
        current_context = (
            str(self._current_project),
            self._current_subject,
            self._current_record,
        )
        if active_context != current_context:
            self.statusBar().showMessage(
                "Match cancelled: stale subject/record context."
            )
            return
        if not ok_summary:
            self._show_warning("Match", message_summary)
            self.statusBar().showMessage(
                f"Match unavailable: failed to load reconstruction contacts ({message_summary})."
            )
            return
        self._localize_reconstruction_summary = summary

        ok_channels, message_channels, channels = self._load_record_channels_for_match()
        if not ok_channels:
            self._show_warning("Match", message_channels)
            return

        current_payload = self._load_localize_match_payload()
        dialog = self._create_localize_match_dialog(
            channel_names=channels,
            lead_specs=[
                item for item in summary.get("leads", []) if isinstance(item, dict)
            ],
            current_payload=current_payload,
            config_store=self._config_store,
            parent=self,
        )
        if dialog.exec() != QDialog.Accepted or dialog.selected_payload is None:
            return
        payload = dict(dialog.selected_payload)
        payload.update(
            {
                "subject": self._current_subject,
                "record": self._current_record,
                "space": self._localize_inferred_space,
                "atlas": self._localize_selected_atlas,
            }
        )
        self._save_localize_match_payload(payload)
        self._persist_record_params_snapshot(reason="localize_match_save")
        self._localize_match_completed = bool(payload.get("completed", False))
        self._refresh_localize_match_status()
        self._refresh_localize_action_state()
        self.statusBar().showMessage("Localize match mapping saved.")
