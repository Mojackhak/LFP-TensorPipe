"""Localize state and control MainWindow methods."""

from __future__ import annotations

import csv
from lfptensorpipe.app.localize.paths import (
    LEAD_DBS,
    USER_LABELS,
    applied_source,
    localize_match_signature,
    localize_record_dir,
    localize_selected_regions_signature,
    source_files_ready,
    source_results,
)
from lfptensorpipe.app.localize.lead_inputs import compare_lead_inputs
from lfptensorpipe.app.localize.user_labels import UserLabels, read_labels, save_labels
from lfptensorpipe.app.runlog_store import read_run_log
from PySide6.QtWidgets import QDialog

from lfptensorpipe.gui.shell.common import (
    Any,
    _nested_get,
    _nested_set,
)

SPACE_LOCALIZE_DEFAULTS_KEY = "space_localize_defaults"


class MainWindowLocalizeStateMixin:
    def _load_localization_config(self) -> dict[str, Any]:
        payload = self._config_store.read_yaml("localization.yml", default={})
        return dict(payload) if isinstance(payload, dict) else {}

    def _load_localize_defaults(self) -> dict[str, dict[str, Any]]:
        payload = self._load_localization_config()
        raw_defaults = payload.get(SPACE_LOCALIZE_DEFAULTS_KEY, {})
        if not isinstance(raw_defaults, dict):
            return {}

        out: dict[str, dict[str, Any]] = {}
        for raw_space, raw_entry in raw_defaults.items():
            space = str(raw_space).strip()
            if not space or not isinstance(raw_entry, dict):
                continue
            atlas = str(raw_entry.get("atlas", "")).strip()
            selected_regions = [
                str(item).strip()
                for item in raw_entry.get("selected_regions", [])
                if str(item).strip()
            ]
            out[space] = {
                "atlas": atlas,
                "selected_regions": selected_regions,
            }
        return out

    def _localize_atlas_path(self, atlas: str) -> Any | None:
        atlas_name = str(atlas).strip()
        if (
            not atlas_name
            or self._localize_inferred_space is None
            or atlas_name not in self._localize_available_atlases
        ):
            return None
        return (
            self._localize_paths.leaddbs_dir
            / "templates"
            / "space"
            / self._localize_inferred_space
            / "atlases"
            / atlas_name
        )

    def _localize_region_names_for_atlas(self, atlas: str) -> tuple[str, ...]:
        atlas_name = str(atlas).strip()
        if not atlas_name:
            return ()
        cached = self._localize_region_names_by_atlas.get(atlas_name)
        if cached is not None:
            return cached
        atlas_path = self._localize_atlas_path(atlas_name)
        if atlas_path is None or not atlas_path.is_dir():
            self._localize_region_names_by_atlas[atlas_name] = ()
            return ()
        try:
            region_names = tuple(self._discover_localize_regions_runtime(atlas_path))
        except Exception:
            region_names = ()
        self._localize_region_names_by_atlas[atlas_name] = region_names
        return region_names

    def _normalize_localize_selected_regions(
        self,
        atlas: str,
        selected_regions: Any,
    ) -> tuple[str, ...]:
        region_names = self._localize_region_names_for_atlas(atlas)
        valid = set(region_names)
        if not valid or not isinstance(selected_regions, list | tuple):
            return ()
        seen: set[str] = set()
        out: list[str] = []
        for raw_name in selected_regions:
            name = str(raw_name).strip()
            if not name or name not in valid or name in seen:
                continue
            seen.add(name)
            out.append(name)
        return tuple(out)

    def _localize_atlas_config(
        self, params: dict[str, Any]
    ) -> tuple[str | None, tuple[str, ...]]:
        atlas = _nested_get(params, ("localize", "atlas"))
        atlas_name = str(atlas).strip() if isinstance(atlas, str) else ""
        if not atlas_name or atlas_name not in self._localize_available_atlases:
            return None, ()
        selected_regions = _nested_get(params, ("localize", "selected_regions"))
        return atlas_name, self._normalize_localize_selected_regions(
            atlas_name,
            selected_regions,
        )

    def _save_localize_atlas_config(
        self,
        *,
        atlas: str,
        selected_regions: tuple[str, ...] | list[str],
    ) -> bool:
        context = self._record_context()
        if context is None:
            return False
        atlas_name = str(atlas).strip()
        normalized_regions = self._normalize_localize_selected_regions(
            atlas_name,
            selected_regions,
        )
        ok, existing, _ = self._load_record_params_payload(context)
        if not ok:
            existing = {}
        merged = dict(existing)
        _nested_set(merged, ("localize", "atlas"), atlas_name)
        _nested_set(merged, ("localize", "selected_regions"), list(normalized_regions))
        wrote = self._write_record_params_payload(
            context,
            params=merged,
            reason="localize_atlas_save",
        )
        if wrote:
            self._localize_selected_atlas = atlas_name
            self._localize_selected_regions = normalized_regions
            self._record_param_dirty_keys.discard("localize.atlas")
            self._record_param_dirty_keys.discard("localize.selected_regions")
        return wrote

    def _save_localize_draft(
        self,
        *,
        atlas: str,
        selected_regions: tuple[str, ...] | list[str],
        match_payload: dict[str, Any],
    ) -> bool:
        context = self._record_context()
        if context is None:
            return False
        atlas_name = str(atlas).strip()
        normalized_regions = self._normalize_localize_selected_regions(
            atlas_name,
            selected_regions,
        )
        ok, existing, _ = self._load_record_params_payload(context)
        if not ok:
            existing = {}
        merged = dict(existing)
        _nested_set(merged, ("localize", "atlas"), atlas_name)
        _nested_set(merged, ("localize", "selected_regions"), list(normalized_regions))
        _nested_set(merged, ("localize", "match"), dict(match_payload))
        wrote = self._write_record_params_payload(
            context,
            params=merged,
            reason="localize_config_import",
        )
        if wrote:
            self._localize_selected_atlas = atlas_name
            self._localize_selected_regions = normalized_regions
            self._localize_match_payload = dict(match_payload)
            self._localize_match_completed = bool(match_payload.get("completed", False))
            self._record_param_dirty_keys.discard("localize.atlas")
            self._record_param_dirty_keys.discard("localize.selected_regions")
            self._record_param_dirty_keys.discard("localize.match")
        return wrote

    def _load_localize_match_payload(self) -> dict[str, Any] | None:
        context = self._record_context()
        if context is None:
            return None
        ok, params, _ = self._load_record_params_payload(context)
        if not ok:
            return None
        payload = _nested_get(params, ("localize", "match"))
        if not isinstance(payload, dict):
            return None
        return payload

    def _save_localize_match_payload(self, payload: dict[str, Any]) -> None:
        context = self._record_context()
        if context is None:
            return
        ok, existing, _ = self._load_record_params_payload(context)
        if not ok:
            existing = {}
        merged = dict(existing)
        _nested_set(merged, ("localize", "match"), dict(payload))
        if self._write_record_params_payload(
            context,
            params=merged,
            reason="localize_match_save",
        ):
            self._localize_match_payload = dict(payload)
            self._record_param_dirty_keys.discard("localize.match")

    def _refresh_localize_match_status(self) -> None:
        if self._localize_match_status_label is None:
            return
        channels = set(self._localize_label_channels)
        mappings = (self._localize_match_payload or {}).get("mappings", [])
        mapped = len(
            {r.get("channel") for r in mappings if isinstance(r, dict)} & channels
        )
        total = len(channels)
        self._localize_match_status_label.setText(f"{mapped}/{total} mapped")
        self._localize_match_status_label.setToolTip(
            "Mapped channels / total channels. "
            f"Apply requires full mapping ({mapped}/{total})."
        )

    def _refresh_localize_atlas_summary(self) -> None:
        if self._localize_atlas_summary_label is None:
            return
        atlas = self._localize_selected_atlas or ""
        total = len(self._localize_region_names_for_atlas(atlas)) if atlas else 0
        selected = len(self._localize_selected_regions) if atlas else 0
        self._localize_atlas_summary_label.setText(f"{selected}/{total} regions")
        self._localize_atlas_summary_label.setToolTip(
            "Saved interested-region count for the current Localize atlas config. "
            f"Current: {selected}/{total}."
        )

    def _refresh_localize_controls(self) -> None:
        has_subject_context = (
            self._current_project is not None and self._current_subject is not None
        )
        context = self._record_context()

        self._localize_inferred_space = None
        self._localize_space_error = ""
        self._localize_reconstruction_exists = False
        self._localize_reconstruction_summary = None
        self._localize_available_atlases = ()
        self._localize_region_names_by_atlas = {}
        self._localize_selected_atlas = None
        self._localize_selected_regions = ()
        if self._space_value_edit is not None:
            self._space_value_edit.setText("")
        if self._localize_elmodel_edit is not None:
            self._localize_elmodel_edit.setText("")

        if has_subject_context:
            assert self._current_project is not None
            assert self._current_subject is not None
            space, space_message = self._infer_subject_space_runtime(
                self._current_project,
                self._current_subject,
            )
            self._localize_inferred_space = space
            self._localize_space_error = space_message
            if self._space_value_edit is not None:
                if space is not None:
                    self._space_value_edit.setText(space)
                elif space_message:
                    self._space_value_edit.setText(space_message)

            self._localize_reconstruction_exists = self._has_reconstruction_mat_runtime(
                self._current_project,
                self._current_subject,
            )

        atlases: list[str] = []
        if has_subject_context and self._localize_inferred_space is not None:
            atlases = self._discover_atlases_runtime(
                self._localize_paths.leaddbs_dir, self._localize_inferred_space
            )
        self._localize_available_atlases = tuple(atlases)

        self._localize_match_completed = False
        self._localize_match_payload = None
        self._localize_label_channels = ()
        self._localize_user_draft = UserLabels()
        self._localize_label_error = ""
        self._localize_selected_source = None
        if context is not None:
            _, _, self._localize_label_channels = self._load_record_channels_for_match()
            self._load_user_label_draft()
            ok, settings, _ = self._load_record_params_payload(context)
            if ok:
                self._localize_selected_atlas, self._localize_selected_regions = (
                    self._localize_atlas_config(settings)
                )
                match = _nested_get(settings, ("localize", "match"))
                if isinstance(match, dict):
                    self._localize_match_payload = match
                    self._localize_match_completed = bool(match.get("completed"))
                self._localize_selected_source = _nested_get(
                    settings, ("localize", "selected_source")
                )
        self._refresh_localize_match_status()
        self._refresh_localize_atlas_summary()
        self._refresh_localize_action_state()
        self._refresh_localize_matlab_status()

    def _load_user_label_draft(self) -> None:
        path = (
            localize_record_dir(
                self._current_project, self._current_subject, self._current_record
            )
            / USER_LABELS
            / "input.csv"
        )
        self._localize_label_error = ""
        try:
            self._localize_user_draft = (
                read_labels(path, self._localize_label_channels)
                if path.is_file()
                else UserLabels()
            )
        except (OSError, ValueError, csv.Error) as exc:
            self._localize_user_draft = UserLabels()
            self._localize_label_error = str(exc)

    def _on_localize_user_labels(self) -> None:
        from lfptensorpipe.gui.dialogs.user_labels import UserLabelsDialog

        context = self._record_context()
        if context is None or not self._localize_label_channels:
            return
        read_only = (
            self._demo_data_source_readonly is not None
            and self._demo_data_source_readonly.resolve()
            == context.project_root.resolve()
        )
        self._load_user_label_draft()
        if self._localize_label_error:
            self._show_warning("User Labels", self._localize_label_error)
            return
        channels = self._localize_label_channels
        dialog = UserLabelsDialog(
            context,
            channels,
            self._localize_user_draft,
            read_only=read_only,
            parent=self,
        )
        if (
            dialog.exec() != QDialog.Accepted
            or read_only
            or self._record_context() != context
        ):
            return
        draft = dialog.labels()
        dest = (
            localize_record_dir(context.project_root, context.subject, context.record)
            / USER_LABELS
            / "input.csv"
        )
        try:
            save_labels(dest, draft, channels)
        except OSError as exc:
            self._show_warning("User Labels", str(exc))
            return
        self._localize_user_draft = draft
        self._refresh_localize_action_state()
        self.statusBar().showMessage(
            "User Labels saved. Apply to use the updated table."
        )

    def _on_localize_source_changed(self, index: int) -> None:
        context = self._record_context()
        if (
            context is None
            or index < 0
            or (
                self._demo_data_source_readonly is not None
                and context.project_root.resolve()
                == self._demo_data_source_readonly.resolve()
            )
        ):
            return
        ok, settings, _ = self._load_record_params_payload(context)
        if not ok:
            return
        selected = self._localize_source_combo.itemData(index)
        _nested_set(settings, ("localize", "selected_source"), selected)
        if self._write_record_params_payload(
            context, params=settings, reason="localize_source"
        ):
            self._localize_selected_source = selected
        self._refresh_localize_action_state()

    def _refresh_localize_action_state(self) -> None:
        combo = self._localize_source_combo
        if combo is None:
            return
        context = self._record_context()
        writable = context is not None and (
            self._demo_data_source_readonly is None
            or context.project_root.resolve()
            != self._demo_data_source_readonly.resolve()
        )
        channels = self._localize_label_channels
        space = self._localize_inferred_space
        atlas = self._localize_selected_atlas
        regions = self._localize_selected_regions
        mapping = localize_match_signature(self._localize_match_payload)
        configured_paths, path_message = self._can_open_contact_viewer_runtime(
            self._localize_paths
        )
        matched = bool(
            channels and mapping and {r["channel"] for r in mapping} == set(channels)
        )
        can_match = bool(
            writable
            and channels
            and configured_paths
            and self._localize_reconstruction_exists
        )
        can_atlas = bool(writable and space and self._localize_available_atlases)
        draft = self._localize_user_draft
        user_ready = bool(
            writable
            and channels
            and draft.regions
            and draft.assessed
            and not self._localize_label_error
        )
        atlas_path = self._localize_atlas_path(atlas) if atlas else None
        atlas_index_ready = bool(
            atlas_path and (atlas_path / "atlas_index.mat").is_file()
        )
        lead_ready = bool(
            can_match and can_atlas and matched and atlas_index_ready and regions
        )
        reasons = []
        if not writable:
            reasons.append("Select a writable record.")
        if not configured_paths:
            reasons.append(path_message)
        if not self._localize_reconstruction_exists:
            reasons.append("Missing subject reconstruction.")
        if not space:
            reasons.append(self._localize_space_error or "No normalization space.")
        if not matched:
            reasons.append("Complete Match for every record channel.")
        if not atlas_index_ready or not regions:
            reasons.append("Save a compatible atlas with at least one region.")
        lead_reason = " ".join(reasons)
        options = {LEAD_DBS: lead_ready, USER_LABELS: user_ready}
        root = (
            localize_record_dir(context.project_root, context.subject, context.record)
            if context
            else None
        )
        log_error = ""
        try:
            payload = read_run_log(root / "lfptensorpipe_log.json") if root else None
        except (OSError, ValueError) as exc:
            payload = None
            log_error = f"Cannot read the applied Localize result: {exc}"
        active = applied_source(payload)
        receipts = source_results(payload)
        selected = self._localize_selected_source or active
        if selected is None:
            selected = next(
                (name for name, enabled in options.items() if enabled), None
            )
        combo.blockSignals(True)
        for i in range(combo.count()):
            name = combo.itemData(i)
            item = combo.model().item(i)
            item.setEnabled(options[name])
            why = (
                lead_reason
                if name == LEAD_DBS
                else (
                    self._localize_label_error
                    or "Save at least one assessed channel and region."
                )
            )
            item.setToolTip("" if options[name] else why)
        combo.setCurrentIndex(
            combo.findData(selected) if selected and options.get(selected) else -1
        )
        combo.blockSignals(False)
        combo.setEnabled(any(options.values()))
        user_reason = (
            self._localize_label_error
            or "Save at least one assessed channel and region."
        )
        combo.setToolTip(
            "\n".join(
                filter(
                    None,
                    (
                        "Choose the source for the next Apply.",
                        f"Lead-DBS: {lead_reason}" if not lead_ready else "",
                        f"User Labels: {user_reason}" if not user_ready else "",
                    ),
                )
            )
        )
        names = {LEAD_DBS: "Lead-DBS", USER_LABELS: "User Labels"}
        active_ready = bool(active and source_files_ready(root, active))
        active_text = names.get(active, "—")
        compact_name = {LEAD_DBS: "Lead-DBS", USER_LABELS: "User"}.get(active, "—")
        if active and not active_ready:
            active_text += " (not ready)"
            compact_name += " ⚠"
        self._localize_active_label.setText(f"Active: {compact_name}")
        self._localize_active_label.setToolTip(f"Applied label source: {active_text}")
        self._localize_active_label.setAccessibleName(f"Applied label source: {active_text}")
        self._localize_labels_summary.setText(
            f"{len(draft.assessed)}/{len(channels)} assessed"
        )
        self._localize_labels_button.setEnabled(
            bool(
                root
                and channels
                and (writable or (root / USER_LABELS / "input.csv").is_file())
            )
        )
        self._localize_match_button.setEnabled(can_match)
        self._localize_match_button.setToolTip(
            "Map record channels to contacts." if can_match else lead_reason
        )
        self._localize_atlas_button.setEnabled(can_atlas)
        self._localize_atlas_button.setToolTip(
            "Choose an atlas and regions." if can_atlas else lead_reason
        )
        self._localize_import_button.setEnabled(bool(can_match and can_atlas))
        self._localize_export_button.setEnabled(
            bool(
                configured_paths
                and matched
                and atlas
                and regions
                and self._localize_reconstruction_exists
            )
        )
        self._localize_configs_button.setEnabled(
            self._localize_import_button.isEnabled()
            or self._localize_export_button.isEnabled()
        )
        lead_receipt = receipts.get(LEAD_DBS, {})
        geometry_csv = root / "channel_representative_coords.csv" if root else None
        viewer_atlas = lead_receipt.get("atlas")
        viewer_space = lead_receipt.get("space")
        self._contact_viewer_button.setEnabled(
            bool(
                configured_paths
                and viewer_atlas
                and viewer_space
                and (
                    self._localize_paths.leaddbs_dir
                    / "templates"
                    / "space"
                    / viewer_space
                    / "atlases"
                    / viewer_atlas
                ).is_dir()
                and geometry_csv
                and geometry_csv.is_file()
            )
        )
        selected_ready = options.get(selected, False) and not log_error
        saved = receipts.get(selected, {})
        reusable = False
        if selected_ready and saved and source_files_ready(root, selected):
            if selected == USER_LABELS:
                reusable = saved.get("label_signature") == draft.signature(channels)
            elif (
                saved.get("space") == space
                and saved.get("atlas") == atlas
                and saved.get("match_signature") == mapping
                and saved.get("selected_regions_signature")
                == localize_selected_regions_signature(regions)
            ):
                try:
                    _, _, same_geometry, same_masks = compare_lead_inputs(
                        saved,
                        context.project_root,
                        context.subject,
                        space,
                        atlas,
                        localize_selected_regions_signature(regions),
                        self._localize_paths,
                        mapping,
                        preview=True,
                    )
                    reusable = same_geometry and same_masks
                except (OSError, ValueError, KeyError, RuntimeError) as exc:
                    selected_ready = False
                    lead_reason = str(exc)
        current = reusable and active == selected
        self._localize_apply_button.setEnabled(bool(selected_ready and not current))
        if current:
            tip = "Up to date."
        elif reusable:
            tip = "Use existing " + names[selected] + " result."
        elif selected_ready:
            tip = "Apply saved " + names[selected] + " settings."
        else:
            tip = lead_reason if selected == LEAD_DBS else "Select an available source."
        self._localize_apply_button.setToolTip(log_error or tip)
        self._localize_apply_button.setText("Up to date" if current else "Apply")
        failed = payload is not None and not payload.get("completed")
        state = (
            "green"
            if current
            else "yellow"
            if active or selected_ready or failed
            else "gray"
        )
        if log_error or (failed and not current):
            state = "yellow"
        self._set_indicator_color(self._localize_indicator, state)
        self._localize_indicator.setToolTip(f"{tip} Active: {active_text}.")
