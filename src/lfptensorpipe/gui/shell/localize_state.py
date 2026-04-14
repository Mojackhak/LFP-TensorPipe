"""Localize state and control MainWindow methods."""

from __future__ import annotations

from lfptensorpipe.gui.shell.common import (
    Any,
    _nested_get,
    _nested_set,
    localize_representative_csv_path,
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

    def _load_localize_atlas_config(self) -> tuple[str | None, tuple[str, ...]]:
        context = self._record_context()
        if context is None:
            return None, ()
        ok, params, _ = self._load_record_params_payload(context)
        if not ok:
            return None, ()
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
        payload = (
            dict(self._localize_match_payload)
            if isinstance(self._localize_match_payload, dict)
            else self._load_localize_match_payload()
        )
        if isinstance(payload, dict):
            self._localize_match_payload = dict(payload)
        mapped = 0
        total = 0
        if isinstance(payload, dict):
            mappings = payload.get("mappings")
            channels = payload.get("channels")
            if isinstance(mappings, list):
                mapped = sum(1 for item in mappings if isinstance(item, dict))
            if isinstance(channels, list):
                total = len(channels)
            if total <= 0:
                total = mapped
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
        self._localize_atlas_summary_label.setText(
            f"{selected}/{total} regions selected"
        )
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

        if (
            context is not None
            and self._current_project is not None
            and self._current_subject is not None
            and self._current_record is not None
        ):
            atlas_name, selected_regions = self._load_localize_atlas_config()
            self._localize_selected_atlas = atlas_name
            self._localize_selected_regions = selected_regions

        self._localize_match_completed = False
        self._localize_match_payload = None
        if (
            context is not None
            and self._current_project is not None
            and self._current_subject is not None
            and self._current_record is not None
        ):
            payload = self._load_localize_match_payload()
            if isinstance(payload, dict):
                self._localize_match_payload = dict(payload)
                self._localize_match_completed = bool(payload.get("completed", False))
            self._refresh_localize_match_status()
        else:
            self._refresh_localize_match_status()

        self._refresh_localize_atlas_summary()
        self._refresh_localize_action_state()
        self._refresh_localize_matlab_status()

    def _refresh_localize_action_state(self) -> None:
        context = self._record_context()
        has_context = context is not None
        selected_space = self._localize_inferred_space
        selected_atlas = self._localize_selected_atlas
        selected_regions = self._localize_selected_regions
        read_only_project = (
            self._current_project is not None
            and self._demo_data_source_readonly is not None
            and self._current_project.resolve()
            == self._demo_data_source_readonly.resolve()
        )
        can_apply = bool(
            has_context
            and selected_space
            and selected_atlas
            and selected_regions
            and not read_only_project
            and self._localize_reconstruction_exists
            and not self._localize_space_error
            and self._localize_match_completed
        )
        can_match = bool(
            has_context
            and self._localize_reconstruction_exists
            and not read_only_project
        )
        can_atlas = bool(
            has_context
            and selected_space
            and self._localize_available_atlases
            and not read_only_project
        )
        can_import = bool(can_atlas and self._localize_reconstruction_exists)
        can_export = bool(
            has_context
            and selected_space
            and selected_atlas
            and selected_regions
            and self._localize_match_completed
            and self._localize_reconstruction_exists
        )
        can_viewer_paths, viewer_paths_message = self._can_open_contact_viewer_runtime(
            self._localize_paths
        )
        can_viewer = bool(can_viewer_paths and has_context and selected_atlas)

        apply_tooltip = "Generate representative localize artifacts for current record."
        if not can_apply:
            apply_reasons: list[str] = []
            if not has_context:
                apply_reasons.append("Select project, subject, and record.")
            if self._localize_space_error:
                apply_reasons.append(self._localize_space_error)
            if not selected_space and not self._localize_space_error:
                apply_reasons.append("No inferred subject space.")
            if not selected_atlas:
                apply_reasons.append("Save atlas config first.")
            if selected_atlas and not selected_regions:
                apply_reasons.append("Select at least one interested region.")
            if read_only_project:
                apply_reasons.append("Read-only demo project.")
            if not self._localize_reconstruction_exists:
                apply_reasons.append("Missing reconstruction.mat.")
            if not self._localize_match_completed:
                apply_reasons.append("Complete channel mapping in Match Configure.")
            if apply_reasons:
                apply_tooltip = (
                    f"{apply_tooltip} Unavailable: {' '.join(apply_reasons)}"
                )

        match_tooltip = "Map each record channel to Lead-DBS contacts for this record."
        if not can_match:
            match_reasons: list[str] = []
            if not has_context:
                match_reasons.append("Select project, subject, and record.")
            if read_only_project:
                match_reasons.append("Read-only demo project.")
            if not self._localize_reconstruction_exists:
                match_reasons.append("Missing reconstruction.mat.")
            if match_reasons:
                match_tooltip = (
                    f"{match_tooltip} Unavailable: {' '.join(match_reasons)}"
                )

        atlas_tooltip = "Choose the atlas and interested regions for this record."
        if not can_atlas:
            atlas_reasons: list[str] = []
            if not has_context:
                atlas_reasons.append("Select project, subject, and record.")
            if self._localize_space_error:
                atlas_reasons.append(self._localize_space_error)
            if not selected_space and not self._localize_space_error:
                atlas_reasons.append("No inferred subject space.")
            if not self._localize_available_atlases:
                atlas_reasons.append("No atlas choices available.")
            if read_only_project:
                atlas_reasons.append("Read-only demo project.")
            if atlas_reasons:
                atlas_tooltip = (
                    f"{atlas_tooltip} Unavailable: {' '.join(atlas_reasons)}"
                )

        import_tooltip = (
            "Import Localize atlas and match config for the current record."
        )
        if not can_import:
            import_reasons: list[str] = []
            if not has_context:
                import_reasons.append("Select project, subject, and record.")
            if self._localize_space_error:
                import_reasons.append(self._localize_space_error)
            if not self._localize_reconstruction_exists:
                import_reasons.append("Missing reconstruction.mat.")
            if read_only_project:
                import_reasons.append("Read-only demo project.")
            if import_reasons:
                import_tooltip = (
                    f"{import_tooltip} Unavailable: {' '.join(import_reasons)}"
                )

        export_tooltip = (
            "Export Localize atlas and match config for the current record."
        )
        if not can_export:
            export_reasons: list[str] = []
            if not has_context:
                export_reasons.append("Select project, subject, and record.")
            if not selected_atlas:
                export_reasons.append("Save atlas config first.")
            if selected_atlas and not selected_regions:
                export_reasons.append("Select at least one interested region.")
            if not self._localize_match_completed:
                export_reasons.append("Complete channel mapping in Match Configure.")
            if not self._localize_reconstruction_exists:
                export_reasons.append("Missing reconstruction.mat.")
            if export_reasons:
                export_tooltip = (
                    f"{export_tooltip} Unavailable: {' '.join(export_reasons)}"
                )

        viewer_tooltip = (
            "Open Contact Viewer with current atlas and representative CSV."
        )
        if not can_viewer:
            viewer_reasons: list[str] = []
            if not has_context:
                viewer_reasons.append("Select project, subject, and record.")
            if not selected_atlas:
                viewer_reasons.append("Save atlas config first.")
            if not can_viewer_paths and viewer_paths_message:
                viewer_reasons.append(viewer_paths_message)
            if viewer_reasons:
                viewer_tooltip = (
                    f"{viewer_tooltip} Unavailable: {' '.join(viewer_reasons)}"
                )
        elif (
            self._current_project is not None
            and self._current_subject is not None
            and self._current_record is not None
        ):
            viewer_csv_path = localize_representative_csv_path(
                self._current_project,
                self._current_subject,
                self._current_record,
            )
            if not viewer_csv_path.is_file():
                viewer_tooltip = (
                    f"{viewer_tooltip} "
                    "Representative CSV not found yet; run Apply first."
                )

        if self._localize_apply_button is not None:
            self._localize_apply_button.setEnabled(can_apply)
            self._localize_apply_button.setToolTip(apply_tooltip)
        if self._localize_match_button is not None:
            self._localize_match_button.setEnabled(can_match)
            self._localize_match_button.setToolTip(match_tooltip)
        if self._localize_atlas_button is not None:
            self._localize_atlas_button.setEnabled(can_atlas)
            self._localize_atlas_button.setToolTip(atlas_tooltip)
        if self._localize_import_button is not None:
            self._localize_import_button.setEnabled(can_import)
            self._localize_import_button.setToolTip(import_tooltip)
        if self._localize_export_button is not None:
            self._localize_export_button.setEnabled(can_export)
            self._localize_export_button.setToolTip(export_tooltip)
        if self._contact_viewer_button is not None:
            self._contact_viewer_button.setEnabled(can_viewer)
            self._contact_viewer_button.setToolTip(viewer_tooltip)

        if self._localize_indicator is not None:
            if has_context:
                assert context is not None
                state = self._localize_panel_state_runtime(
                    context.project_root,
                    context.subject,
                    context.record,
                    atlas=selected_atlas,
                    selected_regions=selected_regions,
                    match_payload=self._localize_match_payload,
                )
            else:
                state = "gray"
            self._set_indicator_color(self._localize_indicator, state)
            state_label = {
                "gray": "not run",
                "yellow": "failed or stale",
                "green": "current draft matches last apply",
            }.get(state, state)
            self._localize_indicator.setToolTip(
                "Localize panel state: gray=not run, yellow=failed or draft differs, "
                f"green=current draft matches last apply. Current: {state_label}."
            )
