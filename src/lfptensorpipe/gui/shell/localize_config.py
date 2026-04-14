"""Localize atlas-config dialog and import/export MainWindow methods."""

from __future__ import annotations

import json

from lfptensorpipe.gui.dialogs.localize_match_actions import build_lead_signature
from lfptensorpipe.gui.shell.common import (
    Any,
    Path,
    PathResolver,
    QDialog,
    RecordContext,
)

LOCALIZE_CONFIG_SCHEMA = "lfptensorpipe.localize-config"
LOCALIZE_CONFIG_VERSION = 1
LOCALIZE_CONFIG_FILE_NAME = "lfptensorpipe_localize_config.json"


class MainWindowLocalizeConfigMixin:
    def _localize_config_default_path(self, context: RecordContext) -> Path:
        return PathResolver(context).lfp_root / LOCALIZE_CONFIG_FILE_NAME

    def _normalize_localize_lead_signature(self, payload: Any) -> list[dict[str, Any]]:
        if not isinstance(payload, list):
            raise ValueError(
                "Localize config is missing required `lead_signature` list."
            )
        normalized = build_lead_signature(
            [item for item in payload if isinstance(item, dict)]
        )
        if not normalized:
            raise ValueError("Localize config `lead_signature` is empty.")
        return normalized

    def _normalize_localize_match_payload(
        self,
        payload: Any,
        *,
        expected_channels: tuple[str, ...],
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("Localize config is missing required `match` object.")
        channels_raw = payload.get("channels")
        mappings_raw = payload.get("mappings")
        if not isinstance(channels_raw, list):
            raise ValueError(
                "Localize config match is missing required `channels` list."
            )
        if not isinstance(mappings_raw, list):
            raise ValueError(
                "Localize config match is missing required `mappings` list."
            )
        channels = tuple(str(item).strip() for item in channels_raw)
        if channels != expected_channels:
            raise ValueError(
                "Imported Localize match channels do not match the current record."
            )

        rows: list[dict[str, str]] = []
        seen_channels: set[str] = set()
        for item in mappings_raw:
            if not isinstance(item, dict):
                continue
            channel = str(item.get("channel", "")).strip()
            anode = str(item.get("anode", "")).strip()
            cathode = str(item.get("cathode", "")).strip()
            rep_coord = str(item.get("rep_coord", "Mid")).strip().title()
            if channel not in expected_channels:
                continue
            if not anode or not cathode:
                continue
            if rep_coord not in {"Anode", "Cathode", "Mid"}:
                rep_coord = "Mid"
            if channel in seen_channels:
                raise ValueError(
                    f"Duplicate Localize match mapping for channel `{channel}`."
                )
            seen_channels.add(channel)
            rows.append(
                {
                    "channel": channel,
                    "anode": anode,
                    "cathode": cathode,
                    "rep_coord": rep_coord,
                }
            )

        if len(rows) != len(expected_channels):
            raise ValueError(
                "Imported Localize match must fully map every current channel."
            )

        rows.sort(key=lambda row: expected_channels.index(row["channel"]))
        return {
            "completed": True,
            "channels": list(expected_channels),
            "mapped_count": len(rows),
            "mappings": rows,
        }

    def _resolve_localize_dialog_seed(self) -> tuple[str, tuple[str, ...]]:
        atlas_names = self._localize_available_atlases
        if not atlas_names:
            return "", ()

        current_atlas = self._localize_selected_atlas
        if current_atlas in atlas_names:
            selected_regions = self._localize_selected_regions
            return current_atlas, selected_regions

        defaults = self._load_localize_defaults().get(
            self._localize_inferred_space or ""
        )
        default_atlas = (
            str(defaults.get("atlas", "")).strip() if isinstance(defaults, dict) else ""
        )
        if default_atlas not in atlas_names:
            default_atlas = atlas_names[0]
        default_regions = (
            defaults.get("selected_regions", []) if isinstance(defaults, dict) else []
        )
        normalized_regions = self._normalize_localize_selected_regions(
            default_atlas,
            default_regions,
        )
        if not normalized_regions:
            normalized_regions = self._localize_region_names_for_atlas(default_atlas)
        return default_atlas, normalized_regions

    def _load_current_localize_lead_signature(
        self,
        *,
        action_label: str,
    ) -> tuple[bool, str, list[dict[str, Any]]]:
        if self._current_project is None or self._current_subject is None:
            return False, "Select project and subject first.", []
        active_context = (
            str(self._current_project),
            self._current_subject,
            self._current_record,
        )
        ok_summary, message_summary, summary = self._run_with_busy(
            action_label,
            lambda: self._load_reconstruction_contacts_runtime(
                self._current_project,
                self._current_subject,
                self._localize_paths,
            ),
        )
        if (
            self._current_project is None
            or self._current_subject is None
            or self._current_record is None
        ):
            return False, "Stale Localize context.", []
        current_context = (
            str(self._current_project),
            self._current_subject,
            self._current_record,
        )
        if active_context != current_context:
            return False, "Stale Localize context.", []
        if not ok_summary:
            return False, message_summary, []
        if not isinstance(summary, dict):
            return False, "Invalid reconstruction summary.", []
        self._localize_reconstruction_summary = summary
        return (
            True,
            "",
            build_lead_signature(
                [item for item in summary.get("leads", []) if isinstance(item, dict)]
            ),
        )

    def _on_localize_atlas_configure(self) -> None:
        if (
            self._current_project is None
            or self._current_subject is None
            or self._current_record is None
        ):
            self.statusBar().showMessage(
                "Atlas Configure unavailable: select project/subject/record."
            )
            return
        if (
            self._demo_data_source_readonly is not None
            and self._current_project.resolve()
            == self._demo_data_source_readonly.resolve()
        ):
            self.statusBar().showMessage(
                "Atlas Configure unavailable: read-only demo project."
            )
            return
        if self._localize_space_error:
            self.statusBar().showMessage(
                f"Atlas Configure unavailable: {self._localize_space_error}"
            )
            return
        if not self._localize_inferred_space:
            self.statusBar().showMessage(
                "Atlas Configure unavailable: no inferred subject space."
            )
            return
        if not self._localize_available_atlases:
            self.statusBar().showMessage(
                "Atlas Configure unavailable: no atlas choices available."
            )
            return

        current_atlas, current_regions = self._resolve_localize_dialog_seed()
        dialog = self._create_localize_atlas_dialog(
            space=self._localize_inferred_space,
            atlas_names=self._localize_available_atlases,
            current_atlas=current_atlas,
            current_selected_regions=current_regions,
            region_loader=self._localize_region_names_for_atlas,
            config_store=self._config_store,
            parent=self,
        )
        if dialog.exec() != QDialog.Accepted or dialog.selected_payload is None:
            return
        payload = dict(dialog.selected_payload)
        atlas = str(payload.get("atlas", "")).strip()
        selected_regions = payload.get("selected_regions", [])
        if not atlas:
            return
        if not self._save_localize_atlas_config(
            atlas=atlas,
            selected_regions=selected_regions,
        ):
            self._show_warning(
                "Localize Atlas",
                "Saving Localize atlas config failed.",
            )
            return
        self._persist_record_params_snapshot(reason="localize_atlas_save")
        self._refresh_localize_atlas_summary()
        self._refresh_localize_action_state()
        self.statusBar().showMessage("Localize atlas config saved.")

    def _build_localize_export_payload(self) -> dict[str, Any]:
        context = self._record_context()
        if context is None:
            raise ValueError("Select project, subject, and record first.")
        atlas = self._localize_selected_atlas
        if not atlas:
            raise ValueError("Save Localize atlas config before exporting.")
        if not self._localize_selected_regions:
            raise ValueError("Select at least one Localize region before exporting.")
        if (
            not isinstance(self._localize_match_payload, dict)
            or not self._localize_match_completed
        ):
            raise ValueError(
                "Complete Match Configure before exporting Localize config."
            )
        ok_channels, message_channels, channels = self._load_record_channels_for_match()
        if not ok_channels:
            raise ValueError(message_channels)
        ok_signature, message_signature, lead_signature = (
            self._load_current_localize_lead_signature(
                action_label="Export Localize Configs"
            )
        )
        if not ok_signature:
            raise ValueError(message_signature)
        match_payload = self._normalize_localize_match_payload(
            self._localize_match_payload,
            expected_channels=channels,
        )
        return {
            "schema": LOCALIZE_CONFIG_SCHEMA,
            "version": LOCALIZE_CONFIG_VERSION,
            "localize": {
                "atlas": atlas,
                "selected_regions": list(self._localize_selected_regions),
                "match": match_payload,
                "lead_signature": lead_signature,
            },
        }

    def _normalize_localize_import_payload(
        self,
    ) -> tuple[str, tuple[str, ...], dict[str, Any]]:
        context = self._record_context()
        if context is None:
            raise ValueError("Select project, subject, and record first.")
        default_path = self._localize_config_default_path(context)
        file_path_text, _ = self._open_file_name(
            "Import Localize Configs",
            str(default_path.parent.resolve()),
            "JSON files (*.json);;All files (*)",
        )
        if not file_path_text:
            raise ValueError("")
        import_path = Path(file_path_text)
        with import_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("Localize config must be a JSON object.")
        if payload.get("schema") != LOCALIZE_CONFIG_SCHEMA:
            raise ValueError(
                f"Unsupported Localize config schema: {payload.get('schema')!r}."
            )
        if payload.get("version") != LOCALIZE_CONFIG_VERSION:
            raise ValueError(
                f"Unsupported Localize config version: {payload.get('version')!r}."
            )
        localize_node = payload.get("localize")
        if not isinstance(localize_node, dict):
            raise ValueError("Localize config is missing required `localize` object.")
        atlas = str(localize_node.get("atlas", "")).strip()
        if atlas not in self._localize_available_atlases:
            raise ValueError(
                "Imported Localize atlas is unavailable for the current space."
            )
        selected_regions = self._normalize_localize_selected_regions(
            atlas,
            localize_node.get("selected_regions"),
        )
        if not selected_regions:
            raise ValueError(
                "Imported Localize config does not select any valid region for the current atlas."
            )
        ok_channels, message_channels, channels = self._load_record_channels_for_match()
        if not ok_channels:
            raise ValueError(message_channels)
        ok_signature, message_signature, current_signature = (
            self._load_current_localize_lead_signature(
                action_label="Import Localize Configs"
            )
        )
        if not ok_signature:
            raise ValueError(message_signature)
        imported_signature = self._normalize_localize_lead_signature(
            localize_node.get("lead_signature")
        )
        if imported_signature != current_signature:
            raise ValueError(
                "Imported Localize config is incompatible with the current reconstruction lead signature."
            )
        match_payload = self._normalize_localize_match_payload(
            localize_node.get("match"),
            expected_channels=channels,
        )
        match_payload.update(
            {
                "subject": self._current_subject,
                "record": self._current_record,
                "space": self._localize_inferred_space,
                "atlas": atlas,
            }
        )
        return atlas, selected_regions, match_payload

    def _on_localize_export_config(self) -> None:
        context = self._record_context()
        if context is None:
            self._show_warning(
                "Export Configs",
                "Select project, subject, and record before exporting Localize configs.",
            )
            return
        default_path = self._localize_config_default_path(context)
        file_path_text, _ = self._save_file_name(
            "Export Localize Configs",
            str(default_path.resolve()),
            "JSON files (*.json);;All files (*)",
        )
        if not file_path_text:
            return
        export_path = Path(file_path_text)
        if not export_path.suffix:
            export_path = export_path.with_suffix(".json")
        export_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            payload = self._build_localize_export_payload()
            with export_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        except Exception as exc:  # noqa: BLE001
            self._show_warning("Export Configs", f"Export failed:\n{exc}")
            return

        self.statusBar().showMessage(f"Exported Localize config: {export_path.name}")

    def _on_localize_import_config(self) -> None:
        if self._record_context() is None:
            self._show_warning(
                "Import Configs",
                "Select project, subject, and record before importing Localize configs.",
            )
            return
        try:
            atlas, selected_regions, match_payload = (
                self._normalize_localize_import_payload()
            )
        except ValueError as exc:
            message = str(exc).strip()
            if message:
                self._show_warning("Import Configs", f"Import failed:\n{message}")
            return
        except Exception as exc:  # noqa: BLE001
            self._show_warning("Import Configs", f"Import failed:\n{exc}")
            return

        if not self._save_localize_draft(
            atlas=atlas,
            selected_regions=selected_regions,
            match_payload=match_payload,
        ):
            self._show_warning(
                "Import Configs",
                "Localize config imported, but saving the current record state failed.",
            )
            return

        self._refresh_localize_match_status()
        self._refresh_localize_atlas_summary()
        self._refresh_localize_action_state()
        self.statusBar().showMessage("Imported Localize config.")


__all__ = [
    "LOCALIZE_CONFIG_FILE_NAME",
    "LOCALIZE_CONFIG_SCHEMA",
    "LOCALIZE_CONFIG_VERSION",
    "MainWindowLocalizeConfigMixin",
]
