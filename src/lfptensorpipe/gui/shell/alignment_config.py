"""Alignment config import/export MainWindow methods."""

from __future__ import annotations

import json

from lfptensorpipe.gui.shell.common import (
    Any,
    Path,
    PathResolver,
    RecordContext,
)

ALIGNMENT_CONFIG_SCHEMA = "lfptensorpipe.alignment-config"
ALIGNMENT_CONFIG_VERSION = 1
ALIGNMENT_CONFIG_FILE_NAME = "lfptensorpipe_alignment_config.json"


class MainWindowAlignmentConfigMixin:
    @staticmethod
    def _alignment_config_json_value(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        item_method = getattr(value, "item", None)
        if callable(item_method):
            try:
                return MainWindowAlignmentConfigMixin._alignment_config_json_value(
                    item_method()
                )
            except Exception:
                pass
        if isinstance(value, dict):
            return {
                str(key): MainWindowAlignmentConfigMixin._alignment_config_json_value(
                    item
                )
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [
                MainWindowAlignmentConfigMixin._alignment_config_json_value(item)
                for item in value
            ]
        raise TypeError(f"Unsupported alignment config value: {type(value).__name__}")

    @staticmethod
    def _alignment_config_normalize_labels(
        annotation_labels: list[str] | tuple[str, ...],
    ) -> tuple[str, ...]:
        out: list[str] = []
        seen: set[str] = set()
        for item in annotation_labels:
            label = str(item).strip()
            if not label or label in seen:
                continue
            seen.add(label)
            out.append(label)
        return tuple(out)

    @staticmethod
    def _alignment_config_slug_token(slug: str) -> str:
        token = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in str(slug).strip()
        )
        return token.strip("-_")

    def _alignment_config_default_path(
        self,
        context: RecordContext,
        *,
        trial_slug: str,
    ) -> Path:
        token = self._alignment_config_slug_token(trial_slug)
        filename = (
            ALIGNMENT_CONFIG_FILE_NAME
            if not token
            else f"lfptensorpipe_alignment_{token}_config.json"
        )
        return PathResolver(context).lfp_root / filename

    def _build_alignment_config_export_payload(self) -> dict[str, Any]:
        paradigm = self._current_alignment_paradigm()
        if paradigm is None:
            raise ValueError("Select one trial first.")

        method_key = str(paradigm.get("method", "")).strip()
        if self._alignment_method_combo is not None:
            method_data = self._alignment_method_combo.currentData()
            if isinstance(method_data, str) and method_data.strip():
                method_key = method_data.strip()

        params = paradigm.get("method_params", {})
        if not isinstance(params, dict):
            params = {}
        ok, normalized_params, message = self._validate_alignment_method_params_runtime(
            method_key,
            params,
            annotation_labels=[],
        )
        if not ok:
            raise ValueError(message)
        return {
            "schema": ALIGNMENT_CONFIG_SCHEMA,
            "version": ALIGNMENT_CONFIG_VERSION,
            "alignment": {
                "method": method_key,
                "method_params": self._alignment_config_json_value(normalized_params),
            },
        }

    def _filter_alignment_config_import_params(
        self,
        method_key: str,
        params_node: dict[str, Any],
        *,
        annotation_labels: list[str] | tuple[str, ...],
    ) -> tuple[dict[str, Any], list[str]]:
        normalized_labels = self._alignment_config_normalize_labels(annotation_labels)
        if not normalized_labels:
            return dict(params_node), []

        available_labels = set(normalized_labels)
        filtered_params = dict(params_node)
        warnings: list[str] = []

        if method_key in {"pad_warper", "stack_warper", "concat_warper"}:
            raw_annotations = params_node.get("annotations")
            if isinstance(raw_annotations, list):
                kept: list[str] = []
                seen: set[str] = set()
                dropped = 0
                for item in raw_annotations:
                    label = str(item).strip()
                    if not label or label in seen:
                        continue
                    seen.add(label)
                    if label in available_labels:
                        kept.append(label)
                    else:
                        dropped += 1
                filtered_params["annotations"] = kept
                if dropped > 0:
                    warnings.append(
                        f"Ignored {dropped} unavailable annotation label(s)."
                    )

        if method_key == "linear_warper":
            raw_anchors = params_node.get("anchors_percent")
            if isinstance(raw_anchors, dict):
                kept_anchors: dict[str, str] = {}
                dropped = 0
                for raw_percent, raw_label in raw_anchors.items():
                    label = str(raw_label).strip()
                    if label and label in available_labels:
                        kept_anchors[str(raw_percent)] = label
                    else:
                        dropped += 1
                filtered_params["anchors_percent"] = kept_anchors
                if dropped > 0:
                    warnings.append(f"Ignored {dropped} unavailable anchor label(s).")

        return filtered_params, warnings

    def _normalize_alignment_config_import_payload(
        self,
        payload: dict[str, Any],
        *,
        annotation_labels: list[str] | tuple[str, ...],
    ) -> tuple[dict[str, Any], list[str]]:
        if not isinstance(payload, dict):
            raise ValueError("Alignment config must be a JSON object.")
        if payload.get("schema") != ALIGNMENT_CONFIG_SCHEMA:
            raise ValueError(
                f"Unsupported alignment config schema: {payload.get('schema')!r}."
            )
        if payload.get("version") != ALIGNMENT_CONFIG_VERSION:
            raise ValueError(
                f"Unsupported alignment config version: {payload.get('version')!r}."
            )

        alignment_node = payload.get("alignment")
        if not isinstance(alignment_node, dict):
            raise ValueError("Alignment config is missing required `alignment` object.")

        method_key = str(alignment_node.get("method", "")).strip()
        if not method_key:
            raise ValueError("Alignment config is missing required `alignment.method`.")

        method_params = alignment_node.get("method_params")
        if not isinstance(method_params, dict):
            raise ValueError(
                "Alignment config is missing required `alignment.method_params` object."
            )

        filtered_params, warnings = self._filter_alignment_config_import_params(
            method_key,
            method_params,
            annotation_labels=annotation_labels,
        )
        ok, normalized_params, message = self._validate_alignment_method_params_runtime(
            method_key,
            filtered_params,
            annotation_labels=[],
        )
        if not ok:
            raise ValueError(message)
        return (
            {
                "method": method_key,
                "method_params": normalized_params,
            },
            warnings,
        )

    def _on_alignment_export_config(self) -> None:
        context = self._record_context()
        slug = self._current_alignment_paradigm_slug()
        if (
            context is None
            or slug is None
            or self._current_alignment_paradigm() is None
        ):
            self._show_warning(
                "Export Configs",
                "Select project, subject, record, and one trial before exporting Align Epochs configs.",
            )
            return

        default_path = self._alignment_config_default_path(context, trial_slug=slug)
        file_path_text, _ = self._save_file_name(
            "Export Align Epochs Configs",
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
            payload = self._build_alignment_config_export_payload()
            with export_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        except Exception as exc:  # noqa: BLE001
            self._show_warning("Export Configs", f"Export failed:\n{exc}")
            return

        self.statusBar().showMessage(
            f"Exported Align Epochs config: {export_path.name}"
        )

    def _on_alignment_import_config(self) -> None:
        context = self._record_context()
        slug = self._current_alignment_paradigm_slug()
        if (
            context is None
            or slug is None
            or self._current_alignment_paradigm() is None
        ):
            self._show_warning(
                "Import Configs",
                "Select project, subject, record, and one trial before importing Align Epochs configs.",
            )
            return

        default_path = self._alignment_config_default_path(context, trial_slug=slug)
        file_path_text, _ = self._open_file_name(
            "Import Align Epochs Configs",
            str(default_path.parent.resolve()),
            "JSON files (*.json);;All files (*)",
        )
        if not file_path_text:
            return

        import_path = Path(file_path_text)
        annotation_labels = self._load_alignment_annotation_labels_runtime(context)
        try:
            with import_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            alignment_snapshot, warnings = (
                self._normalize_alignment_config_import_payload(
                    payload,
                    annotation_labels=annotation_labels,
                )
            )
        except Exception as exc:  # noqa: BLE001
            self._show_warning("Import Configs", f"Import failed:\n{exc}")
            return

        ok, message = self._update_alignment_paradigm_runtime(
            self._config_store,
            slug=slug,
            method=alignment_snapshot["method"],
            method_params=alignment_snapshot["method_params"],
            context=context,
        )
        if not ok:
            self._show_warning("Import Configs", message)
            return

        self._reload_alignment_paradigms(preferred_slug=slug)
        persisted = self._persist_record_params_snapshot(
            reason="alignment_import_config"
        )
        self.statusBar().showMessage(
            f"Imported Align Epochs config: {import_path.name}"
        )
        if warnings:
            self._show_information(
                "Import Configs",
                "Align Epochs config imported with warnings:\n- "
                + "\n- ".join(warnings),
            )
        if not persisted:
            self._show_warning(
                "Import Configs",
                "Align Epochs config imported, but persisting record UI state failed.",
            )


__all__ = [
    "ALIGNMENT_CONFIG_FILE_NAME",
    "ALIGNMENT_CONFIG_SCHEMA",
    "ALIGNMENT_CONFIG_VERSION",
    "MainWindowAlignmentConfigMixin",
]
