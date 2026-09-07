"""Alignment config import/export MainWindow methods."""

from __future__ import annotations

import json

from lfptensorpipe.app.shared.page_config import json_value, page_config_node
from lfptensorpipe.gui.shell.common import (
    Any,
    Path,
    PathResolver,
    QMessageBox,
    RecordContext,
)

ALIGNMENT_CONFIG_SCHEMA = "lfptensorpipe.alignment-config"
ALIGNMENT_CONFIG_VERSION = 1
ALIGNMENT_CONFIG_FILE_NAME = "lfptensorpipe_alignment_config.json"


class MainWindowAlignmentConfigMixin:
    _alignment_config_json_value = staticmethod(json_value)

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
        if method_key != "linear_warper" and not normalized_params.get("annotations"):
            raise ValueError(
                "Select at least one annotation before exporting Align Epochs configs."
            )
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
        alignment_node = page_config_node(payload, "alignment")

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
        defaults = self._default_alignment_method_params_runtime(method_key)
        if not isinstance(defaults, dict):
            raise ValueError(f"Unknown alignment method: {method_key}")
        allowed_keys = set(defaults)
        unknown_keys = sorted(set(filtered_params) - allowed_keys)
        if unknown_keys:
            warnings.append("Removed unknown field(s): " + ", ".join(unknown_keys))
        candidate = {
            key: filtered_params[key] for key in allowed_keys if key in filtered_params
        }
        for key, value in defaults.items():
            if key in candidate:
                continue
            if key == "annotations" and method_key != "linear_warper":
                raise ValueError(
                    "Alignment annotation selection is required and has no safe automatic default."
                )
            candidate[key] = value
            warnings.append(f"Missing default restored: alignment.method_params.{key}.")

        if method_key != "linear_warper" and not candidate.get("annotations"):
            raise ValueError(
                "Alignment annotation selection is required and has no safe automatic default."
            )

        repair_keys_by_message = {
            "sample_rate": ("sample_rate",),
            "sample rate": ("sample_rate",),
            "anchor": ("anchors_percent",),
            "epoch duration": ("epoch_duration_range",),
            "percent_tolerance": ("percent_tolerance",),
            "duration": ("duration_range",),
            "pad_left": ("pad_left",),
            "anno_left": ("anno_left",),
            "anno_right": ("anno_right",),
            "pad_right": ("pad_right",),
            "total window": ("pad_left", "anno_left", "anno_right", "pad_right"),
        }
        normalized_params: dict[str, Any] = {}
        for _ in range(len(candidate) + 1):
            ok, normalized_params, message = (
                self._validate_alignment_method_params_runtime(
                    method_key,
                    candidate,
                    annotation_labels=[],
                )
            )
            if ok:
                break
            lowered = message.lower()
            if "annotation" in lowered:
                raise ValueError(
                    "Alignment annotation selection is invalid and has no safe automatic default."
                )
            repair_keys = next(
                (
                    keys
                    for token, keys in repair_keys_by_message.items()
                    if token in lowered
                ),
                (),
            )
            repair_keys = tuple(key for key in repair_keys if key in defaults)
            if not repair_keys or all(
                candidate.get(key) == defaults[key] for key in repair_keys
            ):
                raise ValueError(message)
            for key in repair_keys:
                candidate[key] = defaults[key]
                warnings.append(
                    f"Invalid value restored: alignment.method_params.{key}."
                )
        else:
            raise ValueError("Alignment config normalization did not converge.")
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
        try:
            payload = self._build_alignment_config_export_payload()
            export_path.parent.mkdir(parents=True, exist_ok=True)
            with export_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    payload,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
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

        preview_lines = [
            "Review the Align Epochs config normalization before importing:",
            "",
        ]
        if warnings:
            preview_lines.extend(f"- {warning}" for warning in warnings)
        else:
            preview_lines.append("- No normalization was required.")
        preview_lines.extend(["", "Apply this imported configuration?"])
        confirmed = self._ask_question(
            "Import Align Epochs Configs",
            "\n".join(preview_lines),
            buttons=QMessageBox.Yes | QMessageBox.No,
            default_button=QMessageBox.No,
        )
        if confirmed != QMessageBox.Yes:
            self.statusBar().showMessage("Align Epochs config import cancelled.")
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
