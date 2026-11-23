"""Epoch Inspector table, commands, and run-owned decisions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import QDialog, QInputDialog, QTableWidgetItem

from lfptensorpipe.app import PathResolver
from lfptensorpipe.app.alignment.epoch_view import load_epoch_review, save_epoch_review
from lfptensorpipe.app.alignment.inspector_config import (
    SIGNAL,
    effective_program,
    factory_scheme,
    normalize_scheme,
)
from lfptensorpipe.io.atomic_file import write_json_atomically


def _qc_tooltip(result, resolved):
    if not result:
        return ""
    heading = {True: "QC flagged", False: "QC clear", None: "QC not assessable"}[
        result["flag"]
    ]
    lines = [heading]
    for name, rule in result["rules"].items():
        spec = resolved[name]["rule"]
        metric = "Preprocessed Signal" if spec["metric"] == SIGNAL else spec["metric"]
        reason = (
            f"MAD > {spec['threshold']:g}"
            if spec["method"] == "MAD"
            else f"valid coverage < {spec['coverage']:g}%"
        )
        if rule["hit"] is None:
            lines.append(f"{name} · {metric} · {reason} · Not assessable")
        if rule["evidence"]:
            lines.append(f"{name} · {metric} · {reason}")
        for e in rule["evidence"]:
            low, high = e["time"]
            suffix = "%" if e["unit"] == "%" else " s"
            text = f"channel: {e['channel']}, time: {low:g}–{high:g}{suffix}"
            if "frequency" in e:
                a, b = e["frequency"]
                text += f", frequency: {a:g}–{b:g} Hz"
            elif "bins" in e:
                text += (
                    ", parameter: "
                    if spec["metric"] == "aperiodic"
                    else ", frequency: "
                ) + ", ".join(e["bins"])
            if "coverage_percent" in e:
                text += (
                    f" (evaluated window, valid coverage {e['coverage_percent']:.1f}%)"
                )
            lines.append(text)
            if e.get("missing_time"):
                gaps = ", ".join(f"{a:g}–{b:g}{suffix}" for a, b in e["missing_time"])
                lines.append(f"Missing support: {gaps}")
    return "\n".join(lines)


class MainWindowAlignmentEpochsMixin:
    def _set_alignment_epoch_rows(self, rows):
        self._alignment_epoch_rows = list(rows)
        context, slug = self._record_context(), self._current_alignment_paradigm_slug()
        self._epoch_review = (
            load_epoch_review(context, slug) if rows and context and slug else {}
        )
        table = self._alignment_epoch_table
        if table is None:
            return
        table.blockSignals(True)
        table.setRowCount(len(rows))
        view = set(self._epoch_review.get("view") or [])
        for pos, row in enumerate(rows):
            idx = row.get("epoch_index", pos)
            duration = row.get("duration_s", np.nan)
            for col, text in enumerate(
                (
                    "",
                    f"{idx+1} · {row.get('epoch_label', '')}",
                    f"{duration:.3f} s" if np.isfinite(duration) else "",
                    "",
                    "",
                )
            ):
                item = QTableWidgetItem(text)
                if col in (0, 3):
                    item.setFlags(
                        Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable
                    )
                    checked = row.get("pick", True) if col == 0 else idx in view
                    item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
                table.setItem(pos, col, item)
        table.blockSignals(False)
        self._paint_review_rows()

    def _paint_review_rows(self):
        table = self._alignment_epoch_table
        report = self._epoch_review.get("qc_result") or {}
        results = report.get("epochs", [])
        notes = self._epoch_review.get("notes") or {}
        table.blockSignals(True)
        for row, record in enumerate(self._alignment_epoch_rows):
            index = record.get("epoch_index", row)
            outcome = results[index] if index < len(results) else {}
            brush = QBrush(QColor("#f5adae")) if outcome.get("flag") else QBrush()
            foreground = QBrush(QColor("#281416")) if outcome.get("flag") else QBrush()
            note = notes.get(str(index), "")
            table.item(row, 4).setText(note.replace("\n", " "))
            tip = _qc_tooltip(outcome, report.get("resolved", {}))
            for col in range(5):
                item = table.item(row, col)
                item.setBackground(brush)
                item.setForeground(foreground)
                item.setToolTip(
                    "\n".join(v for v in (tip, note if col == 4 else "") if v)
                )
        table.blockSignals(False)

    def _checked_epochs(self, col):
        table = self._alignment_epoch_table
        if table is None:
            return []
        return [
            r.get("epoch_index", i)
            for i, r in enumerate(self._alignment_epoch_rows)
            if table.item(i, col) is not None
            and table.item(i, col).checkState() == Qt.Checked
        ]

    def _selected_alignment_epoch_indices(self):
        return self._checked_epochs(0)

    def _collect_alignment_pick_indices(self):
        return self._selected_alignment_epoch_indices()

    def _save_review(self, patch, *, owner=None):
        ctx, slug = self._record_context(), self._current_alignment_paradigm_slug()
        if ctx and slug and not self._record_param_syncing:
            if owner is None:
                owner = self._epoch_review.get("run_history_index")
            if not save_epoch_review(ctx, slug, patch, run_index=owner):
                raise ValueError("The trial Run log is unavailable.")
            self._epoch_review.update(patch)

    def _change_column(self, col, selected, *, exclusions=None):
        table = self._alignment_epoch_table
        previous = set(self._checked_epochs(col))
        if selected == previous:
            return
        table.blockSignals(True)
        for pos, row in enumerate(self._alignment_epoch_rows):
            item = table.item(pos, col)
            item.setCheckState(
                Qt.Checked if row.get("epoch_index", pos) in selected else Qt.Unchecked
            )
        table.blockSignals(False)
        self._save_selection(col, exclusions=exclusions)

    def _save_selection(self, col, *, exclusions=None):
        if self._record_param_syncing:
            return
        values = self._checked_epochs(col)
        if col == 0:
            decisions = dict(
                exclusions
                if exclusions is not None
                else self._epoch_review.get("exclusions") or {}
            )
            for i in values:
                decisions.pop(str(i), None)
            patch = {"picked_epoch_indices": values, "exclusions": decisions}
        else:
            patch = {"view": values}
        try:
            self._save_review(patch)
        except (OSError, ValueError) as exc:
            self._refresh_alignment_selected_paradigm()
            self._show_warning("Epoch Inspector", str(exc))
            return
        if col == 0:
            self._mark_record_param_dirty("alignment.picks")
            self._refresh_alignment_controls()
            self._refresh_trial_stage_controls()
        self._refresh_alignment_select_all_button_label()

    def _on_alignment_epoch_item_changed(self, item):
        if item.column() in (0, 3):
            self._save_selection(item.column())

    def _toggle_column(self, col):
        all_ids = {
            r.get("epoch_index", i) for i, r in enumerate(self._alignment_epoch_rows)
        }
        self._change_column(
            col, set() if set(self._checked_epochs(col)) == all_ids else all_ids
        )

    def _on_alignment_select_all(self):
        self._toggle_column(0)

    def _toggle_epoch_view(self):
        self._toggle_column(3)

    def _refresh_alignment_select_all_button_label(self):
        for col, label, name in (
            (0, "Include", "_alignment_select_all_button"),
            (3, "View", "_view_all"),
        ):
            btn = getattr(self, name, None)
            if btn is not None:
                all_on = self._alignment_epoch_rows and len(
                    self._checked_epochs(col)
                ) == len(self._alignment_epoch_rows)
                btn.setText(f"{'Clear all' if all_on else 'Select all'} {label}")

    def _refresh_epoch_review_controls(self, ready):
        if not hasattr(self, "_qc_run"):
            return
        result = self._epoch_review.get("qc_result")
        self._qc_run.setToolTip("QC current" if result else "QC needs run")
        for btn in (self._qc_run, self._qc_advance, self._qc_import, self._qc_export):
            btn.setEnabled(ready)
        self._view_all.setEnabled(bool(self._alignment_epoch_rows))
        for btn in (self._qc_view, self._qc_exclude):
            btn.setEnabled(ready and result is not None)
        self._refresh_alignment_select_all_button_label()

    def _flagged(self):
        return {
            i
            for i, r in enumerate(self._epoch_review["qc_result"]["epochs"])
            if r["flag"] is True
        }

    def _view_flagged(self):
        self._change_column(3, self._flagged())

    def _exclude_flagged(self):
        result = self._epoch_review["qc_result"]
        include = set(self._checked_epochs(0))
        flagged = self._flagged()
        decisions = dict(self._epoch_review.get("exclusions") or {})
        for idx in include & flagged:
            decisions[str(idx)] = {
                "config": result["config"],
                "resolved": result["resolved"],
                "result": result["epochs"][idx],
            }
        self._change_column(0, include - flagged, exclusions=decisions)

    def _inspector_data(self):
        from lfptensorpipe.app.alignment.inspector_data import InspectorData

        data = InspectorData(
            PathResolver(self._record_context()),
            self._current_alignment_paradigm_slug(),
        )
        if data.run_index != self._epoch_review.get("run_history_index"):
            self._refresh_alignment_selected_paradigm()
            raise ValueError("The Align Run changed. Review the updated epoch list.")
        return data

    def _qc_default(self):
        settings = self._config_store.read_yaml("alignment_preview.yml", default={})
        return normalize_scheme(settings.get("qc_default", factory_scheme()))

    def _save_qc_default(self, config):
        if not self._epoch_review.get("qc_config"):
            self._save_review({"qc_config": self._qc_scheme()})
        settings = self._config_store.read_yaml("alignment_preview.yml", default={})
        self._config_store.write_yaml(
            "alignment_preview.yml", {**settings, "qc_default": config}
        )

    def _qc_scheme(self):
        return self._epoch_review.get("qc_config") or self._qc_default()

    def _edit_epoch_qc(self, checked=False, *, imported=None):
        from lfptensorpipe.gui.dialogs.epoch_qc import EpochQCDialog

        try:
            data = self._inspector_data()
            current = self._qc_scheme()
            dialog = EpochQCDialog(
                self,
                data,
                imported or current,
                self._load_tensor_bands_defaults(),
                self._qc_default,
                self._save_qc_default,
            )
            if dialog.exec() != QDialog.Accepted:
                return
            patch = {"qc_config": dialog.scheme}
            result = self._epoch_review.get("qc_result")
            if result and effective_program(result["config"]) != effective_program(
                dialog.scheme
            ):
                patch["qc_result"] = None
            self._save_review(patch, owner=data.run_index)
            self._paint_review_rows()
            self._refresh_epoch_review_controls(True)
        except (OSError, ValueError) as exc:
            self._show_warning("QC Advance", str(exc))

    def _qc_export_path(self):
        root = self._alignment_config_default_path(
            self._record_context(), trial_slug=self._current_alignment_paradigm_slug()
        ).parent
        return root / "epoch_qc.json"

    def _export_epoch_qc(self):
        name, _ = self._save_file_name(
            "Export Epoch Inspector QC",
            str(self._qc_export_path()),
            "JSON files (*.json)",
        )
        if name:
            try:
                path = Path(name)
                write_json_atomically(
                    path if path.suffix else path.with_suffix(".json"),
                    self._qc_scheme(),
                )
            except (OSError, ValueError) as exc:
                self._show_warning("Export QC", str(exc))

    def _import_epoch_qc(self):
        name, _ = self._open_file_name(
            "Import Epoch Inspector QC",
            str(self._qc_export_path().parent),
            "JSON files (*.json)",
        )
        if name:
            try:
                config = normalize_scheme(
                    json.loads(Path(name).read_text(encoding="utf-8"))
                )
            except (OSError, ValueError) as exc:
                self._show_warning("Import QC", str(exc))
                return
            self._edit_epoch_qc(imported=config)

    def _run_epoch_qc(self):
        from lfptensorpipe.app.alignment.inspector_qc import run_qc

        try:
            data = self._inspector_data()
            config = self._qc_scheme()
            result = self._run_with_busy("Run QC", lambda: run_qc(data, config))
            self._save_review(
                {"qc_config": config, "qc_result": result}, owner=data.run_index
            )
        except (OSError, ValueError) as exc:
            self._show_warning("Run QC", str(exc))
            return
        self._paint_review_rows()
        self._refresh_epoch_review_controls(True)
        epochs = result["epochs"]
        unknown = sum(r["flag"] is None for r in epochs)
        self.statusBar().showMessage(
            f"QC: {len(self._flagged())}/{len(epochs)} flagged, {unknown} not assessable."
        )

    def _open_epoch_cell(self, row, col):
        index = self._alignment_epoch_rows[row].get("epoch_index", row)
        if col == 4:
            notes = dict(self._epoch_review.get("notes") or {})
            key = str(index)
            editor = QInputDialog(self)
            editor.setWindowTitle("Epoch notes")
            editor.setLabelText("Notes (optional)")
            editor.setOption(QInputDialog.UsePlainTextEditForTextInput)
            editor.setTextValue(notes.get(key, ""))
            editor.setOkButtonText("Save")
            if editor.exec() == QDialog.Accepted:
                note = editor.textValue()
                if note:
                    notes[key] = note
                else:
                    notes.pop(key, None)
                try:
                    self._save_review({"notes": notes})
                except (OSError, ValueError) as exc:
                    self._show_warning("Epoch notes", str(exc))
                    return
                self._paint_review_rows()
        elif col in (1, 2) and self._enable_plots:
            try:
                self._show_epoch_signal(self, self._inspector_data(), index)
            except (OSError, ValueError) as exc:
                self._show_warning("Preprocessed Signal", str(exc))

    def _show_epoch_signal(self, parent, data, idx, *, channels=None):
        from lfptensorpipe.gui.dialogs.epoch_preview import show_waveform

        rows = (self._epoch_review.get("qc_result") or {}).get("epochs", [])
        show_waveform(
            parent, data, idx, rows[idx] if idx < len(rows) else None, channels=channels
        )

    def _on_alignment_preview(self):
        if not self._enable_plots:
            return
        from lfptensorpipe.gui.dialogs.epoch_preview import EpochPreview

        try:
            data = self._inspector_data()

            def persist(prefs):
                if prefs != self._epoch_review.get("preview"):
                    self._save_review({"preview": prefs}, owner=data.run_index)

            dialog = EpochPreview(
                self,
                data,
                self._checked_epochs(3),
                self._epoch_review.get("preview") or {},
                self._load_tensor_bands_defaults(),
                persist,
                lambda i: self._show_epoch_signal(
                    dialog, data, i, channels=dialog.channels.values()
                ),
                style_for_metric=self._load_alignment_preview_plot_params,
            )
            dialog.exec()
        except (OSError, ValueError) as exc:
            self._show_warning("Preview", str(exc))
