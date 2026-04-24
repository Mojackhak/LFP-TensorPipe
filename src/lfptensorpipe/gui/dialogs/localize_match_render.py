"""Render/state helpers for the localize match dialog."""

from __future__ import annotations

from typing import Any

from .common import (
    ACTION_PAYLOAD_ROLE,
    QListWidgetItem,
    QGroupBox,
    QPushButton,
    QTableWidgetItem,
    QVBoxLayout,
    make_action_table_item,
)


def _build_contact_maps(dialog) -> None:
    dialog._contact_by_token = {}
    dialog._lead_auto_tokens = []
    dialog._lead_auto_sides = []
    for lead in dialog._lead_specs:
        lead_tokens: dict[str, str] = {}
        lead_sides: set[str] = set()
        contacts = lead.get("contacts", [])
        if not isinstance(contacts, list):
            dialog._lead_auto_tokens.append(lead_tokens)
            dialog._lead_auto_sides.append(lead_sides)
            continue
        for contact in contacts:
            if not isinstance(contact, dict):
                continue
            token = str(contact.get("token", "")).strip()
            if not token:
                continue
            dialog._contact_by_token[token] = contact
            auto_index, auto_side = dialog._auto_contact_index_side(
                str(contact.get("contact_name", ""))
            )
            if auto_index:
                lead_tokens.setdefault(auto_index, token)
            if auto_side:
                lead_sides.add(auto_side)
        dialog._lead_auto_tokens.append(lead_tokens)
        dialog._lead_auto_sides.append(lead_sides)


def _load_current_payload(dialog, payload: dict[str, Any] | None) -> None:
    dialog._mapping = {}
    if not isinstance(payload, dict):
        return
    raw_mappings = payload.get("mappings")
    if not isinstance(raw_mappings, list):
        return
    for item in raw_mappings:
        if not isinstance(item, dict):
            continue
        channel = str(item.get("channel", "")).strip()
        if channel not in dialog._all_channels:
            continue
        anode = str(item.get("anode", "")).strip()
        cathode = str(item.get("cathode", "")).strip()
        rep_coord = str(item.get("rep_coord", "")).strip().title()
        if rep_coord not in {"Anode", "Cathode", "Mid"}:
            rep_coord = "Mid"
        if not channel or not anode or not cathode:
            continue
        dialog._mapping[channel] = {
            "anode": anode,
            "cathode": cathode,
            "rep_coord": rep_coord,
        }


def _render_lead_columns(dialog) -> None:
    while dialog._lead_scroll_layout.count():
        item = dialog._lead_scroll_layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()
    for lead in dialog._lead_specs:
        column = QGroupBox(str(lead.get("display_name", "Lead")))
        column_layout = QVBoxLayout(column)
        column_layout.setContentsMargins(6, 6, 6, 6)
        column_layout.setSpacing(4)
        contacts = lead.get("contacts", [])
        if not isinstance(contacts, list):
            contacts = []
        for contact in contacts:
            if not isinstance(contact, dict):
                continue
            token = str(contact.get("token", "")).strip()
            name = str(contact.get("contact_name", "")).strip()
            if not token or not name:
                continue
            button = QPushButton(name)
            button.clicked.connect(
                lambda checked=False, tk=token: dialog._on_contact_clicked(tk, False)
            )
            button.setToolTip("Select this contact for the current channel binding.")
            column_layout.addWidget(button)
        column_layout.addStretch(1)
        dialog._lead_scroll_layout.addWidget(column)
    dialog._lead_scroll_layout.addStretch(1)


def _render_channel_list(dialog) -> None:
    token = dialog._search_edit.text().strip().lower()
    dialog._channel_list.clear()
    unmapped = [item for item in dialog._all_channels if item not in dialog._mapping]
    for channel in unmapped:
        if token and token not in channel.lower():
            continue
        dialog._channel_list.addItem(QListWidgetItem(channel))
    if dialog._active_channel is None and dialog._channel_list.count() > 0:
        dialog._channel_list.setCurrentRow(0)
    _refresh_status(dialog)


def _render_mapping_table(dialog) -> None:
    ordered = [
        channel for channel in dialog._all_channels if channel in dialog._mapping
    ]
    dialog._mapping_table.setRowCount(len(ordered))
    for row_idx, channel in enumerate(ordered):
        mapping = dialog._mapping[channel]
        dialog._mapping_table.setItem(row_idx, 0, QTableWidgetItem(channel))
        dialog._mapping_table.setItem(row_idx, 1, QTableWidgetItem(mapping["anode"]))
        dialog._mapping_table.setItem(row_idx, 2, QTableWidgetItem(mapping["cathode"]))
        dialog._mapping_table.setItem(
            row_idx, 3, QTableWidgetItem(mapping["rep_coord"])
        )
        dialog._mapping_table.setItem(
            row_idx,
            4,
            make_action_table_item(
                "Del",
                channel,
                tool_tip="Delete this channel mapping.",
            ),
        )
    _refresh_status(dialog)


def _refresh_status(dialog) -> None:
    mapped = len(dialog._mapping)
    total = len(dialog._all_channels)
    dialog._status_label.setText(f"Status: {mapped}/{total} mapped")
    dialog._status_label.setToolTip(
        "Mapped channels / total channels. "
        f"Save requires full mapping ({mapped}/{total})."
    )


def _on_channel_selected(dialog) -> None:
    item = dialog._channel_list.currentItem()
    if item is None:
        return
    dialog._active_channel = item.text().strip()
    dialog._selected_channel_label.setText(dialog._active_channel)
    dialog._selected_channel_label.setToolTip(
        "Current record channel being edited. "
        f"Selected: {dialog._active_channel or '-'}."
    )
    existing = dialog._mapping.get(dialog._active_channel)
    if existing is not None:
        dialog._draft_anode = existing.get("anode", "")
        dialog._draft_cathode = existing.get("cathode", "")
        dialog._draft_rep_coord = existing.get("rep_coord", "Mid")
    else:
        dialog._draft_anode = ""
        dialog._draft_cathode = ""
        dialog._draft_rep_coord = "Mid"
    _sync_draft_widgets(dialog)


def _on_mapping_row_clicked(dialog, row: int, col: int) -> None:
    if col == 4:
        action_item = dialog._mapping_table.item(row, 4)
        if action_item is None:
            return
        channel = str(action_item.data(ACTION_PAYLOAD_ROLE) or "").strip()
        if not channel:
            return
        dialog._on_delete_mapping(channel)
        return

    item = dialog._mapping_table.item(row, 0)
    if item is None:
        return
    channel = item.text().strip()
    if not channel:
        return
    dialog._active_channel = channel
    dialog._selected_channel_label.setText(channel)
    dialog._selected_channel_label.setToolTip(
        "Current record channel being edited. " f"Selected: {channel or '-'}."
    )
    existing = dialog._mapping.get(channel)
    if existing is None:
        dialog._draft_anode = ""
        dialog._draft_cathode = ""
        dialog._draft_rep_coord = "Mid"
    else:
        dialog._draft_anode = existing.get("anode", "")
        dialog._draft_cathode = existing.get("cathode", "")
        dialog._draft_rep_coord = existing.get("rep_coord", "Mid")
    _sync_draft_widgets(dialog)


def _clear_draft_endpoint(dialog, endpoint: str) -> None:
    if endpoint == "anode":
        dialog._draft_anode = ""
    else:
        dialog._draft_cathode = ""
    _sync_draft_widgets(dialog)


def _sync_draft_widgets(dialog) -> None:
    dialog._anode_button.setText(dialog._draft_anode)
    dialog._cathode_button.setText(dialog._draft_cathode)
    rep_options = ["Anode", "Cathode", "Mid"]
    cathode_lower = dialog._draft_cathode.strip().lower()
    if cathode_lower in {"case", "ground"}:
        rep_options = ["Anode"]
        dialog._draft_rep_coord = "Anode"
    dialog._rep_combo.blockSignals(True)
    dialog._rep_combo.clear()
    dialog._rep_combo.addItems(rep_options)
    target_idx = dialog._rep_combo.findText(dialog._draft_rep_coord)
    dialog._rep_combo.setCurrentIndex(target_idx if target_idx >= 0 else 0)
    dialog._rep_combo.blockSignals(False)
    dialog._draft_rep_coord = dialog._rep_combo.currentText()


def _on_rep_coord_changed(dialog, _index: int) -> None:
    dialog._draft_rep_coord = dialog._rep_combo.currentText()
