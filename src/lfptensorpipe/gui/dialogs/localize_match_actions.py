"""Action helpers for the localize match dialog."""

from __future__ import annotations

from .common import Any, QMessageBox
from .localize_match_render import (
    _render_channel_list,
    _render_mapping_table,
    _sync_draft_widgets,
)

MATCH_DEFAULTS_KEY = "match_defaults"


def _mapping_rows(dialog) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for channel in dialog._all_channels:
        mapping = dialog._mapping.get(channel)
        if mapping is None:
            continue
        rows.append(
            {
                "channel": channel,
                "anode": str(mapping.get("anode", "")).strip(),
                "cathode": str(mapping.get("cathode", "")).strip(),
                "rep_coord": str(mapping.get("rep_coord", "Mid")).strip().title(),
            }
        )
    return rows


def build_lead_signature(lead_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    signature: list[dict[str, Any]] = []
    for lead in lead_specs:
        if not isinstance(lead, dict):
            continue
        contacts_out: list[dict[str, str]] = []
        contacts = lead.get("contacts", [])
        if not isinstance(contacts, list):
            contacts = []
        for contact in contacts:
            if not isinstance(contact, dict):
                continue
            token = str(contact.get("token", "")).strip()
            contact_name = str(contact.get("contact_name", "")).strip()
            if not token or not contact_name:
                continue
            contacts_out.append(
                {
                    "token": token,
                    "contact_name": contact_name,
                }
            )
        signature.append(
            {
                "display_name": str(lead.get("display_name", "")).strip(),
                "contacts": contacts_out,
            }
        )
    return signature


def _current_lead_signature(dialog) -> list[dict[str, Any]]:
    return build_lead_signature(dialog._lead_specs)


def _normalize_saved_match_defaults(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    channels_raw = payload.get("channels")
    signature_raw = payload.get("lead_signature")
    mappings_raw = payload.get("mappings")
    if not isinstance(channels_raw, list):
        return None
    if not isinstance(signature_raw, list):
        return None
    if not isinstance(mappings_raw, list):
        return None

    channels = [str(item).strip() for item in channels_raw]
    signature: list[dict[str, Any]] = []
    for lead in signature_raw:
        if not isinstance(lead, dict):
            return None
        contacts_raw = lead.get("contacts", [])
        if not isinstance(contacts_raw, list):
            return None
        contacts_out: list[dict[str, str]] = []
        for contact in contacts_raw:
            if not isinstance(contact, dict):
                return None
            token = str(contact.get("token", "")).strip()
            contact_name = str(contact.get("contact_name", "")).strip()
            if not token or not contact_name:
                continue
            contacts_out.append(
                {
                    "token": token,
                    "contact_name": contact_name,
                }
            )
        signature.append(
            {
                "display_name": str(lead.get("display_name", "")).strip(),
                "contacts": contacts_out,
            }
        )

    mapping_out: dict[str, dict[str, str]] = {}
    valid_channels = set(channels)
    for item in mappings_raw:
        if not isinstance(item, dict):
            continue
        channel = str(item.get("channel", "")).strip()
        anode = str(item.get("anode", "")).strip()
        cathode = str(item.get("cathode", "")).strip()
        rep_coord = str(item.get("rep_coord", "")).strip().title()
        if channel not in valid_channels or not anode or not cathode:
            continue
        if rep_coord not in {"Anode", "Cathode", "Mid"}:
            rep_coord = "Mid"
        mapping_out[channel] = {
            "anode": anode,
            "cathode": cathode,
            "rep_coord": rep_coord,
        }
    return {
        "channels": channels,
        "lead_signature": signature,
        "mapping": mapping_out,
    }


def _show_status_message(dialog, message: str) -> None:
    parent = dialog.parentWidget()
    if parent is None or not hasattr(parent, "statusBar"):
        return
    status_bar = parent.statusBar()
    if status_bar is None:
        return
    status_bar.showMessage(message)


def _on_set_default(dialog) -> None:
    if dialog._config_store is None:
        _show_status_message(
            dialog, "Match defaults unavailable: config store missing."
        )
        return
    payload = dialog._config_store.read_yaml("localization.yml", default={})
    if not isinstance(payload, dict):
        payload = {}
    payload = dict(payload)
    payload[MATCH_DEFAULTS_KEY] = {
        "channels": list(dialog._all_channels),
        "lead_signature": _current_lead_signature(dialog),
        "mappings": _mapping_rows(dialog),
    }
    dialog._config_store.write_yaml("localization.yml", payload)
    _show_status_message(dialog, "Localize Match defaults saved to app storage.")


def _on_restore_default(dialog) -> None:
    if dialog._config_store is None:
        _show_status_message(
            dialog, "Match defaults unavailable: config store missing."
        )
        return
    payload = dialog._config_store.read_yaml("localization.yml", default={})
    if not isinstance(payload, dict):
        payload = {}
    saved = _normalize_saved_match_defaults(payload.get(MATCH_DEFAULTS_KEY))
    if saved is None:
        _show_status_message(
            dialog,
            "No compatible Localize Match default is available.",
        )
        return
    if tuple(saved["channels"]) != dialog._all_channels:
        _show_status_message(
            dialog,
            "No compatible Localize Match default is available.",
        )
        return
    if saved["lead_signature"] != _current_lead_signature(dialog):
        _show_status_message(
            dialog,
            "No compatible Localize Match default is available.",
        )
        return
    dialog._mapping = dict(saved["mapping"])
    dialog._active_channel = None
    dialog._selected_channel_label.setText("-")
    dialog._draft_anode = ""
    dialog._draft_cathode = ""
    dialog._draft_rep_coord = "Mid"
    _sync_draft_widgets(dialog)
    _render_mapping_table(dialog)
    _render_channel_list(dialog)
    _show_status_message(dialog, "Localize Match defaults restored.")


def _on_contact_clicked(dialog, token: str, cathode_only: bool) -> None:
    if dialog._active_channel is None:
        QMessageBox.warning(dialog, "Match", "Select one channel first.")
        return
    if cathode_only:
        if not dialog._draft_anode:
            QMessageBox.warning(
                dialog,
                "Match",
                "Select Anode first before choosing case/ground.",
            )
            return
        if dialog._draft_cathode:
            return
        dialog._draft_cathode = token
        _sync_draft_widgets(dialog)
        return
    if not dialog._draft_anode:
        dialog._draft_anode = token
        _sync_draft_widgets(dialog)
        return
    if not dialog._draft_cathode:
        dialog._draft_cathode = token
        _sync_draft_widgets(dialog)


def _on_bind_update(dialog) -> None:
    channel = (dialog._active_channel or "").strip()
    if not channel:
        QMessageBox.warning(dialog, "Match", "Select one channel first.")
        return
    anode = dialog._draft_anode.strip()
    cathode = dialog._draft_cathode.strip()
    rep_coord = dialog._draft_rep_coord.strip().title()
    if not anode or not cathode:
        QMessageBox.warning(dialog, "Match", "Anode and Cathode are required.")
        return
    if anode.lower() in {"case", "ground"}:
        QMessageBox.warning(dialog, "Match", "Anode cannot be case/ground.")
        return
    if anode == cathode:
        QMessageBox.warning(dialog, "Match", "Anode and Cathode cannot be identical.")
        return
    if anode not in dialog._contact_by_token:
        QMessageBox.warning(dialog, "Match", f"Unknown anode token: {anode}")
        return
    cathode_lower = cathode.lower()
    if (
        cathode_lower not in {"case", "ground"}
        and cathode not in dialog._contact_by_token
    ):
        QMessageBox.warning(dialog, "Match", f"Unknown cathode token: {cathode}")
        return
    if cathode_lower in {"case", "ground"} and rep_coord != "Anode":
        QMessageBox.warning(
            dialog,
            "Match",
            "Rep. coord must be Anode when Cathode is case/ground.",
        )
        return
    dialog._mapping[channel] = {
        "anode": anode,
        "cathode": cathode,
        "rep_coord": rep_coord,
    }
    _render_mapping_table(dialog)
    _render_channel_list(dialog)
    dialog._draft_anode = ""
    dialog._draft_cathode = ""
    dialog._draft_rep_coord = "Mid"
    _sync_draft_widgets(dialog)


def _on_delete_mapping(dialog, channel: str) -> None:
    dialog._mapping.pop(channel, None)
    _render_mapping_table(dialog)
    _render_channel_list(dialog)


def _on_auto_match(dialog) -> None:
    for channel in dialog._all_channels:
        if channel in dialog._mapping:
            continue
        parsed = dialog._auto_channel_pair(channel)
        if parsed is None:
            continue
        left, right, side_constraint = parsed
        candidates: list[tuple[str, str]] = []
        for lead_tokens, lead_sides in zip(
            dialog._lead_auto_tokens, dialog._lead_auto_sides
        ):
            if side_constraint is not None and side_constraint not in lead_sides:
                continue
            anode = lead_tokens.get(left)
            cathode = lead_tokens.get(right)
            if not anode or not cathode or anode == cathode:
                continue
            candidates.append((anode, cathode))
        if len(candidates) != 1:
            continue
        anode, cathode = candidates[0]
        dialog._mapping[channel] = {
            "anode": anode,
            "cathode": cathode,
            "rep_coord": "Mid",
        }
    _render_mapping_table(dialog)
    _render_channel_list(dialog)


def _on_reset_all(dialog) -> None:
    confirm = QMessageBox.question(
        dialog,
        "Reset",
        "Clear all mappings for this record?",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No,
    )
    if confirm != QMessageBox.Yes:
        return
    dialog._mapping = {}
    dialog._draft_anode = ""
    dialog._draft_cathode = ""
    dialog._draft_rep_coord = "Mid"
    _sync_draft_widgets(dialog)
    _render_mapping_table(dialog)
    _render_channel_list(dialog)


def _on_save(dialog) -> None:
    if len(dialog._mapping) != len(dialog._all_channels):
        missing = len(dialog._all_channels) - len(dialog._mapping)
        QMessageBox.warning(
            dialog,
            "Match",
            f"All channels must be mapped before save. Missing: {missing}",
        )
        return
    rows: list[dict[str, str]] = []
    for channel in dialog._all_channels:
        mapping = dialog._mapping.get(channel)
        if mapping is None:
            continue
        rows.append(
            {
                "channel": channel,
                "anode": mapping["anode"],
                "cathode": mapping["cathode"],
                "rep_coord": mapping["rep_coord"],
            }
        )
    dialog._selected_payload = {
        "completed": len(rows) == len(dialog._all_channels),
        "channels": list(dialog._all_channels),
        "mapped_count": len(rows),
        "mappings": rows,
    }
    dialog.accept()
