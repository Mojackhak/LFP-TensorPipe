"""Localize configuration identity and channel matching."""

from __future__ import annotations

from typing import Any


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


def normalize_localize_lead_signature(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise ValueError("Localize config is missing required `lead_signature` list.")
    normalized = build_lead_signature(
        [item for item in payload if isinstance(item, dict)]
    )
    if not normalized:
        raise ValueError("Localize config `lead_signature` is empty.")
    return normalized


def normalize_localize_match_payload(
    payload: Any,
    *,
    expected_channels: tuple[str, ...],
    strict: bool = False,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Localize config is missing required `match` object.")
    channels_raw = payload.get("channels")
    mappings_raw = payload.get("mappings")
    if not isinstance(channels_raw, list):
        raise ValueError("Localize config match is missing required `channels` list.")
    if not isinstance(mappings_raw, list):
        raise ValueError("Localize config match is missing required `mappings` list.")
    channels = tuple(str(item).strip() for item in channels_raw)
    if channels != expected_channels:
        raise ValueError(
            "Imported Localize match channels do not match the current record."
        )

    rows: list[dict[str, str]] = []
    seen_channels: set[str] = set()
    for item in mappings_raw:
        if not isinstance(item, dict):
            if strict:
                raise ValueError("Every Localize match mapping must be an object.")
            continue
        channel = str(item.get("channel", "")).strip()
        anode = str(item.get("anode", "")).strip()
        cathode = str(item.get("cathode", "")).strip()
        rep_coord = str(item.get("rep_coord", "Mid")).strip().title()
        if channel not in expected_channels:
            if strict:
                raise ValueError(f"Unknown Localize mapping channel: {channel}")
            continue
        if not anode or not cathode:
            if strict:
                raise ValueError(f"Missing Localize contact pair for {channel}.")
            continue
        if strict and "rep_coord" not in item:
            raise ValueError(
                f"Missing Localize representative coordinate for {channel}."
            )
        if rep_coord not in {"Anode", "Cathode", "Mid"}:
            if strict:
                raise ValueError(
                    f"Invalid Localize representative coordinate: {rep_coord}"
                )
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
        "mappings": rows,
    }
