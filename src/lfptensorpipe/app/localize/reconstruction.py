"""Reconstruction parsing helpers for Localize workflows."""

from __future__ import annotations

from concurrent.futures import TimeoutError as FutureTimeout
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import scipy.io

from .paths import reconstruction_mat_path


def ordinal_label(index_1based: int) -> str:
    if index_1based == 1:
        return "R"
    if index_1based == 2:
        return "L"
    suffix = "th"
    if index_1based % 10 == 1 and index_1based % 100 != 11:
        suffix = "st"
    elif index_1based % 10 == 2 and index_1based % 100 != 12:
        suffix = "nd"
    elif index_1based % 10 == 3 and index_1based % 100 != 13:
        suffix = "rd"
    return f"{index_1based}{suffix}"


def unwrap_matlab_node(value: Any) -> Any:
    current = value
    while isinstance(current, np.ndarray):
        if current.size == 0:
            return None
        current = current.flat[0]
    return current


def extract_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, np.str_):
        text = str(value).strip()
        return text or None
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="ignore").strip()
        return text or None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        if value.dtype.kind in {"U", "S"}:
            text = str(value.flat[0]).strip()
            return text or None
        if value.size == 1:
            return extract_string(value.flat[0])
    return None


def extract_coords_by_space(reco: Any, space_key: str) -> list[np.ndarray]:
    space_node = unwrap_matlab_node(getattr(reco, space_key, None))
    if space_node is None or not hasattr(space_node, "coords_mm"):
        return []
    coords_cell = getattr(space_node, "coords_mm")
    if not isinstance(coords_cell, np.ndarray):
        return []
    out: list[np.ndarray] = []
    for item in coords_cell.flat:
        arr = np.asarray(item, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3:
            out.append(np.zeros((0, 3), dtype=float))
            continue
        out.append(arr)
    return out


def extract_elmodels(reco: Any, n_leads: int) -> list[str]:
    props = getattr(reco, "props", None)
    if props is None:
        return ["Unknown"] * n_leads
    values: list[str] = []
    if isinstance(props, np.ndarray):
        iterable = list(props.flat)
    else:
        iterable = [props]
    for node in iterable:
        item = unwrap_matlab_node(node)
        if item is None or not hasattr(item, "elmodel"):
            continue
        text = extract_string(getattr(item, "elmodel"))
        if text:
            values.append(text)
    if not values:
        values = ["Unknown"]
    if len(values) < n_leads:
        values.extend([values[0]] * (n_leads - len(values)))
    return values[:n_leads]


def json_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        decoded = json.loads(raw)
    except Exception:
        return []
    if isinstance(decoded, list):
        return [str(item) for item in decoded]
    if isinstance(decoded, str):
        return [decoded]
    return []


def resolve_elspec(paths: Any, elmodel: str) -> dict[str, Any]:
    from . import service as svc

    model_key = elmodel.strip()
    cached = svc._ELSPEC_CACHE.get(model_key)
    if cached is not None:
        return cached

    def _task(eng: Any) -> dict[str, Any]:
        eng.workspace["lfptp_elmodel"] = model_key
        eng.eval("lfptp_opts = struct('elmodel', lfptp_elmodel);", nargout=0)
        eng.eval("lfptp_opts = ea_resolve_elspec(lfptp_opts);", nargout=0)

        matfname = str(eng.eval("char(lfptp_opts.elspec.matfname);", nargout=1)).strip()
        contact_length = float(
            eng.eval("double(lfptp_opts.elspec.contact_length);", nargout=1)
        )
        contact_spacing_raw = str(
            eng.eval(
                "jsonencode(double(lfptp_opts.elspec.contact_spacing));",
                nargout=1,
            )
        )
        try:
            contact_spacing = json.loads(contact_spacing_raw)
        except Exception:
            contact_spacing = contact_spacing_raw
        contactnames = json_list(
            str(
                eng.eval(
                    "jsonencode(cellfun(@char, lfptp_opts.elspec.contactnames, 'UniformOutput', false));",
                    nargout=1,
                )
            )
        )
        return {
            "elmodel": model_key,
            "matfname": matfname,
            "contact_length": contact_length,
            "contact_spacing": contact_spacing,
            "contactnames": contactnames,
        }

    future = svc.submit_matlab_task(paths, _task)
    try:
        payload = future.result(timeout=svc._MATLAB_TASK_TIMEOUT_S)
    except FutureTimeout as exc:
        message = f"MATLAB request timed out after {int(svc._MATLAB_TASK_TIMEOUT_S)}s."
        svc._set_matlab_runtime_status("failed", message)
        raise RuntimeError(message) from exc

    svc._ELSPEC_CACHE[model_key] = payload
    return payload


def split_contactnames(
    contactnames: list[str],
) -> tuple[list[str], list[str], list[str]]:
    right: list[str] = []
    left: list[str] = []
    other: list[str] = []
    for name in contactnames:
        token = str(name).strip()
        if not token:
            continue
        match = re.search(r"\((R|L)\)\s*$", token, flags=re.IGNORECASE)
        if match is None:
            other.append(token)
            continue
        side = match.group(1).upper()
        if side == "R":
            right.append(token)
        else:
            left.append(token)
    return right, left, other


def fallback_contactnames(
    template: list[str], lead_label: str, n_contacts: int
) -> list[str]:
    out: list[str] = []
    for base in template:
        token = str(base).strip()
        if not token:
            continue
        if re.search(r"\((R|L)\)\s*$", token, flags=re.IGNORECASE):
            token = re.sub(
                r"\((R|L)\)\s*$",
                f"({lead_label})",
                token,
                flags=re.IGNORECASE,
            )
        elif f"({lead_label})" not in token:
            token = f"{token} ({lead_label})"
        out.append(token)
    if len(out) > n_contacts:
        out = out[:n_contacts]
    while len(out) < n_contacts:
        out.append(f"K{len(out)} ({lead_label})")
    return out


def load_reconstruction_contacts(
    project_root: Path,
    subject: str,
    paths: Any,
) -> tuple[bool, str, dict[str, Any]]:
    """Load per-lead contact names and native/mni coordinates from reconstruction."""
    recon_path = reconstruction_mat_path(project_root, subject)
    if not recon_path.is_file():
        return False, f"Missing reconstruction file: {recon_path}", {}

    try:
        payload = scipy.io.loadmat(recon_path, squeeze_me=False, struct_as_record=False)
    except Exception as exc:  # noqa: BLE001
        return False, f"Failed to read reconstruction mat: {exc}", {}

    reco_raw = payload.get("reco")
    if not isinstance(reco_raw, np.ndarray) or reco_raw.size == 0:
        return False, "Invalid reconstruction mat: missing reco struct.", {}
    reco = unwrap_matlab_node(reco_raw)
    if reco is None:
        return False, "Invalid reconstruction mat: empty reco struct.", {}

    native_coords = extract_coords_by_space(reco, "native")
    mni_coords = extract_coords_by_space(reco, "mni")
    if not native_coords or not mni_coords:
        return False, "Reconstruction missing native/mni contact coordinates.", {}

    n_leads = min(len(native_coords), len(mni_coords))
    native_coords = native_coords[:n_leads]
    mni_coords = mni_coords[:n_leads]

    elmodels = extract_elmodels(reco, n_leads)
    leads: list[dict[str, Any]] = []

    for lead_idx in range(n_leads):
        lead_label = ordinal_label(lead_idx + 1)
        elmodel = elmodels[lead_idx] if lead_idx < len(elmodels) else "Unknown"
        n_contacts = int(
            min(native_coords[lead_idx].shape[0], mni_coords[lead_idx].shape[0])
        )
        if n_contacts <= 0:
            contacts: list[dict[str, Any]] = []
            leads.append(
                {
                    "lead_index": int(lead_idx + 1),
                    "lead_label": lead_label,
                    "elmodel": elmodel,
                    "display_name": f"{elmodel} {lead_label}",
                    "contacts": contacts,
                }
            )
            continue

        try:
            elspec = resolve_elspec(paths, elmodel)
            all_names = list(elspec.get("contactnames", []))
        except Exception:
            all_names = []

        right_names, left_names, other_names = split_contactnames(all_names)
        if lead_label == "R" and right_names:
            lead_names = fallback_contactnames(right_names, lead_label, n_contacts)
        elif lead_label == "L" and left_names:
            lead_names = fallback_contactnames(left_names, lead_label, n_contacts)
        else:
            template = right_names or left_names or other_names
            lead_names = fallback_contactnames(template, lead_label, n_contacts)

        contacts = []
        for ci in range(n_contacts):
            name = lead_names[ci]
            token = f"{lead_label}_{name}"
            native_xyz = native_coords[lead_idx][ci, :].astype(float)
            mni_xyz = mni_coords[lead_idx][ci, :].astype(float)
            contacts.append(
                {
                    "contact_index": int(ci),
                    "contact_name": name,
                    "token": token,
                    "native": [
                        float(native_xyz[0]),
                        float(native_xyz[1]),
                        float(native_xyz[2]),
                    ],
                    "mni": [float(mni_xyz[0]), float(mni_xyz[1]), float(mni_xyz[2])],
                }
            )

        leads.append(
            {
                "lead_index": int(lead_idx + 1),
                "lead_label": lead_label,
                "elmodel": elmodel,
                "display_name": f"{elmodel} {lead_label}",
                "contacts": contacts,
            }
        )

    summary = {
        "subject": subject,
        "reconstruction_path": str(recon_path),
        "n_leads": len(leads),
        "leads": leads,
    }
    return True, "", summary
