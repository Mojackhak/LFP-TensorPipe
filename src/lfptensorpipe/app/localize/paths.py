"""Localize path/discovery helpers."""

from __future__ import annotations

from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

from lfptensorpipe.app.config_store import AppConfigStore
from .config import moved_representative_channels
from lfptensorpipe.app.runlog_store import read_run_log
from lfptensorpipe.app.shared.generation_lineage import input_generation_receipts_match
from lfptensorpipe.app.shared.runlog_store import cache_in_run_log_read_snapshot
from lfptensorpipe.matlab import infer_matlab_root

if TYPE_CHECKING:
    from .service import LocalizePaths


def record_channel_fif(context):
    """Locate the acquired channels in an imported record or an isolated derivative."""
    from lfptensorpipe.app.path_resolver import PathResolver
    from lfptensorpipe.app.preproc.paths import (
        rawdata_input_fif_path,
        preproc_step_raw_path,
    )

    path = rawdata_input_fif_path(context)
    return (
        path
        if path.is_file()
        else preproc_step_raw_path(PathResolver(context), "raw", create=False)
    )


def normalize_localize_path_payload(payload: Any) -> tuple[dict[str, Any], bool]:
    """Normalize `paths.yml` payload and migrate legacy MATLAB keys."""
    source = payload if isinstance(payload, dict) else {}
    normalized = dict(source)

    leaddbs_raw = str(source.get("leaddbs_dir", "")).strip()
    matlab_root_raw = str(source.get("matlab_root", "")).strip()
    legacy_raw = str(source.get("matlab_engine_path", "")).strip()

    if not matlab_root_raw and legacy_raw:
        inferred_root = infer_matlab_root(Path(legacy_raw).expanduser())
        matlab_root_raw = (
            str(inferred_root) if inferred_root is not None else legacy_raw
        )

    normalized["leaddbs_dir"] = leaddbs_raw
    normalized["matlab_root"] = matlab_root_raw
    normalized.pop("matlab_engine_path", None)
    changed = normalized != source
    return normalized, changed


def load_localize_paths(config_store: AppConfigStore) -> "LocalizePaths":
    """Read localize runtime paths from app settings storage."""
    from . import service as svc

    payload = config_store.read_yaml("paths.yml", default={})
    normalized_payload, changed = normalize_localize_path_payload(payload)
    if changed:
        config_store.write_yaml("paths.yml", normalized_payload)

    leaddbs_raw = str(normalized_payload.get("leaddbs_dir", "")).strip()
    matlab_raw = str(normalized_payload.get("matlab_root", "")).strip()

    leaddbs_dir = (
        Path(leaddbs_raw).expanduser()
        if leaddbs_raw
        else Path("__missing_leaddbs_dir__")
    )
    matlab_root = (
        Path(matlab_raw).expanduser() if matlab_raw else Path("__missing_matlab_root__")
    )
    return svc.LocalizePaths(
        leaddbs_dir=leaddbs_dir,
        matlab_root=matlab_root,
    )


def discover_spaces(leaddbs_dir: Path) -> list[str]:
    """Discover Lead-DBS spaces from `{leaddbs_dir}/templates/space/`."""
    base = leaddbs_dir / "templates" / "space"
    if not base.exists():
        return []
    return sorted(path.name for path in base.iterdir() if path.is_dir())


def discover_atlases(leaddbs_dir: Path, space: str) -> list[str]:
    """Discover atlases from `{leaddbs_dir}/templates/space/{space}/atlases/`."""
    base = leaddbs_dir / "templates" / "space" / space / "atlases"
    if not base.exists():
        return []
    return sorted(path.name for path in base.iterdir() if path.is_dir())


def reconstruction_root(project_root: Path, subject: str) -> Path:
    return project_root / "derivatives" / "leaddbs" / subject / "reconstruction"


def reconstruction_mat_path(project_root: Path, subject: str) -> Path:
    return (
        reconstruction_root(project_root, subject)
        / f"{subject}_desc-reconstruction.mat"
    )


def has_reconstruction_mat(project_root: Path, subject: str) -> bool:
    return reconstruction_mat_path(project_root, subject).is_file()


def localize_record_dir(project_root: Path, subject: str, record: str) -> Path:
    return (
        project_root / "derivatives" / "lfptensorpipe" / subject / record / "localize"
    )


def localize_representative_csv_path(
    project_root: Path, subject: str, record: str
) -> Path:
    return (
        localize_record_dir(project_root, subject, record)
        / "channel_representative_coords.csv"
    )


def localize_representative_pkl_path(
    project_root: Path, subject: str, record: str
) -> Path:
    return (
        localize_record_dir(project_root, subject, record)
        / "channel_representative_coords.pkl"
    )


def localize_ordered_pair_representative_csv_path(
    project_root: Path, subject: str, record: str
) -> Path:
    return (
        localize_record_dir(project_root, subject, record)
        / "channel_pair_ordered_representative_coords.csv"
    )


def localize_ordered_pair_representative_pkl_path(
    project_root: Path, subject: str, record: str
) -> Path:
    return (
        localize_record_dir(project_root, subject, record)
        / "channel_pair_ordered_representative_coords.pkl"
    )


def localize_undirected_pair_representative_csv_path(
    project_root: Path, subject: str, record: str
) -> Path:
    return (
        localize_record_dir(project_root, subject, record)
        / "channel_pair_undirected_representative_coords.csv"
    )


def localize_undirected_pair_representative_pkl_path(
    project_root: Path, subject: str, record: str
) -> Path:
    return (
        localize_record_dir(project_root, subject, record)
        / "channel_pair_undirected_representative_coords.pkl"
    )


def localize_log_path(project_root: Path, subject: str, record: str) -> Path:
    """Return Localize log path for one subject+record."""
    return localize_record_dir(project_root, subject, record) / "lfptensorpipe_log.json"


LEAD_DBS = "leaddbs"
USER_LABELS = "user_labels"
TABLE_STEMS = (
    "channel_representative_coords",
    "channel_pair_ordered_representative_coords",
    "channel_pair_undirected_representative_coords",
)


def source_directory(root: Path, source: str) -> Path:
    if source == USER_LABELS:
        return root / USER_LABELS
    if source == LEAD_DBS:
        return root
    raise ValueError(f"Unknown Localize source: {source!r}")


def source_results(payload: dict | None) -> dict[str, dict]:
    """Accepted receipts, including the single-source v0.3.0 writer."""
    if not payload:
        return {}
    state = payload.get("state", {})
    if not isinstance(state, dict):
        return {}
    if "source_results" in state:
        stored = state["source_results"]
        if not isinstance(stored, dict):
            return {}
        return {
            name: params
            for name, params in stored.items()
            if name in (LEAD_DBS, USER_LABELS)
            and input_generation_receipts_match(
                {"completed": True, "params": params}, expected={}
            )
        }
    if payload.get("completed") and input_generation_receipts_match(
        payload, expected={}
    ):
        return {
            LEAD_DBS: {**payload["params"], "computed_at": payload.get("timestamp_utc")}
        }
    return {}


def applied_source(payload: dict | None) -> str | None:
    if not payload:
        return None
    state = payload.get("state", {})
    if not isinstance(state, dict):
        return None
    active = state.get("active_source", LEAD_DBS)
    if active not in (LEAD_DBS, USER_LABELS):
        return None
    return active if active in source_results(payload) else None


def source_files_ready(root: Path, source: str) -> bool:
    directory = source_directory(root, source)
    return all(
        (directory / (stem + ext)).is_file()
        for stem in TABLE_STEMS
        for ext in (".pkl", ".csv")
    )


@cache_in_run_log_read_snapshot(
    lambda project_root, subject, record: (project_root, subject, record)
)
def localize_indicator_state(project_root: Path, subject: str, record: str) -> str:
    """Derive Localize indicator state from record-level localize log."""
    try:
        payload = read_run_log(localize_log_path(project_root, subject, record))
    except (OSError, ValueError):
        return "yellow"
    if payload is None:
        return "gray"
    return (
        "green"
        if _localize_result_ready(project_root, subject, record, payload)
        else "yellow"
    )


def _localize_result_ready(
    project_root: Path, subject: str, record: str, payload: dict[str, Any]
) -> bool:
    active = applied_source(payload)
    if active is None:
        return False
    return source_files_ready(
        localize_record_dir(project_root, subject, record), active
    )


def localize_result_generation_id(
    project_root: Path,
    subject: str,
    record: str,
) -> str | None:
    """Return the current Localize accepted generation ID, if present."""
    if localize_indicator_state(project_root, subject, record) != "green":
        return None
    try:
        payload = read_run_log(localize_log_path(project_root, subject, record))
    except Exception:
        return None
    active = applied_source(payload)
    if active is None:
        return None
    return source_results(payload)[active]["result_generation_id"]


def localize_match_signature(
    payload: dict[str, Any] | None,
) -> list[dict[str, str]] | None:
    """Return a stable mapping signature for Localize Match payloads."""
    if not isinstance(payload, dict) or not bool(payload.get("completed", False)):
        return None
    mappings_raw = payload.get("mappings")
    if not isinstance(mappings_raw, list):
        return None
    normalized: list[dict[str, str]] = []
    for item in mappings_raw:
        if not isinstance(item, dict):
            continue
        channel = str(item.get("channel", "")).strip()
        anode = str(item.get("anode", "")).strip()
        cathode = str(item.get("cathode", "")).strip()
        rep_coord = str(item.get("rep_coord", "Mid")).strip().title()
        if rep_coord not in {"Anode", "Cathode", "Mid"}:
            rep_coord = "Mid"
        if not channel or not anode or not cathode:
            return None
        normalized.append(
            {
                "channel": channel,
                "anode": anode,
                "cathode": cathode,
                "rep_coord": rep_coord,
            }
        )
    if not normalized:
        return None
    normalized.sort(key=lambda row: row["channel"])
    return normalized


def localize_selected_regions_signature(value: Any) -> list[str] | None:
    """Return a stable interested-region signature for Localize atlas config."""
    if not isinstance(value, list | tuple):
        return None
    seen: set[str] = set()
    out: list[str] = []
    for raw_name in value:
        name = str(raw_name).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    if not out:
        return None
    out.sort(key=lambda item: (item.casefold(), item))
    return out


def _normalized_atlas(value: Any) -> str | None:
    atlas = str(value).strip()
    return atlas or None


def localize_panel_state(
    project_root: Path,
    subject: str,
    record: str,
    *,
    atlas: Any,
    selected_regions: Any,
    match_payload: dict[str, Any] | None,
    source: str | None = None,
    label_signature: dict | None = None,
) -> str:
    """Return `gray|yellow|green` for the inline editable Localize panel light."""
    try:
        payload = read_run_log(localize_log_path(project_root, subject, record))
    except (OSError, ValueError):
        return "yellow"
    if payload is None:
        return "gray"
    completed = payload.get("completed")
    if completed is False:
        return "yellow"
    if completed is not True:
        return "gray"
    if not _localize_result_ready(project_root, subject, record, payload):
        return "yellow"
    active = applied_source(payload)
    if source is not None and source != active:
        return "yellow"
    params = source_results(payload).get(active)
    if active == USER_LABELS:
        return (
            "green"
            if label_signature is None
            or params.get("label_signature") == label_signature
            else "yellow"
        )
    if not isinstance(params, dict):
        return "green"
    logged_atlas = _normalized_atlas(params.get("atlas"))
    logged_regions_signature = localize_selected_regions_signature(
        params.get("selected_regions_signature")
    )
    logged_match_signature = params.get("match_signature")
    if (
        not isinstance(logged_match_signature, list)
        or logged_atlas is None
        or logged_regions_signature is None
    ):
        return "yellow"
    current_atlas = _normalized_atlas(atlas)
    current_regions_signature = localize_selected_regions_signature(selected_regions)
    current_match_signature = localize_match_signature(match_payload)
    if (
        current_atlas is None
        or current_regions_signature is None
        or current_match_signature is None
    ):
        return "yellow"
    return (
        "green"
        if (
            current_atlas == logged_atlas
            and current_regions_signature == logged_regions_signature
            and current_match_signature == logged_match_signature
            and not moved_representative_channels(
                current_match_signature,
                params.get("lead_signature"),
                match_payload.get("lead_signature"),
            )
        )
        else "yellow"
    )


def localize_csv_path(project_root: Path, subject: str) -> Path:
    """Return legacy Localize midpoint CSV artifact path under Lead-DBS reconstruction."""
    return reconstruction_root(project_root, subject) / "contact_midpoint_coords.csv"


def localize_mat_path(project_root: Path, subject: str) -> Path:
    """Return legacy Localize midpoint MAT artifact path under Lead-DBS reconstruction."""
    return reconstruction_root(project_root, subject) / "contact_midpoint_coords.mat"


def infer_subject_spaces(project_root: Path, subject: str) -> list[str]:
    """Infer available spaces for one subject using required priority rules."""
    trans_dir = (
        project_root
        / "derivatives"
        / "leaddbs"
        / subject
        / "normalization"
        / "transformations"
    )
    spaces_a: set[str] = set()
    if trans_dir.is_dir():
        pattern = re.compile(
            rf"^{re.escape(subject)}_from-anchorNative_to-(.+?)_desc-ants\.nii\.gz$"
        )
        for item in trans_dir.iterdir():
            if not item.is_file():
                continue
            match = pattern.match(item.name)
            if match:
                spaces_a.add(match.group(1).strip())
    if spaces_a:
        return sorted(space for space in spaces_a if space)

    anat_dir = (
        project_root / "derivatives" / "leaddbs" / subject / "normalization" / "anat"
    )
    spaces_b: set[str] = set()
    if anat_dir.is_dir():
        pattern = re.compile(
            rf"^{re.escape(subject)}_.*_space-([^_]+)_.*\.nii(?:\.gz)?$"
        )
        for item in anat_dir.iterdir():
            if not item.is_file():
                continue
            match = pattern.match(item.name)
            if match:
                spaces_b.add(match.group(1).strip())
    return sorted(space for space in spaces_b if space)


def infer_subject_space(project_root: Path, subject: str) -> tuple[str | None, str]:
    """Infer one unique space for one subject."""
    spaces = infer_subject_spaces(project_root, subject)
    if not spaces:
        return None, "No valid space discovered from subject normalization files."
    if len(spaces) > 1:
        return None, f"Multiple spaces discovered: {', '.join(spaces)}"
    return spaces[0], ""


__all__ = [
    "load_localize_paths",
    "discover_spaces",
    "discover_atlases",
    "reconstruction_root",
    "reconstruction_mat_path",
    "has_reconstruction_mat",
    "localize_record_dir",
    "localize_representative_csv_path",
    "localize_representative_pkl_path",
    "localize_ordered_pair_representative_csv_path",
    "localize_ordered_pair_representative_pkl_path",
    "localize_undirected_pair_representative_csv_path",
    "localize_undirected_pair_representative_pkl_path",
    "localize_log_path",
    "localize_indicator_state",
    "localize_result_generation_id",
    "localize_match_signature",
    "localize_panel_state",
    "localize_csv_path",
    "localize_mat_path",
    "infer_subject_spaces",
    "infer_subject_space",
]
