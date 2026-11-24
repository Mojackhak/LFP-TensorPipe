"""Inputs governing reuse of Lead-DBS coordinates and atlas membership."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

from . import atlas_lookup
from .paths import reconstruction_mat_path


def electrode_model_source(paths) -> tuple[str, int | None]:
    """Identify the configured Lead-DBS function that supplies model contact names."""
    resolver = (
        paths.leaddbs_dir / "templates/electrode_models/ea_resolve_elspec.m"
    ).resolve()
    return str(resolver), resolver.stat().st_mtime_ns if resolver.is_file() else None


def cached_lead_signature(saved: dict, reconstruction: Path, paths) -> list | None:
    """Reuse an identity established from these reconstruction and model inputs."""
    source = saved.get("lead_signature_elspec")
    if not source or not reconstruction.is_file():
        return None
    stamp = saved.get("geometry_inputs", {}).get("files", {}).get(str(reconstruction))
    model_source = electrode_model_source(paths)
    if (
        model_source[1] is None
        or stamp != reconstruction.stat().st_mtime_ns
        or list(model_source) != source
    ):
        return None
    return saved.get("lead_signature")


def _file_times(files):
    return {str(p): p.stat().st_mtime_ns for p in sorted(files) if p.is_file()}


def _json_object(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path.name}.")
    return value


def lead_inputs(
    project: Path,
    subject: str,
    space: str,
    atlas: str,
    regions,
    paths,
    match,
    *,
    saved,
    preview=False,
):
    """Observe only geometry and selected mask inputs; parse configuration by value."""
    reco = reconstruction_mat_path(project, subject)
    files = [reco]
    method = ""
    selected_anchor = None
    reference_images = None
    if any(row["rep_coord"] == "Mid" for row in match):
        base = reco.parent.parent
        method_path = base / "normalization" / "log" / f"{subject}_desc-normmethod.json"
        if method_path.is_file():
            name = _json_object(method_path).get("method")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"Expected a normalization method in {method_path.name}."
                )
            method = name.strip().lower()
        transforms = base / "normalization" / "transformations"
        prefix = f"{subject}_from-{space}_to-anchorNative_desc-"
        if "fsl" in method or "fnirt" in method:
            warp = transforms / (prefix + "fnirt.nii")
            files.append(paths.leaddbs_dir / "templates" / "space" / space / "t2.nii")
            raw_images = base / "prefs" / f"{subject}_desc-rawimages.json"
            if raw_images.is_file():
                preop = _json_object(raw_images).get("preop", {})
                if not isinstance(preop, dict):
                    raise ValueError(f"Expected a preop object in {raw_images.name}.")
                reference_images = preop.get("anat")
            selected_anchor = saved.get("geometry_inputs", {}).get("anchor_file")
            if selected_anchor:
                files.append(Path(selected_anchor))
        else:
            warp = transforms / (
                prefix + ("ants.mat" if "affine" in method else "ants.nii.gz")
            )
            if "spm" in method:
                files.append(transforms / (prefix + "spm.nii"))
        files.append(
            base
            / "brainshift"
            / "transformations"
            / f"{subject}_from-anchorNative_to-anchorNativeBSC_desc-scrf.mat"
        )
        files.append(warp)
    geometry = {
        "space": space,
        "method": method,
        "files": _file_times(files),
        "anchor_file": selected_anchor,
        "preop_images": reference_images,
    }
    atlas_path = paths.leaddbs_dir / "templates" / "space" / space / "atlases" / atlas
    selected = [
        side / (name + ext)
        for side in atlas_path.iterdir()
        if side.is_dir()
        for name in regions
        for ext in (".nii", ".nii.gz")
    ]
    index = atlas_path / "atlas_index.mat"
    stat = index.stat()
    index_stamp = [str(index), stat.st_mtime_ns, stat.st_size]
    prior_masks = saved.get("region_inputs", {})
    if index_stamp == prior_masks.get("index_stamp"):
        threshold = prior_masks["threshold"]
    elif preview and not prior_masks:
        # A v0.3 receipt can only compare input dates. Reading the full atlas
        # index while browsing the record supplies no additional evidence.
        threshold = None
    else:
        threshold = atlas_lookup.read_atlas_threshold(index)
    membership = {
        "atlas": atlas,
        "threshold": threshold,
        "index_stamp": index_stamp,
        "files": _file_times(selected),
    }
    return geometry, membership


def inputs_match(
    saved: dict,
    key: str,
    current: dict,
    legacy_time: str | None,
    config_path: Path | None = None,
) -> bool:
    if key in saved:
        if key == "region_inputs":
            return all(
                saved[key].get(k) == current[k] for k in ("atlas", "threshold", "files")
            )
        return saved[key] == current
    # The v0.3 writer recorded a generation and computation time, but did
    # not retain individual resource stamps. Reject inputs newer than that run.
    if not legacy_time or not current["files"]:
        return False
    end_of_second = (
        int(datetime.fromisoformat(legacy_time.replace("Z", "+00:00")).timestamp()) + 1
    )
    stamps = list(current["files"].values())
    if config_path is not None and config_path.is_file():
        stamps.append(config_path.stat().st_mtime_ns)
    return all(stamp < end_of_second * 1_000_000_000 for stamp in stamps)


def compare_lead_inputs(
    saved, project, subject, space, atlas, regions, paths, mapping, *, preview=False
):
    geometry, masks = lead_inputs(
        project,
        subject,
        space,
        atlas,
        regions,
        paths,
        mapping,
        saved=saved,
        preview=preview,
    )
    when = saved.get("computed_at")
    norm_config = (
        reconstruction_mat_path(project, subject).parent.parent
        / "normalization"
        / "log"
        / f"{subject}_desc-normmethod.json"
    )
    index = (
        paths.leaddbs_dir
        / "templates"
        / "space"
        / space
        / "atlases"
        / atlas
        / "atlas_index.mat"
    )
    same_geometry = saved.get("space") == space and inputs_match(
        saved,
        "geometry_inputs",
        geometry,
        when,
        norm_config if any(r["rep_coord"] == "Mid" for r in mapping) else None,
    )
    if "geometry_inputs" not in saved:
        files = geometry["files"]
        if str(reconstruction_mat_path(project, subject)) not in files:
            same_geometry = False
        if any(r["rep_coord"] == "Mid" for r in mapping):
            has_warp = any(
                "/normalization/transformations/" in p.replace("\\", "/") for p in files
            )
            same_geometry = same_geometry and has_warp and bool(geometry["method"])
            if "fsl" in geometry["method"] or "fnirt" in geometry["method"]:
                same_geometry = (
                    False  # The v0.3 receipt has no FSL anchor image identity.
                )
    same_masks = (
        saved.get("atlas") == atlas
        and saved.get("selected_regions_signature") == regions
        and inputs_match(saved, "region_inputs", masks, when, index)
    )
    return geometry, masks, same_geometry, same_masks
