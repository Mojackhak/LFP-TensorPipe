"""Select existing Feature outputs that can serve an edit to the phase table."""

from dataclasses import asdict
from pathlib import Path

from lfptensorpipe.app.shared.generation_lineage import parse_generation_lineage
from . import generation
from .indicator import _normalize_axes_by_metric
from .phase_time import same_phase_windows


def collapse_settings(config):
    settings = dict(config)
    if "out_cols" in settings:
        settings["out_cols"] = asdict(settings["out_cols"])
    return settings


def _reduction_options(metric, stem, axes, settings, semantics, method):
    """Select the intervals, settings and semantics consumed by this output."""
    kind = stem.rsplit("-", 1)[-1]
    if kind == "raw":
        return {}
    if metric == "burst":
        return {"burst_support": semantics.get(generation.BURST_SAMPLE_SUPPORT_KEY)}
    options = {
        key: settings.get(key)
        for key in (
            "value_col",
            "out_cols",
            "inclusive",
            "keep_full_dim_cols",
            "drop_value",
            "on_missing",
        )
    }
    if kind in {"trace", "scalar"}:
        options.update(
            bands=axes["bands"], freq_interval_mode=settings.get("freq_interval_mode")
        )
    if kind in {"spectral", "scalar"}:
        options["time_interval_mode"] = (
            "percent" if "time_unit" in axes else settings.get("time_interval_mode")
        )
    if stem.startswith("mean-"):
        options["mean"] = semantics.get(generation.NUMERIC_MEAN_SEMANTICS_KEY)
        if method in {"pad_warper", "concat_warper"} and kind in {"spectral", "scalar"}:
            options["piecewise_mean"] = semantics.get(
                generation.CLIP_STITCH_MEAN_SUPPORT_KEY
            )
    return options


def reusable_axis_outputs(
    prior,
    *,
    slug,
    alignment_method,
    paths_by_metric,
    axes,
    settings,
    input_generations,
    duration,
):
    """Retain or relabel only outputs whose numerical inputs match their receipt."""
    keep = set()
    relabel = set()
    lineage = parse_generation_lineage(prior)
    if lineage is None or not prior.get("completed"):
        return keep, relabel
    params = prior["params"]
    old_settings = params.get("collapse_settings")
    previous = _normalize_axes_by_metric(params.get("axes_by_metric"))
    if old_settings is None or previous is None:
        return keep, relabel
    if params.get("alignment_method") != alignment_method:
        return keep, relabel
    manifest = generation.outputs_from_features_entry(prior)
    if manifest is None:
        return keep, relabel
    current_semantics = {
        generation.NUMERIC_MEAN_SEMANTICS_KEY: generation.NUMERIC_MEAN_SEMANTICS,
        generation.CLIP_STITCH_MEAN_SUPPORT_KEY: generation.CLIP_STITCH_MEAN_SUPPORT,
        generation.BURST_SAMPLE_SUPPORT_KEY: generation.BURST_SAMPLE_SUPPORT,
    }
    finish_ref = generation.alignment_generation_ref(slug, "build_raw_table")
    for metric, paths in paths_by_metric.items():
        if metric not in previous or metric not in axes:
            continue
        old, new = previous[metric], axes[metric]
        same_time = same_phase_windows(old, new, duration)
        for stem, path in paths.items():
            if (
                Path(metric, path.name) not in manifest.get(metric, ())
                or not path.is_file()
            ):
                continue
            refs = [finish_ref]
            if generation._native_feature_tensor_dependencies(
                alignment_method=alignment_method, outputs_by_metric={metric: [path]}
            ):
                refs.extend(
                    (
                        generation.alignment_generation_ref(slug, "run_align_epochs"),
                        generation.tensor_generation_ref(metric),
                    )
                )
            if any(
                ref not in lineage.input_generations
                or ref not in input_generations
                or lineage.input_generations[ref] != input_generations[ref]
                for ref in refs
            ):
                continue
            kind = stem.rsplit("-", 1)[-1]
            uses_time = kind in {"spectral", "scalar"}
            if uses_time and not same_time:
                continue
            if _reduction_options(
                metric, stem, old, old_settings, params, alignment_method
            ) != _reduction_options(
                metric, stem, new, settings, current_semantics, alignment_method
            ):
                continue
            if uses_time and (old["times"], old.get("time_unit")) != (
                new["times"],
                new.get("time_unit"),
            ):
                relabel.add(path)
            else:
                keep.add(path)
    return keep, relabel
