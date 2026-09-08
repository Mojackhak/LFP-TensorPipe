"""Compute local PSI from the shared complex-coherency spectral backend.

Both Morlet and Multitaper use centered spectral averaging. Frequencies within
one original band share the longest required averaging interval and padding,
while retaining frequency-specific kernels. Notch-separated segments contribute
only their own adjacent-frequency products. Output is signed, unnormalized PSI
with shape (1, n_pairs, n_bands, n_times).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple
import warnings

import numpy as np
import mne
from joblib import Parallel, delayed

from ..common.timefreq import (
    compute_decimation,
    decimated_times_from_raw,
)
from ..connectivity.grid import grid as connectivity_grid
from ..runtime.tensor_helpers import (
    apply_dynamic_edge_mask_strict,
    build_annotation_skip_time_mask,
    connectivity_consumed_support_radii_seconds,
)
from . import PSI_COHERENCY_ESTIMATION

BandValue = Tuple[float, float]
BandValueOrSegments = BandValue | list[BandValue]
BandSpec = Mapping[str, BandValueOrSegments]


def _normalize_bands(
    bands: BandSpec,
) -> tuple[list[str], list[list[BandValue]], np.ndarray, np.ndarray, np.ndarray]:
    """Normalize band specs.

    Supports either:
      - name -> (fmin,fmax)
      - name -> [(fmin,fmax), (fmin2,fmax2), ...]

    Returns:
        band_names: list[str]
        band_segments: list[list[(fmin,fmax)]] aligned with band_names
        seg_edges: array (n_segments,2)
        seg_to_band: array (n_segments,) with band indices
        band_union_edges: array (n_bands,2) with (min_lo,max_hi) per band
    """
    if not isinstance(bands, Mapping) or len(bands) == 0:
        raise ValueError(
            "`bands` must be a non-empty mapping name -> (fmin,fmax) or list[(fmin,fmax)]."
        )

    band_names: list[str] = []
    band_segments: list[list[BandValue]] = []

    seg_edges: list[tuple[float, float]] = []
    seg_to_band: list[int] = []
    union_edges: list[tuple[float, float]] = []

    for band_i, (name, spec) in enumerate(bands.items()):
        bname = str(name)
        if (
            isinstance(spec, (tuple, list))
            and len(spec) == 2
            and not isinstance(spec[0], (tuple, list))
        ):
            segments: list[BandValue] = [(float(spec[0]), float(spec[1]))]  # type: ignore[arg-type]
        elif isinstance(spec, list):
            segments = [(float(a), float(b)) for (a, b) in spec]
        else:
            raise ValueError(
                f"Band '{bname}' must be (fmin,fmax) or list[(fmin,fmax)]."
            )

        if len(segments) == 0:
            raise ValueError(f"Band '{bname}' has no segments.")

        for a, b in segments:
            if not np.isfinite(a) or not np.isfinite(b):
                raise ValueError(f"Band '{bname}' has non-finite bounds: {(a, b)}")
            if a <= 0 or b <= 0:
                raise ValueError(f"Band '{bname}' bounds must be > 0 Hz, got {(a, b)}")
            if b <= a:
                raise ValueError(
                    f"Band '{bname}' must satisfy fmax > fmin, got {(a, b)}"
                )
            seg_edges.append((a, b))
            seg_to_band.append(band_i)

        lo = min(a for a, _ in segments)
        hi = max(b for _, b in segments)
        union_edges.append((lo, hi))

        band_names.append(bname)
        band_segments.append(segments)

    seg_edges_arr = np.asarray(seg_edges, dtype=float)
    seg_to_band_arr = np.asarray(seg_to_band, dtype=int)
    union_edges_arr = np.asarray(union_edges, dtype=float)
    return band_names, band_segments, seg_edges_arr, seg_to_band_arr, union_edges_arr


def _segment_frequency_support(
    seg_edges: np.ndarray,
    frequencies_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Count strict-internal frequencies and adjacent PSI terms per segment."""
    freqs = np.asarray(frequencies_hz, dtype=float)
    counts = np.asarray(
        [
            int(np.sum((freqs > float(low)) & (freqs < float(high))))
            for low, high in np.asarray(seg_edges, dtype=float)
        ],
        dtype=int,
    )
    return counts, np.maximum(counts - 1, 0)


def _validate_band_frequency_support(
    *,
    method: str,
    band_names: list[str],
    seg_edges: np.ndarray,
    seg_to_band: np.ndarray,
    n_internal_frequencies: np.ndarray,
    n_adjacent_pairs: np.ndarray,
) -> None:
    unsupported_bands: list[str] = []
    for band_i, band_name in enumerate(band_names):
        segment_indices = np.flatnonzero(seg_to_band == band_i)
        if bool(np.any(n_adjacent_pairs[segment_indices] > 0)):
            continue
        segment_details = ", ".join(
            f"[{float(seg_edges[index, 0]):g}, {float(seg_edges[index, 1]):g}] Hz: "
            f"{int(n_internal_frequencies[index])} internal frequencies, "
            f"{int(n_adjacent_pairs[index])} adjacent pairs"
            for index in segment_indices
        )
        unsupported_bands.append(f"'{band_name}' ({segment_details})")

    if not unsupported_bands:
        return
    advice = "Decrease Step (Hz) or widen the band segment."
    raise ValueError(
        f"{method.capitalize()} PSI requires at least two frequencies strictly "
        "inside at least one segment of every band. Unsupported band(s): "
        + "; ".join(unsupported_bands)
        + f". {advice}"
    )


def _partial_frequency_support_message(
    *,
    band_names: list[str],
    seg_edges: np.ndarray,
    seg_to_band: np.ndarray,
    n_internal_frequencies: np.ndarray,
    n_adjacent_pairs: np.ndarray,
) -> str | None:
    dropped: list[str] = []
    for segment_index in np.flatnonzero(n_adjacent_pairs == 0):
        band_name = band_names[int(seg_to_band[segment_index])]
        dropped.append(
            f"'{band_name}' [{float(seg_edges[segment_index, 0]):g}, "
            f"{float(seg_edges[segment_index, 1]):g}] Hz "
            f"({int(n_internal_frequencies[segment_index])} internal frequencies, "
            f"{int(n_adjacent_pairs[segment_index])} adjacent pairs)"
        )
    if not dropped:
        return None
    return (
        "PSI uses partial frequency support; dropped segment(s) without an "
        "adjacent frequency pair: " + "; ".join(dropped) + "."
    )


def _frequency_support_by_band(
    *,
    band_names: list[str],
    seg_edges: np.ndarray,
    seg_to_band: np.ndarray,
    n_internal_frequencies: np.ndarray,
    n_adjacent_pairs: np.ndarray,
) -> dict[str, dict[str, list[Any]]]:
    support: dict[str, dict[str, list[Any]]] = {}
    for band_i, band_name in enumerate(band_names):
        segment_indices = np.flatnonzero(seg_to_band == band_i)
        configured = [
            [float(seg_edges[index, 0]), float(seg_edges[index, 1])]
            for index in segment_indices
        ]
        used = [
            [float(seg_edges[index, 0]), float(seg_edges[index, 1])]
            for index in segment_indices
            if int(n_adjacent_pairs[index]) > 0
        ]
        dropped = [
            [float(seg_edges[index, 0]), float(seg_edges[index, 1])]
            for index in segment_indices
            if int(n_adjacent_pairs[index]) == 0
        ]
        support[str(band_name)] = {
            "configured_segments_hz": configured,
            "used_segments_hz": used,
            "dropped_segments_hz": dropped,
            "n_internal_frequencies_by_segment": [
                int(n_internal_frequencies[index]) for index in segment_indices
            ],
            "n_adjacent_pairs_by_segment": [
                int(n_adjacent_pairs[index]) for index in segment_indices
            ],
        }
    return support


def _default_freqs(band_edges: np.ndarray) -> np.ndarray:
    """Create a simple 1-Hz grid covering all bands."""
    fmin_all = float(np.min(band_edges[:, 0]))
    fmax_all = float(np.max(band_edges[:, 1]))

    # 1 Hz steps are usually fine for typical neuroscience bands (delta..gamma).
    start = max(0.1, np.floor(fmin_all))
    stop = np.ceil(fmax_all)
    freqs = np.arange(start, stop + 1.0, 1.0, dtype=float)

    # Ensure strictly positive.
    freqs = freqs[freqs > 0]
    if freqs.size < 2:
        raise ValueError(
            "Could not build a valid `freqs` grid (need at least 2 frequencies). "
            "Provide `freqs` explicitly."
        )
    return freqs


def _normalize_method(method: str) -> str:
    token = str(method).strip().lower()
    if token in {"morlet", "cwt_morlet", "cwt"}:
        return "morlet"
    if token in {"multitaper", "mt"}:
        return "multitaper"
    raise ValueError("`method` must be 'morlet' or 'multitaper'.")


def _run_coherency_block(
    raw: mne.io.BaseRaw,
    *,
    band_index: int,
    grid_kwargs: dict[str, Any],
    time_indices: np.ndarray,
    segment_indices: list[np.ndarray],
) -> tuple[int, np.ndarray, np.ndarray]:
    """Reduce a bounded complex-coherency block to signed PSI immediately."""
    blocks, _ = connectivity_grid(
        raw,
        **grid_kwargs,
        task_group_id=0,
        task_time_indices=time_indices,
        return_task_blocks=True,
    )
    block = blocks[0]
    cohy = np.asarray(block["data"]["cohy"])
    finite = np.all(np.isfinite(cohy), axis=-1)
    cohy = np.where(np.isfinite(cohy), cohy, 0.0)
    values = np.zeros(cohy.shape[:2], dtype=float)
    for indices in segment_indices:
        products = cohy[..., indices[:-1]].conj() * cohy[..., indices[1:]]
        values += np.imag(np.sum(products, axis=-1))
    values[~finite] = np.nan
    return band_index, np.asarray(block["time_indices"], dtype=int), values.T


def grid(
    raw: mne.io.BaseRaw,
    *,
    bands: BandSpec,
    method: str = "morlet",
    time_resolution_s: float,
    pairs: Sequence[Tuple[str, str]] | None = None,
    groups: Dict[str, Sequence[str]] | None = None,
    ordered_pairs: bool = True,
    hop_s: float | None = 0.025,
    decim: int | None = None,
    target_n_times: int | None = None,
    picks: list[str] | None = None,
    freqs: np.ndarray | None = None,
    min_cycles: float | None = 1.0,
    max_cycles: float | None = None,
    mt_time_bandwidth_product: float = 4.0,
    mt_min_cycles: float = 3.0,
    mt_max_cycles: float | None = None,
    n_jobs: int = 1,
    outer_n_jobs: int | None = None,
    mask_annotations: bool = False,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Compute PSI using the same local spectral estimates as Coherence.

    ``freqs`` supplies the computation grid for both methods; omission uses a
    1-Hz grid covering the bands. Each original band's retained frequencies use
    one shared central averaging interval selected by the connectivity backend.
    Cycle limits still apply independently to each frequency's kernel.

    ``hop_s`` or ``decim`` selects output centers, without downsampling input.
    ``mask_annotations`` excludes BAD/EDGE support across the complete estimator
    interval, including channel-specific annotations. Incomplete input windows
    remain NaN regardless of this setting. ``outer_n_jobs`` controls parallel
    blocks; when omitted, ``n_jobs`` is used. Blocks keep inner parallelism at 1.
    """
    method_use = _normalize_method(method)
    mode_use = "cwt_morlet" if method_use == "morlet" else "multitaper"
    resolved_outer_n_jobs = int(n_jobs) if outer_n_jobs is None else int(outer_n_jobs)
    band_names, band_segments, seg_edges, seg_to_band, union_edges = _normalize_bands(
        bands
    )
    sfreq = float(raw.info["sfreq"])
    if float(np.max(seg_edges[:, 1])) > sfreq / 2.0:
        raise ValueError("PSI band upper edges exceed Nyquist (sfreq / 2).")
    frequencies = np.asarray(
        _default_freqs(seg_edges) if freqs is None else freqs, dtype=float
    )
    if (
        frequencies.ndim != 1
        or frequencies.size < 2
        or np.any(~np.isfinite(frequencies))
        or np.any(frequencies <= 0)
        or np.any(frequencies > sfreq / 2.0)
        or np.any(np.diff(frequencies) <= 0)
    ):
        raise ValueError(
            "PSI `freqs` must contain at least two finite, strictly increasing "
            "positive frequencies not exceeding Nyquist."
        )
    n_internal_frequencies, n_adjacent_pairs = _segment_frequency_support(
        seg_edges, frequencies
    )
    _validate_band_frequency_support(
        method=method_use,
        band_names=band_names,
        seg_edges=seg_edges,
        seg_to_band=seg_to_band,
        n_internal_frequencies=n_internal_frequencies,
        n_adjacent_pairs=n_adjacent_pairs,
    )
    frequency_support = _frequency_support_by_band(
        band_names=band_names,
        seg_edges=seg_edges,
        seg_to_band=seg_to_band,
        n_internal_frequencies=n_internal_frequencies,
        n_adjacent_pairs=n_adjacent_pairs,
    )
    partial_message = _partial_frequency_support_message(
        band_names=band_names,
        seg_edges=seg_edges,
        seg_to_band=seg_to_band,
        n_internal_frequencies=n_internal_frequencies,
        n_adjacent_pairs=n_adjacent_pairs,
    )
    decim_eff, hop_s_eff = compute_decimation(sfreq, hop_s=hop_s, decim=decim)
    times = decimated_times_from_raw(
        raw, decim=decim_eff, target_n_times=target_n_times
    )
    n_columns_total = int(np.sum(np.isfinite(times)))
    band_params: dict[str, dict[str, Any]] = {}
    band_frequencies: dict[str, list[float]] = {}
    support_radii: list[float] = []
    valid_counts: list[int] = []
    skipped_counts: list[int] = []
    incomplete_counts: list[int] = []
    tasks: list[dict[str, Any]] = []

    for band_index, band_name in enumerate(band_names):
        used_edges = seg_edges[(seg_to_band == band_index) & (n_adjacent_pairs > 0)]
        retained = np.any(
            (frequencies[None, :] > used_edges[:, 0, None])
            & (frequencies[None, :] < used_edges[:, 1, None]),
            axis=0,
        )
        band_freqs = frequencies[retained]
        indices_by_segment = [
            np.flatnonzero((band_freqs > low) & (band_freqs < high))
            for low, high in used_edges
        ]
        grid_kwargs = dict(
            freqs=band_freqs,
            method="cohy",
            spectral_mode=mode_use,
            time_resolution_s=float(time_resolution_s),
            hop_s=hop_s,
            decim=decim_eff,
            target_n_times=target_n_times,
            pairs=pairs,
            groups=groups,
            ordered_pairs=bool(ordered_pairs),
            picks=picks,
            min_cycles=min_cycles,
            max_cycles=max_cycles,
            mt_time_bandwidth_product=float(mt_time_bandwidth_product),
            mt_min_cycles=float(mt_min_cycles),
            mt_max_cycles=mt_max_cycles,
            shared_window=True,
            outer_n_jobs=1,
        )
        description, cohy_metadata = connectivity_grid(
            raw, **grid_kwargs, return_task_description=True
        )
        pair_names = description["pair_names"]
        radius = float(
            np.max(
                connectivity_consumed_support_radii_seconds(
                    cohy_metadata, sfreq=sfreq, n_freqs=band_freqs.size
                )
            )
        )
        valid_indices = np.asarray(description["groups"][0]["valid_time_indices"])
        incomplete_counts.append(n_columns_total - int(valid_indices.size))
        skip_mask = (
            build_annotation_skip_time_mask(
                raw, times_s=times, radius_s=radius, output_channels=pair_names
            )
            if mask_annotations
            else np.zeros(times.size, dtype=bool)
        )
        skipped_counts.append(int(np.sum(np.isfinite(times) & skip_mask)))
        valid_indices = valid_indices[~skip_mask[valid_indices]]
        valid_counts.append(int(valid_indices.size))
        support_radii.append(radius)
        band_params[band_name] = cohy_metadata["params"]
        band_frequencies[band_name] = band_freqs.tolist()
        # Reuse indexed connectivity blocks so a long record never materializes
        # all overlapping raw windows or a complete complex-coherency tensor.
        runs = np.split(valid_indices, np.flatnonzero(np.diff(valid_indices) != 1) + 1)
        for run in runs:
            for start in range(0, run.size, 64):
                tasks.append(
                    dict(
                        band_index=band_index,
                        grid_kwargs=grid_kwargs,
                        time_indices=run[start : start + 64],
                        segment_indices=indices_by_segment,
                    )
                )

    out = np.full((1, len(pair_names), len(band_names), times.size), np.nan)
    # Raw cleanup can unlink shared memmaps before later tasks deserialize them.
    results = Parallel(
        n_jobs=resolved_outer_n_jobs, return_as="generator", max_nbytes=None
    )(delayed(_run_coherency_block)(raw, **task) for task in tasks)
    for band_index, time_indices, values in results:
        out[0, :, band_index, :][:, time_indices] = values

    metadata: Dict[str, Any] = dict(
        axes=dict(
            epoch=np.arange(1, dtype=int),
            channel=list(pair_names),
            freq=list(band_names),
            time=times,
            shape=out.shape,
        ),
        params=dict(
            bands_segments_hz={
                name: [[float(a), float(b)] for a, b in segments]
                for name, segments in zip(band_names, band_segments)
            },
            bands_union_hz={
                name: union_edges[index].tolist()
                for index, name in enumerate(band_names)
            },
            band_names=list(band_names),
            segments_flat_hz=seg_edges.tolist(),
            segments_to_band=seg_to_band.tolist(),
            partial_frequency_support=partial_message is not None,
            frequency_support_by_band=frequency_support,
            frequencies_by_band=band_frequencies,
            coherency_params_by_band=band_params,
            coherency_estimation=PSI_COHERENCY_ESTIMATION,
            band_support_radii_s=support_radii,
            time_axis_mode="sliding_window",
            time_resolution_s=float(time_resolution_s),
            hop_s=hop_s,
            decim=decim_eff,
            hop_s_eff=hop_s_eff,
            target_n_times=target_n_times,
            picks=picks,
            ordered_pairs=bool(ordered_pairs),
            pairs=list(pair_names),
            method=method_use,
            spectral_mode=mode_use,
            mt_time_bandwidth_product=(
                float(mt_time_bandwidth_product) if method_use == "multitaper" else None
            ),
            mt_min_cycles=float(mt_min_cycles) if method_use == "multitaper" else None,
            mt_max_cycles=mt_max_cycles if method_use == "multitaper" else None,
            min_cycles=min_cycles if method_use == "morlet" else None,
            max_cycles=max_cycles if method_use == "morlet" else None,
            freqs=frequencies,
            annotation_skip_enabled=bool(mask_annotations),
            n_columns_total=n_columns_total,
            n_columns_skipped_masked=max(skipped_counts),
            n_columns_skipped_masked_by_band=skipped_counts,
            n_columns_dropped_incomplete_window=max(incomplete_counts),
            n_columns_dropped_incomplete_window_by_band=incomplete_counts,
            n_valid_windows_by_band=valid_counts,
            n_jobs=int(n_jobs),
            outer_n_jobs=resolved_outer_n_jobs,
            sfreq_used_hz=sfreq,
        ),
    )
    if mask_annotations:
        out, metadata = apply_dynamic_edge_mask_strict(
            raw=raw,
            tensor=out,
            metadata=metadata,
            metric_label="PSI",
            freqs_lookup=band_names,
            radii_s=support_radii,
            warn_fully_masked=False,
        )
    unusable = int(np.sum(~np.any(np.isfinite(out), axis=(0, 1, 3))))
    if unusable:
        warnings.warn(
            f"PSI contains {unusable} bands with no usable output centers; "
            "they remain NaN.",
            UserWarning,
            stacklevel=2,
        )
    if partial_message is not None:
        warnings.warn(partial_message, UserWarning, stacklevel=2)
    return out, metadata
