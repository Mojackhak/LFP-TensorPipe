"""Decompose a time-frequency power tensor into aperiodic and periodic components.

This file contains SpecParam-based utilities to parameterize a spectrogram/TFR
and to reconstruct (in the linear power domain):

  - the aperiodic 1/f component
  - the periodic (oscillatory peak) component
  - the full modeled spectrum

The public entry point is :func:`decompose`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def _require_specparam() -> tuple[Any, Any, Any]:
    """Import specparam lazily to keep core TFR code usable without it."""
    try:
        import specparam  # type: ignore
        from specparam import SpectralTimeModel  # type: ignore
        from specparam.sim import sim_spectrogram  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "SpecParam is required for TFR decomposition. Install `specparam` to use this module."
        ) from exc
    return specparam, SpectralTimeModel, sim_spectrogram


def _fit_one_epoch_channel(
    e_idx: int,
    ch_idx: int,
    psd_ft: np.ndarray,  # (n_freqs, n_times)
    freqs: np.ndarray,
    settings: Dict[str, Any],
    *,
    report_dir: Optional[str | Path] = None,
    save_prefix: Optional[str] = None,
    original_time_indices: np.ndarray | None = None,
    original_times_s: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Fit SpecParam SpectralTimeModel for one (epoch, channel) and simulate components."""
    specparam, SpectralTimeModel, sim_spectrogram = _require_specparam()

    model = SpectralTimeModel(
        peak_width_limits=settings.get("peak_width_limits", [2, 12]),
        max_n_peaks=settings.get("max_n_peaks", np.inf),
        min_peak_height=settings.get("min_peak_height", 0),
        peak_threshold=settings.get("peak_threshold", 2.0),
        aperiodic_mode=settings.get("aperiodic_mode", "fixed"),
        verbose=settings.get("verbose", False),
    )

    n_freqs, n_times = psd_ft.shape
    if freqs.shape[0] != n_freqs:
        raise ValueError("`freqs` length must match n_freqs of psd_ft.")

    freq_res = float(np.mean(np.diff(freqs))) if n_freqs > 1 else 1.0
    freq_fit_range = [float(freqs[0]), float(freqs[-1])]

    freq_range = settings.get("freq_range", None)
    model.fit(freqs, psd_ft, freq_range=freq_range)

    if report_dir is not None:
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(
            report_dir, f"{save_prefix or 'fit'}_e{e_idx:03d}_ch{ch_idx:03d}.pdf"
        )
        try:
            model.save_report(report_path)
        except Exception as exc:
            raise RuntimeError(
                f"SpecParam report export failed for epoch {e_idx}, channel {ch_idx}: {report_path}"
            ) from exc
        if original_time_indices is not None:
            original_idx = np.asarray(original_time_indices, dtype=int).ravel()
            if original_idx.size != n_times:
                raise ValueError(
                    "SpecParam report time-index mapping must match fitted columns."
                )
            if original_times_s is None:
                original_seconds = original_idx.astype(float)
            else:
                all_times = np.asarray(original_times_s, dtype=float).ravel()
                original_seconds = all_times[original_idx]
            sidecar_path = Path(report_path).with_suffix(".time-index.json")
            with sidecar_path.open("w", encoding="utf-8") as stream:
                json.dump(
                    [
                        {
                            "compressed_time_index": int(compressed_idx),
                            "original_time_index": int(source_idx),
                            "original_time_seconds": float(original_seconds[compressed_idx]),
                        }
                        for compressed_idx, source_idx in enumerate(original_idx)
                    ],
                    stream,
                    indent=2,
                )
                stream.write("\n")

    # The specparam distribution uses the 2.x results API: SpectralTimeModel
    # exposes one FitResults object per window through results.group_results.
    fit_results = list(model.results.group_results)
    if len(fit_results) != n_times:
        raise RuntimeError(
            "SpecParam fit-result count does not match the spectrogram time axis."
        )

    try:
        major = int(str(getattr(specparam, "__version__", "0")).split(".")[0])
    except Exception:
        major = 0
    use_new_sim_api = major >= 2

    def _simulate_component(ap_vec: np.ndarray, peak_list: list[list[float]]):
        ap_vec_list = np.asarray(ap_vec, dtype=float).tolist()
        peak_list_list = (
            np.asarray(peak_list, dtype=float).tolist() if len(peak_list) > 0 else []
        )

        ap_dict = (
            {"knee": ap_vec_list} if len(ap_vec_list) == 3 else {"fixed": ap_vec_list}
        )
        peak_dict = {"gaussian": peak_list_list}

        attempts = []
        if use_new_sim_api:
            attempts.append((ap_dict, peak_dict))
        attempts.append((ap_vec_list, peak_list_list))
        if not use_new_sim_api:
            attempts.append((ap_dict, peak_dict))

        first_exc: Exception | None = None
        for ap_arg, peak_arg in attempts:
            try:
                _, spec = sim_spectrogram(
                    n_windows=1,
                    freq_range=freq_fit_range,
                    aperiodic_params=ap_arg,
                    periodic_params=peak_arg,
                    nlvs=0,
                    freq_res=freq_res,
                )
                return spec
            except Exception as exc:  # pragma: no cover
                if first_exc is None:
                    first_exc = exc

        raise first_exc or RuntimeError(
            "sim_spectrogram failed for both new and old API."
        )

    ap_linear = np.empty((n_freqs, n_times), dtype=float)
    per_linear = np.zeros((n_freqs, n_times), dtype=float)
    full_linear = np.empty((n_freqs, n_times), dtype=float)

    for t, fit_result in enumerate(fit_results):
        ap_t = np.asarray(fit_result.aperiodic_fit, dtype=float)
        ap_linear[:, t] = _simulate_component(ap_t, []).flatten()

        # SpecParam ``peak_fit`` contains Gaussian center, height, and sigma;
        # ``peak_converted`` contains converted peak bandwidth instead.
        peak_fit = np.asarray(fit_result.peak_fit, dtype=float)
        peaks_t = (
            np.atleast_2d(peak_fit).reshape(-1, 3).tolist() if peak_fit.size > 0 else []
        )

        per_linear[:, t] = _simulate_component(
            np.array([0, 0, 0], dtype=float), peaks_t
        ).flatten()
        full_linear[:, t] = _simulate_component(ap_t, peaks_t).flatten()

    params = model.to_df().reset_index().rename(columns={"index": "time_idx"})
    params["epoch"] = int(e_idx)
    params["channel"] = int(ch_idx)

    return ap_linear, per_linear, full_linear, params


def decompose(
    tfr: np.ndarray,  # (n_epochs, n_channels, n_freqs, n_times)
    freqs: np.ndarray,
    *,
    times: np.ndarray | None = None,
    ch_names: list[str] | None = None,
    freq_range: Optional[Tuple[float, float]] = None,
    aperiodic_mode: str = "fixed",
    peak_width_limits: Tuple[float, float] = (2.0, 12.0),
    max_n_peaks: int | float = np.inf,
    min_peak_height: float = 0.0,
    peak_threshold: float = 2.0,
    n_jobs: int = 1,
    report_dir: Optional[str | Path] = None,
    save_prefix: str = "specparam",
    verbose: bool = False,
    valid_time_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Decompose a TFR tensor using SpecParam and return components + aperiodic params tensor.

    Parameters
    ----------
    tfr
        Power tensor with shape (n_epochs, n_channels, n_freqs, n_times).
        Values should be non-negative power (linear domain is fine; SpecParam works in log10 internally).
    freqs
        Frequency axis in Hz, shape (n_freqs,).
    times
        Optional time axis in seconds for metadata. If provided, must have length n_times.
    ch_names
        Optional channel-name list for metadata. If provided, must have length n_channels.
    freq_range
        Optional (fmin, fmax) used to restrict SpecParam fitting.
    aperiodic_mode
        "fixed" -> [offset, exponent]; "knee" -> [offset, knee, exponent].
    peak_width_limits, max_n_peaks, min_peak_height, peak_threshold
        SpecParam peak detection settings.
    n_jobs
        Parallel jobs across (epoch, channel).
    report_dir
        Optional directory for SpecParam PDF reports (one per epoch×channel).
    save_prefix
        Prefix for saved reports.
    verbose
        Verbosity passed to SpecParam.
    valid_time_mask
        Optional boolean mask selecting time columns to fit. Outputs retain the
        original time axis and skipped columns remain NaN.

    Returns
    -------
    tfr_aperiodic
        Linear-domain aperiodic component, shape (n_epochs, n_channels, n_freqs, n_times).
    tfr_periodic
        Linear-domain periodic (peaks) component, shape (n_epochs, n_channels, n_freqs, n_times).
    tfr_full
        Linear-domain full modeled spectrum, shape (n_epochs, n_channels, n_freqs, n_times).
    params_tensor
        Aperiodic + fit statistics tensor, shape (n_epochs, n_channels, n_params, n_times).
        The parameter names are stored in ``params_meta['axes']['freq']`` (list[str]).
    params_meta
        Metadata dict with axes + parameters used for the decomposition.

    Notes
    -----
    - The params tensor is designed to be time-aligned with the input TFR, so it can be
      masked/warped with the same utilities as other tensors.
    - ``params_meta['axes']['freq']`` intentionally stores **strings** (parameter names).
      Downstream code must not assume it is numeric.
    """
    if tfr.ndim != 4:
        raise ValueError("tfr must be 4D: (n_epochs, n_channels, n_freqs, n_times).")

    n_epochs, n_channels, n_freqs, n_times = tfr.shape
    freqs = np.asarray(freqs, dtype=float)
    if freqs.ndim != 1 or freqs.shape[0] != n_freqs:
        raise ValueError("`freqs` must be 1D and match the TFR frequency axis length.")

    if times is not None:
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or times.shape[0] != n_times:
            raise ValueError("`times` must be 1D with length equal to tfr.shape[-1].")

    if ch_names is not None:
        if len(ch_names) != n_channels:
            raise ValueError(
                "`ch_names` length must match the TFR channel axis length."
            )

    if valid_time_mask is None:
        valid_mask = np.ones(n_times, dtype=bool)
    else:
        valid_mask = np.asarray(valid_time_mask, dtype=bool)
        if valid_mask.ndim != 1 or valid_mask.size != n_times:
            raise ValueError(
                "`valid_time_mask` must be 1D with length equal to tfr.shape[-1]."
            )
    fitted_time_indices = np.flatnonzero(valid_mask)
    time_axis = (
        np.arange(n_times, dtype=float)
        if times is None
        else np.asarray(times, dtype=float)
    )

    # Skipped columns remain NaN by construction.
    tfr_aperiodic = np.full_like(tfr, np.nan, dtype=float)
    tfr_periodic = np.full_like(tfr, np.nan, dtype=float)
    tfr_full = np.full_like(tfr, np.nan, dtype=float)

    settings: Dict[str, Any] = dict(
        peak_width_limits=peak_width_limits,
        max_n_peaks=max_n_peaks,
        min_peak_height=min_peak_height,
        peak_threshold=peak_threshold,
        aperiodic_mode=aperiodic_mode,
        verbose=verbose,
        freq_range=freq_range,
    )

    # Prepare tasks across (epoch, channel). Column compression occurs in _job.
    tasks: list[tuple[int, int, np.ndarray]] = []
    for e in range(n_epochs):
        for c in range(n_channels):
            psd_ft = np.ascontiguousarray(
                tfr[e, c, :, :], dtype=float
            )  # (n_freqs, n_times)
            tasks.append((e, c, psd_ft))

    def _job(e: int, c: int, psd_ft: np.ndarray):
        psd_fit = np.ascontiguousarray(psd_ft[:, fitted_time_indices], dtype=float)
        ap, per, full, df = _fit_one_epoch_channel(
            e,
            c,
            psd_fit,
            freqs,
            settings,
            report_dir=report_dir,
            save_prefix=save_prefix,
            original_time_indices=fitted_time_indices,
            original_times_s=time_axis,
        )
        return e, c, ap, per, full, df

    # Infer parameter columns from the first result.
    def _infer_param_names(df: pd.DataFrame) -> list[str]:
        exclude = {"time_idx", "epoch", "channel"}
        names: list[str] = []
        for col in df.columns:
            if col in exclude:
                continue
            # We only keep numeric columns; non-numeric entries are ignored.
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    names.append(str(col))
            except Exception:
                continue
        if not names:
            raise RuntimeError(
                "SpecParam returned no numeric parameter columns (unexpected). "
                "Check SpecParam version and model.to_df() output."
            )
        return names

    schema_reference_fit = False
    schema_reference_time_index: int | None = None
    if fitted_time_indices.size > 0:
        results = Parallel(n_jobs=int(n_jobs), backend="loky")(
            delayed(_job)(e, c, psd) for e, c, psd in tasks
        )
        if len(results) == 0:
            raise RuntimeError(
                "No results returned from SpecParam decomposition (unexpected)."
            )
        param_names = _infer_param_names(results[0][5])
    else:
        if n_epochs < 1 or n_channels < 1 or n_times < 1:
            raise RuntimeError(
                "SpecParam schema-reference fit requires a non-empty input tensor."
            )
        _, _, _, reference_df = _fit_one_epoch_channel(
            0,
            0,
            np.ascontiguousarray(tfr[0, 0, :, 0:1], dtype=float),
            freqs,
            settings,
            report_dir=None,
            save_prefix=save_prefix,
        )
        param_names = _infer_param_names(reference_df)
        results = []
        schema_reference_fit = True
        schema_reference_time_index = 0

    n_params = len(param_names)
    params_tensor = np.full(
        (n_epochs, n_channels, n_params, n_times), np.nan, dtype=np.float64
    )

    # Collect outputs.
    for e, c, ap, per, full, df in results:
        tfr_aperiodic[e, c][:, fitted_time_indices] = ap
        tfr_periodic[e, c][:, fitted_time_indices] = per
        tfr_full[e, c][:, fitted_time_indices] = full

        if not isinstance(df, pd.DataFrame) or "time_idx" not in df.columns:
            raise RuntimeError(
                "SpecParam output DataFrame is missing required 'time_idx' column."
            )

        # Validate param columns exist.
        missing = [p for p in param_names if p not in df.columns]
        if missing:
            raise RuntimeError(
                "SpecParam parameter columns mismatch across channels/epochs. "
                f"Missing columns: {missing}"
            )

        compressed_time_idx = df["time_idx"].astype(int).to_numpy()
        if compressed_time_idx.size == 0:
            continue
        if np.any(compressed_time_idx < 0) or np.any(
            compressed_time_idx >= fitted_time_indices.size
        ):
            raise RuntimeError(
                "SpecParam returned time_idx outside the compressed time axis."
            )
        original_time_idx = fitted_time_indices[compressed_time_idx]
        df = df.copy()
        df["time_idx"] = original_time_idx

        vals = df[param_names].to_numpy(dtype=float)  # (n_rows, n_params)
        if vals.shape[0] != original_time_idx.shape[0]:
            raise RuntimeError("SpecParam params rows do not match time_idx length.")

        # NOTE:
        # NumPy's advanced indexing reorders axes when integer indices appear before
        # an index array (time_idx). Assign via a 2D view so the intended layout is
        # explicit: (param, time).
        params_tensor[e, c][:, original_time_idx] = vals.T.astype(
            np.float64, copy=False
        )

    # --- metadata ---
    if ch_names is None:
        ch_axis = [str(i) for i in range(n_channels)]
    else:
        ch_axis = [str(c) for c in ch_names]

    params_meta: Dict[str, Any] = dict(
        axes=dict(
            epoch=np.arange(n_epochs, dtype=int),
            channel=np.array(ch_axis, dtype=object),
            # IMPORTANT: param names are strings (not numeric frequencies).
            freq=list(param_names),
            time=np.asarray(time_axis, dtype=float),
            shape=tuple(params_tensor.shape),
        ),
        params=dict(
            source="specparam.SpectralTimeModel",
            freq_range=(
                tuple(map(float, freq_range)) if freq_range is not None else None
            ),
            aperiodic_mode=str(aperiodic_mode),
            peak_width_limits=tuple(map(float, peak_width_limits)),
            max_n_peaks=float(max_n_peaks) if max_n_peaks is not None else None,
            min_peak_height=float(min_peak_height),
            peak_threshold=float(peak_threshold),
            n_jobs=int(n_jobs),
            fitted_time_indices=[int(item) for item in fitted_time_indices.tolist()],
            schema_reference_fit=bool(schema_reference_fit),
            schema_reference_time_index=schema_reference_time_index,
        ),
    )

    return tfr_aperiodic, tfr_periodic, tfr_full, params_tensor, params_meta


def make_gof_rsquared_masker(params_tensor_4d, params_meta, threshold):

    params = np.asarray(params_tensor_4d)
    if params.ndim != 4:
        raise ValueError(
            "params_tensor_4d must be 4D: (n_epochs, n_channels, n_params, n_times). "
            f"Got shape={params.shape}."
        )

    param_names = list(
        params_meta["axes"]["freq"]
    )  # KeyError if missing (as requested)
    qc_idx = param_names.index("gof_rsquared")  # ValueError if missing (as requested)

    qc = params[:, :, qc_idx, :]  # (n_epochs, n_channels, n_times)
    mask_bad = (qc < threshold) | (~np.isfinite(qc))  # NaN/inf are bad

    n_epochs, n_channels, _, n_times = params.shape

    def masker(tensor_4d):
        x = np.asarray(tensor_4d)
        if x.ndim != 4:
            raise ValueError(f"Input tensor must be 4D. Got shape={x.shape}.")
        if (
            (x.shape[0] != n_epochs)
            or (x.shape[1] != n_channels)
            or (x.shape[3] != n_times)
        ):
            raise ValueError(
                "Input tensor must match (n_epochs, n_channels, n_times) of params_tensor_4d. "
                f"Expected first dims ({n_epochs}, {n_channels}) and last dim {n_times}, got {x.shape}."
            )

        out = (
            x.astype(np.float64, copy=True)
            if not np.issubdtype(x.dtype, np.floating)
            else x.copy()
        )
        out[np.broadcast_to(mask_bad[:, :, None, :], out.shape)] = np.nan
        return out

    return masker
