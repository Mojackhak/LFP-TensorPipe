"""parallel.py

Small wrappers around joblib Parallel + tqdm progress display.

This module provides:
- `parallel_process_files`: basic joblib parallel map
- `parallel_process_files_tqdm`: parallel map with a progress bar

Implementation note
-------------------
Progress integration depends on `tqdm_joblib`, which is a required
project dependency. Keep the bridge explicit instead of patching
joblib internals at runtime.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import os
from typing import Any, Callable, Sequence, TypeVar

from joblib import Parallel, delayed
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib

T = TypeVar("T")


def parallel_process_files(
    files: Sequence[object],
    func: Callable[..., T],
    n_jobs: int = -1,
    verbose: bool = False,
    **kwargs,
) -> list[T]:
    """Run a function over files in parallel (joblib).

    Parameters
    ----------
    files:
        Iterable of file-like objects (usually str/Path).
    func:
        Callable to execute. It will be called as `func(file, **kwargs)`.
    n_jobs:
        joblib parallelism. -1 uses all cores.
    verbose:
        If True, print a short summary.
    **kwargs:
        Forwarded to `func`.

    Returns
    -------
    list
        Results collected from all jobs, in the same order as `files`.
    """
    if verbose:
        print(f"Processing {len(files)} files with {n_jobs} parallel jobs")

    results: list[T] = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(func)(file, **kwargs) for file in files
    )
    return results


def _call_silently(func: Callable[..., T], file: object, kwargs: dict) -> T:
    """Call `func(file, **kwargs)` while silencing stdout/stderr."""
    with (
        open(os.devnull, "w") as devnull,
        redirect_stdout(devnull),
        redirect_stderr(devnull),
    ):
        return func(file, **kwargs)


def parallel_process_files_tqdm(
    files: Sequence[object],
    func: Callable[..., T],
    n_jobs: int = -1,
    verbose: bool = False,
    prefer: str = "processes",
    silence_workers: bool = True,
    **kwargs: Any,
) -> list[T]:
    """Run a function over files in parallel with a tqdm progress bar.

    Parameters
    ----------
    files:
        Iterable of file-like objects (usually str/Path).
    func:
        Callable to execute. It will be called as `func(file, **kwargs)`.
    n_jobs:
        joblib parallelism. -1 uses all cores.
    verbose:
        If True, show progress bar and a short summary.
    prefer:
        joblib backend preference. Common values: "processes" or "threads".
    silence_workers:
        If True (default), suppress worker stdout/stderr to keep console clean.
        If False, allow worker prints to go through (may interleave across jobs).
    **kwargs:
        Forwarded to `func`.

    Returns
    -------
    list
        Results collected from all jobs, in the same order as `files`.
    """
    if verbose:
        print(f"Processing {len(files)} files with {n_jobs} parallel jobs")

    jl_verbose = 10 if verbose else 0
    bar = tqdm(total=len(files), disable=not verbose)

    with tqdm_joblib(bar):
        if silence_workers:
            results: list[T] = Parallel(
                n_jobs=n_jobs, verbose=jl_verbose, prefer=prefer
            )(delayed(_call_silently)(func, file, kwargs) for file in files)
        else:
            # Direct call: allows worker prints, but output may interleave.
            results = Parallel(n_jobs=n_jobs, verbose=jl_verbose, prefer=prefer)(
                delayed(func)(file, **kwargs) for file in files
            )

    return results
