"""Autosave and MNE-browser backend helpers for preprocess plotting."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import logging
import os

import numpy as np

from lfptensorpipe.app.path_resolver import PathResolver
from lfptensorpipe.app.preproc.indicator import preproc_filter_review_required
from lfptensorpipe.app.preproc.lineage import (
    capture_preproc_input_generation,
    filter_preview_lineage_is_current,
    preproc_input_generation_matches,
    preproc_step_lineage_is_current,
)
from lfptensorpipe.app.preproc.steps.filter import (
    filter_log_has_current_bad_channel_detection_semantics,
)
from lfptensorpipe.app.runlog_store import (
    RunLogRecord,
    append_run_log_event,
    read_run_log,
)
from lfptensorpipe.app.shared.atomic_outputs import AtomicOutputSet
from lfptensorpipe.app.shared.generation_lineage import (
    accepted_result_generation_id,
    new_result_generation_id,
    params_with_generation_lineage,
)
from lfptensorpipe.gui.shell.common import (
    Any,
    QApplication,
    Path,
    QObject,
    QTimer,
    QWidget,
    invalidate_downstream_preproc_steps,
)

logger = logging.getLogger(__name__)

PREPROC_PLOT_WINDOW_SIZE = (1200, 800)
_PREPROC_PLOT_DPI_FALLBACK = 96.0
_PLOT_CHANGE_TRACKED_STEPS = frozenset(
    ("filter", "annotations", "bad_segment_removal", "ecg_artifact_removal")
)
_PLOT_ATOMIC_EDIT_STEPS = _PLOT_CHANGE_TRACKED_STEPS.difference({"filter"})


def _normalize_preproc_plot_orig_time(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    if isinstance(value, np.datetime64):
        return str(value.astype("datetime64[ns]"))
    if isinstance(value, (tuple, list)):
        normalized_items: list[Any] = []
        for item in value:
            if isinstance(item, (np.integer, int)):
                normalized_items.append(int(item))
                continue
            normalized_items.append(item)
        return tuple(normalized_items)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return repr(value)


def _normalize_preproc_plot_annotations(
    raw: Any,
) -> tuple[tuple[str, float, float, tuple[str, ...]], ...]:
    annotations = getattr(raw, "annotations", None)
    if annotations is None:
        return ()
    items: list[tuple[str, float, float, tuple[str, ...]]] = []
    for onset, duration, description, ch_names in zip(
        annotations.onset,
        annotations.duration,
        annotations.description,
        annotations.ch_names,
    ):
        scope = tuple(sorted(str(name) for name in ch_names))
        items.append((str(description).strip(), float(onset), float(duration), scope))
    items.sort(key=lambda item: (item[1], item[2], item[0], item[3]))
    return tuple(items)


def _normalize_preproc_plot_bads(raw: Any) -> tuple[str, ...]:
    info = getattr(raw, "info", None)
    if not hasattr(info, "get"):
        return ()
    bads = info.get("bads", [])
    if not isinstance(bads, (list, tuple)):
        return ()
    normalized = {str(item).strip() for item in bads if str(item).strip()}
    return tuple(sorted(normalized))


def _preproc_plot_raw_signature(raw: Any) -> dict[str, Any]:
    data_getter = getattr(raw, "get_data", None)
    data_obj = (
        data_getter() if callable(data_getter) else np.empty((0, 0), dtype=np.float64)
    )
    data = np.ascontiguousarray(data_obj, dtype=np.float64)
    digest = hashlib.sha256(data.tobytes()).hexdigest()
    return {
        "ch_names": tuple(str(name) for name in getattr(raw, "ch_names", ())),
        "sfreq": float(getattr(raw, "info", {}).get("sfreq", 0.0)),
        "n_times": int(getattr(raw, "n_times", data.shape[-1] if data.ndim else 0)),
        "data_digest": digest,
        "annotations": _normalize_preproc_plot_annotations(raw),
        "bads": _normalize_preproc_plot_bads(raw),
        "orig_time": _normalize_preproc_plot_orig_time(
            getattr(getattr(raw, "annotations", None), "orig_time", None)
        ),
    }


def _preproc_plot_disk_state(raw_path: Any) -> tuple[int, int] | None:
    """Return (size, mtime_ns) of the on-disk artifact, or None when unreadable."""
    if not isinstance(raw_path, Path):
        return None
    try:
        stat_result = raw_path.stat()
    except OSError:
        return None
    return (int(stat_result.st_size), int(stat_result.st_mtime_ns))


def _preproc_plot_generation_snapshot(
    context: Any,
    step: str,
) -> dict[str, Any] | None:
    """Capture one editable step's accepted lineage without creating paths."""
    try:
        resolver = PathResolver(context)
    except Exception:
        return None
    log_path = resolver.preproc_step_dir(step, create=False) / "lfptensorpipe_log.json"
    try:
        payload = read_run_log(log_path)
    except Exception:
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("completed") is not True
        or not preproc_step_lineage_is_current(resolver, step)
    ):
        return None
    captured = capture_preproc_input_generation(resolver, step)
    if captured is None:
        return None
    source_step, input_generations = captured
    return {
        "log_path": log_path,
        "log_disk_state": _preproc_plot_disk_state(log_path),
        "result_generation_id": accepted_result_generation_id(payload),
        "source_step": source_step,
        "input_generations": dict(input_generations),
    }


def _preproc_plot_generation_stale_reason(
    context: Any,
    step: str,
    snapshot: Any,
) -> str | None:
    if not isinstance(snapshot, dict):
        return "accepted generation could not be captured when the plot opened"
    log_path = snapshot.get("log_path")
    if not isinstance(log_path, Path):
        return "accepted log could not be resolved"
    if _preproc_plot_disk_state(log_path) != snapshot.get("log_disk_state"):
        return "accepted log was regenerated on disk after this plot was opened"
    try:
        resolver = PathResolver(context)
        payload = read_run_log(log_path)
    except Exception:
        return "accepted log is no longer readable"
    if (
        not isinstance(payload, dict)
        or payload.get("completed") is not True
        or not preproc_step_lineage_is_current(resolver, step)
    ):
        return "accepted generation is no longer current"
    if accepted_result_generation_id(payload) != snapshot.get("result_generation_id"):
        return "accepted generation changed after this plot was opened"
    captured = capture_preproc_input_generation(resolver, step)
    expected_capture = (
        snapshot.get("source_step"),
        snapshot.get("input_generations"),
    )
    if captured != expected_capture:
        return "input generation changed after this plot was opened"
    return None


def _capture_tracked_plot_open_state(
    owner: Any,
    *,
    raw_path: Path,
    step: str,
) -> dict[str, Any]:
    """Capture one tracked result or Preview before its Raw artifact is read."""
    context = owner._record_context()
    resolver = PathResolver(context)
    opened_disk_state = _preproc_plot_disk_state(raw_path)
    if opened_disk_state is None:
        raise RuntimeError(f"{raw_path.name} no longer exists or is unreadable.")

    if step == "filter" and raw_path.name == "preview_raw.fif":
        qc_dir = resolver.preproc_step_dir("filter", create=False) / "qc"
        paths = (
            qc_dir / "preview_raw.fif",
            qc_dir / "preview_config.yml",
            qc_dir / "preview_log.json",
        )
        if raw_path != paths[0]:
            raise RuntimeError("Filter Preview path does not match this record.")
        payload = read_run_log(paths[2])
        if (
            not isinstance(payload, dict)
            or payload.get("completed") is not False
            or not filter_log_has_current_bad_channel_detection_semantics(payload)
            or not filter_preview_lineage_is_current(resolver, payload)
        ):
            raise RuntimeError("Filter Preview is no longer current.")
        file_states = {path: _preproc_plot_disk_state(path) for path in paths}
        if any(state is None for state in file_states.values()):
            raise RuntimeError("Filter Preview artifacts are incomplete.")
        return {
            "context": context,
            "opened_disk_state": opened_disk_state,
            "preview_file_states": file_states,
            "kind": "filter_preview",
        }

    generation_snapshot = _preproc_plot_generation_snapshot(context, step)
    if generation_snapshot is None:
        raise RuntimeError(
            f"{step} does not have a current accepted generation to plot."
        )
    if step == "filter" and preproc_filter_review_required(resolver):
        raise RuntimeError(
            "A current Filter Preview must be reviewed instead of the accepted "
            "Filter result."
        )
    return {
        "context": context,
        "opened_disk_state": opened_disk_state,
        "generation_snapshot": generation_snapshot,
        "kind": "accepted",
    }


def _tracked_plot_open_stale_reason(
    *,
    raw_path: Path,
    step: str,
    open_state: dict[str, Any],
) -> str | None:
    stale_reason = _preproc_plot_stale_target_reason(
        raw_path,
        opened_disk_state=open_state.get("opened_disk_state"),
    )
    if stale_reason is not None:
        return stale_reason
    if open_state.get("kind") == "filter_preview":
        file_states = open_state.get("preview_file_states")
        if not isinstance(file_states, dict):
            return "Preview state could not be captured when the plot opened"
        if any(
            _preproc_plot_disk_state(path) != state
            for path, state in file_states.items()
        ):
            return "Preview artifacts were regenerated on disk after this plot opened"
        try:
            resolver = PathResolver(open_state.get("context"))
            payload = read_run_log(
                resolver.preproc_step_dir("filter", create=False)
                / "qc"
                / "preview_log.json"
            )
        except Exception:
            return "Preview state is no longer readable"
        if (
            not isinstance(payload, dict)
            or payload.get("completed") is not False
            or not filter_log_has_current_bad_channel_detection_semantics(payload)
            or not filter_preview_lineage_is_current(resolver, payload)
        ):
            return "Preview generation is no longer current"
        return None
    if step == "filter":
        try:
            resolver = PathResolver(open_state.get("context"))
        except Exception:
            return "Filter review source is no longer readable"
        if preproc_filter_review_required(resolver):
            return "a current Filter Preview appeared after this plot opened"
    return _preproc_plot_generation_stale_reason(
        open_state.get("context"),
        step,
        open_state.get("generation_snapshot"),
    )


def _promote_edited_preproc_plot_raw(
    *,
    context: Any,
    step: str,
    raw: Any,
    raw_path: Path,
    opened_raw_disk_state: tuple[int, int] | None,
    generation_snapshot: dict[str, Any],
) -> None:
    """Collectively promote one edited Raw and a fresh successful log event."""
    resolver = PathResolver(context)
    log_path = generation_snapshot["log_path"]
    payload = read_run_log(log_path)
    if not isinstance(payload, dict) or payload.get("completed") is not True:
        raise RuntimeError("Editable Preprocess result is no longer accepted.")
    prior_params = payload.get("params")
    if not isinstance(prior_params, dict):
        raise RuntimeError("Editable Preprocess result has invalid saved params.")
    source_step = str(generation_snapshot["source_step"])
    input_generations = dict(generation_snapshot["input_generations"])
    result_generation_id = new_result_generation_id()

    with AtomicOutputSet(
        [raw_path, log_path],
        cleanup_stale_residues=True,
    ) as output_set:
        raw.save(str(output_set.staged_path(raw_path)), overwrite=True)
        append_run_log_event(
            output_set.staged_path(log_path),
            RunLogRecord(
                step=step,
                completed=True,
                params=params_with_generation_lineage(
                    {
                        **prior_params,
                        "source_step": source_step,
                    },
                    result_generation_id=result_generation_id,
                    input_generations=input_generations,
                ),
                input_path=str(payload.get("input_path", "")),
                output_path=str(raw_path),
                message=f"Saved edits from the {step} MNE plot.",
            ),
            source_path=log_path,
        )
        if _preproc_plot_disk_state(raw_path) != opened_raw_disk_state:
            raise RuntimeError(
                "Editable Preprocess Raw changed while plot edits were staged."
            )
        if _preproc_plot_disk_state(log_path) != generation_snapshot.get(
            "log_disk_state"
        ):
            raise RuntimeError(
                "Editable Preprocess log changed while plot edits were staged."
            )
        if not preproc_input_generation_matches(
            resolver,
            step,
            source_step=source_step,
            input_generations=input_generations,
        ):
            raise RuntimeError(
                "Editable Preprocess input generation changed during plot close."
            )
        output_set.commit()


def _preproc_plot_figsize_env_value(owner: Any) -> str:
    dpi_x = _PREPROC_PLOT_DPI_FALLBACK
    dpi_y = _PREPROC_PLOT_DPI_FALLBACK
    screen = None
    screen_getter = getattr(owner, "screen", None)
    if callable(screen_getter):
        try:
            screen = screen_getter()
        except Exception:
            screen = None
    if screen is None:
        app = QApplication.instance()
        if app is not None:
            try:
                screen = app.primaryScreen()
            except Exception:
                screen = None
    if screen is not None:
        dpi_x_getter = getattr(screen, "logicalDotsPerInchX", None)
        dpi_y_getter = getattr(screen, "logicalDotsPerInchY", None)
        if callable(dpi_x_getter):
            try:
                dpi_x = max(float(dpi_x_getter()), 1.0)
            except Exception:
                dpi_x = _PREPROC_PLOT_DPI_FALLBACK
        if callable(dpi_y_getter):
            try:
                dpi_y = max(float(dpi_y_getter()), 1.0)
            except Exception:
                dpi_y = _PREPROC_PLOT_DPI_FALLBACK
    width_px, height_px = PREPROC_PLOT_WINDOW_SIZE
    return f"{width_px / dpi_x:.6f},{height_px / dpi_y:.6f}"


def _resize_preproc_plot_window(browser: Any) -> None:
    resize = getattr(browser, "resize", None)
    if not callable(resize):
        return
    try:
        resize(*PREPROC_PLOT_WINDOW_SIZE)
    except Exception:
        pass


def _browser_tracking_targets(
    browser: Any,
) -> tuple[Any | None, QObject | None, Any | None]:
    figure = getattr(browser, "fig", None)
    if figure is None and hasattr(browser, "canvas"):
        figure = browser

    canvas = getattr(figure, "canvas", None)
    manager = getattr(canvas, "manager", None) if canvas is not None else None
    manager_window = getattr(manager, "window", None) if manager is not None else None

    qt_object: QObject | None = browser if isinstance(browser, QObject) else None
    if qt_object is None and isinstance(figure, QObject):
        qt_object = figure
    if qt_object is None and isinstance(manager_window, QObject):
        qt_object = manager_window
    return figure, qt_object, manager_window


def _restore_quit_on_last_window_closed(window: Any) -> None:
    app = QApplication.instance()
    if app is None:
        window._mne_browser_shutdown_prev_quit_on_last_window_closed = None
        return
    previous = getattr(
        window,
        "_mne_browser_shutdown_prev_quit_on_last_window_closed",
        None,
    )
    if previous is None:
        return
    try:
        app.setQuitOnLastWindowClosed(bool(previous))
    except Exception:
        pass
    window._mne_browser_shutdown_prev_quit_on_last_window_closed = None


def _finalize_app_shutdown(self) -> None:
    if getattr(self, "_finalizing_mainwindow_close", False):
        return
    self._mne_browser_shutdown_pending = False
    self._finalizing_mainwindow_close = True

    def _close_main_window() -> None:
        try:
            self.close()
        finally:
            _restore_quit_on_last_window_closed(self)
            app = QApplication.instance()
            if app is not None:
                QTimer.singleShot(0, app.quit)

    QTimer.singleShot(0, _close_main_window)


def _preproc_plot_stale_target_reason(
    raw_path: Any,
    *,
    opened_disk_state: tuple[int, int] | None,
) -> str | None:
    """Explain why the plotted artifact must not be written back, or None if safe.

    The plot window outlives the main-window action that opened it, so the record
    can be deleted, renamed, or regenerated behind it. Writing back in those cases
    resurrects a removed record or clobbers a fresher rerun.
    """
    current_disk_state = _preproc_plot_disk_state(raw_path)
    if current_disk_state is None:
        return "no longer exists (record deleted or renamed)"
    if opened_disk_state is None:
        return "could not be verified against the version opened by this plot"
    if current_disk_state != opened_disk_state:
        return "was regenerated on disk after this plot was opened"
    return None


def _finalize_tracked_browser_close(self, token: int, event: Any | None = None) -> None:
    _ = event
    registry = getattr(self, "_active_mne_browsers", None)
    if not isinstance(registry, dict):
        return
    entry = registry.get(token)
    if entry is None or bool(entry.get("closed", False)):
        return
    entry["closed"] = True
    registry.pop(token, None)

    context = entry.get("context")
    raw = entry.get("raw")
    raw_path = entry.get("raw_path")
    step = entry.get("step")
    title_prefix = str(entry.get("title_prefix", "Plot"))

    try:
        if step == "raw":
            self.statusBar().showMessage(f"{title_prefix} plot closed.")
        elif step == "filter" and context is not None and isinstance(raw_path, Path):
            opened_signature = entry.get("opened_signature")
            closed_signature = _preproc_plot_raw_signature(raw)
            output_role = str(entry.get("filter_output_role", "scientific"))
            should_finalize = (
                output_role == "preview" or opened_signature != closed_signature
            )
            if not should_finalize:
                self.statusBar().showMessage(f"{title_prefix} plot closed.")
            else:
                tracked_open_state = entry.get("tracked_open_state")
                stale_reason = (
                    _tracked_plot_open_stale_reason(
                        raw_path=raw_path,
                        step="filter",
                        open_state=tracked_open_state,
                    )
                    if isinstance(tracked_open_state, dict)
                    else _preproc_plot_stale_target_reason(
                        raw_path,
                        opened_disk_state=entry.get("opened_disk_state"),
                    )
                )
                if stale_reason is not None:
                    message = (
                        f"{title_prefix} plot closed: review discarded because "
                        f"{raw_path.name} {stale_reason}."
                    )
                    self.statusBar().showMessage(message)
                    self._show_warning(f"{title_prefix} Plot", message)
                else:
                    reviewed_annotations = raw.annotations.copy()
                    reviewed_bads = list(raw.info.get("bads", []))
                    active_figures = getattr(self, "_active_plot_figures", None)
                    can_switch_to_busy = not registry and not active_figures
                    if can_switch_to_busy:
                        self._set_global_ui_lock("plot", False)

                    def work() -> tuple[bool, str]:
                        review_source_is_current_fn = None
                        if isinstance(tracked_open_state, dict):

                            def review_source_is_current_fn() -> bool:
                                return (
                                    _tracked_plot_open_stale_reason(
                                        raw_path=raw_path,
                                        step="filter",
                                        open_state=tracked_open_state,
                                    )
                                    is None
                                )

                        return self._finalize_filter_review_runtime(
                            context,
                            reviewed_annotations=reviewed_annotations,
                            reviewed_bads=reviewed_bads,
                            review_source_is_current_fn=(review_source_is_current_fn),
                        )

                    if can_switch_to_busy and hasattr(self, "_run_with_busy"):
                        ok, message = self._run_with_busy("Filter Finalize", work)
                    else:
                        ok, message = work()
                    self._refresh_stage_states_from_context()
                    self._refresh_preproc_controls()
                    if ok:
                        self.statusBar().showMessage(
                            f"{title_prefix} plot closed: {message}"
                        )
                    else:
                        warning = f"{title_prefix} plot close failed: {message}"
                        self.statusBar().showMessage(warning)
                        self._show_warning(f"{title_prefix} Plot", warning)
        elif (
            step in _PLOT_CHANGE_TRACKED_STEPS
            and context is not None
            and isinstance(raw_path, Path)
        ):
            opened_signature = entry.get("opened_signature")
            closed_signature = _preproc_plot_raw_signature(raw)
            if opened_signature == closed_signature:
                self.statusBar().showMessage(f"{title_prefix} plot closed.")
            else:
                stale_reason = _preproc_plot_stale_target_reason(
                    raw_path,
                    opened_disk_state=entry.get("opened_disk_state"),
                )
                if stale_reason is not None:
                    message = (
                        f"{title_prefix} plot closed: edits discarded because "
                        f"{raw_path.name} {stale_reason}."
                    )
                    self.statusBar().showMessage(message)
                    self._show_warning(f"{title_prefix} Plot", message)
                else:
                    generation_snapshot = entry.get("generation_snapshot")
                    generation_stale_reason = _preproc_plot_generation_stale_reason(
                        context,
                        step,
                        generation_snapshot,
                    )
                    if generation_stale_reason is not None:
                        message = (
                            f"{title_prefix} plot closed: edits discarded because "
                            f"{raw_path.name} {generation_stale_reason}."
                        )
                        self.statusBar().showMessage(message)
                        self._show_warning(f"{title_prefix} Plot", message)
                    else:
                        assert isinstance(generation_snapshot, dict)
                        _promote_edited_preproc_plot_raw(
                            context=context,
                            step=step,
                            raw=raw,
                            raw_path=raw_path,
                            opened_raw_disk_state=entry.get("opened_disk_state"),
                            generation_snapshot=generation_snapshot,
                        )
                        try:
                            invalidate_downstream_preproc_steps(context, step)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(
                                "Could not invalidate downstream results after "
                                "accepted %s plot edits: %s",
                                step,
                                exc,
                            )
                        self._refresh_stage_states_from_context()
                        self._refresh_preproc_controls()
                        self.statusBar().showMessage(
                            f"{title_prefix} plot closed: saved edited "
                            f"{raw_path.name}; downstream invalidated."
                        )
        elif step is None:
            self.statusBar().showMessage(f"{title_prefix} plot closed.")
    except Exception as exc:  # noqa: BLE001
        self._show_warning(
            f"{title_prefix} Plot",
            f"Plot close handling failed:\n{exc}",
        )
    finally:
        raw_close = getattr(raw, "close", None)
        if callable(raw_close):
            try:
                raw_close()
            except Exception:
                pass
        active_figures = getattr(self, "_active_plot_figures", None)
        if not registry and not active_figures:
            self._set_global_ui_lock("plot", False)
        if getattr(self, "_mne_browser_shutdown_pending", False) and not registry:
            QTimer.singleShot(0, self._finalize_app_shutdown)


def _request_close_registered_browser(entry: dict[str, Any]) -> None:
    if bool(entry.get("close_requested", False)):
        return
    entry["close_requested"] = True
    for candidate_key in ("browser", "qt_object", "manager_window", "figure"):
        candidate = entry.get(candidate_key)
        close = getattr(candidate, "close", None)
        if callable(close):
            try:
                close()
                return
            except Exception:
                continue


def _request_close_all_mne_browsers(self) -> None:
    registry = getattr(self, "_active_mne_browsers", None)
    if not isinstance(registry, dict) or not registry:
        self._finalize_app_shutdown()
        return
    for entry in list(registry.values()):
        _request_close_registered_browser(entry)
    if not registry:
        self._finalize_app_shutdown()


def _defer_close_for_active_mne_browsers(self, event: Any) -> bool:
    if getattr(self, "_finalizing_mainwindow_close", False):
        return False
    registry = getattr(self, "_active_mne_browsers", None)
    if not isinstance(registry, dict) or not registry:
        return False
    try:
        event.ignore()
    except Exception:
        pass
    if getattr(self, "_mne_browser_shutdown_pending", False):
        return True

    self._mne_browser_shutdown_pending = True
    self._mne_browser_shutdown_excluded_tokens = active_mne_browser_tracking_tokens(
        self
    )
    self.statusBar().showMessage("Closing active preprocess plot windows...")
    app = QApplication.instance()
    if app is not None:
        try:
            previous = bool(app.quitOnLastWindowClosed())
        except Exception:
            previous = True
        self._mne_browser_shutdown_prev_quit_on_last_window_closed = previous
        try:
            app.setQuitOnLastWindowClosed(False)
        except Exception:
            pass
    QTimer.singleShot(0, self._request_close_all_mne_browsers)
    return True


def _attach_plot_autosave(
    self,
    *,
    browser: Any,
    raw: Any,
    raw_path: Path,
    step: str,
    title_prefix: str,
    tracked_open_state: dict[str, Any] | None = None,
) -> None:
    _track_mne_browser(
        self,
        browser=browser,
        raw=raw,
        raw_path=raw_path,
        step=step,
        title_prefix=title_prefix,
        tracked_open_state=tracked_open_state,
    )


def _track_mne_browser(
    self,
    *,
    browser: Any,
    raw: Any,
    raw_path: Path,
    step: str | None,
    title_prefix: str,
    tracked_open_state: dict[str, Any] | None = None,
) -> None:
    registry = getattr(self, "_active_mne_browsers", None)
    if not isinstance(registry, dict):
        registry = {}
        self._active_mne_browsers = registry

    figure, qt_object, manager_window = _browser_tracking_targets(browser)
    token = id(browser)
    context = (
        tracked_open_state["context"]
        if isinstance(tracked_open_state, dict)
        else self._record_context()
    )
    entry: dict[str, Any] = {
        "browser": browser,
        "raw": raw,
        "raw_path": raw_path,
        "step": step,
        "title_prefix": title_prefix,
        "figure": figure,
        "qt_object": qt_object,
        "manager_window": manager_window,
        "context": context,
        "opened_signature": (
            _preproc_plot_raw_signature(raw)
            if step in _PLOT_CHANGE_TRACKED_STEPS
            else None
        ),
        "opened_disk_state": (
            tracked_open_state["opened_disk_state"]
            if step in _PLOT_CHANGE_TRACKED_STEPS
            and isinstance(tracked_open_state, dict)
            else (
                _preproc_plot_disk_state(raw_path)
                if step in _PLOT_CHANGE_TRACKED_STEPS
                else None
            )
        ),
        "generation_snapshot": (
            tracked_open_state.get("generation_snapshot")
            if isinstance(tracked_open_state, dict)
            else (
                _preproc_plot_generation_snapshot(context, step)
                if step in _PLOT_ATOMIC_EDIT_STEPS
                else None
            )
        ),
        "tracked_open_state": tracked_open_state,
        "filter_output_role": (
            "preview"
            if step == "filter" and raw_path.name == "preview_raw.fif"
            else "scientific"
        ),
        "close_requested": False,
        "closed": False,
    }
    registry[token] = entry

    def _on_close(event: Any | None = None) -> None:
        _finalize_tracked_browser_close(self, token, event)

    got_closed = getattr(browser, "gotClosed", None)
    used_mne_closed_signal = False
    if got_closed is not None and hasattr(got_closed, "connect"):
        try:
            got_closed.connect(_on_close)
            used_mne_closed_signal = True
        except Exception:
            used_mne_closed_signal = False

    close_hook_attached = used_mne_closed_signal

    if qt_object is not None and not used_mne_closed_signal:
        try:
            close_filter_cls = self._close_autosave_filter_class()
            close_filter = close_filter_cls(_on_close, qt_object)
            entry["close_filter"] = close_filter
            qt_object.installEventFilter(close_filter)
            close_hook_attached = True
        except Exception:
            entry.pop("close_filter", None)
        try:
            qt_object.destroyed.connect(_on_close)
            close_hook_attached = True
        except Exception:
            pass

    if (
        not used_mne_closed_signal
        and figure is not None
        and hasattr(figure, "canvas")
        and hasattr(figure.canvas, "mpl_connect")
    ):
        try:
            callback_id = figure.canvas.mpl_connect("close_event", _on_close)
            entry["mpl_close_callback_id"] = callback_id
            close_hook_attached = True
        except Exception:
            pass

    if not close_hook_attached:
        # Without a close hook the entry would never leave the registry, which
        # would hold the plot lock (and block app shutdown) forever.
        registry.pop(token, None)
        self.statusBar().showMessage(
            f"{title_prefix} plot is untracked: no close signal is available, "
            "so edits made in this window will not be saved."
        )
        return

    self._set_global_ui_lock("plot", True)


def _open_mne_raw_plot(
    self,
    raw_path: Path,
    title_prefix: str,
    *,
    autosave_step: str | None = None,
) -> None:
    if not self._enable_plots:
        return
    if getattr(self, "_mne_browser_shutdown_pending", False):
        self.statusBar().showMessage(
            f"{title_prefix} Plot unavailable: app shutdown is in progress."
        )
        return
    tracked_open_state = None
    if autosave_step in _PLOT_CHANGE_TRACKED_STEPS:
        try:
            tracked_open_state = _capture_tracked_plot_open_state(
                self,
                raw_path=raw_path,
                step=autosave_step,
            )
        except Exception as exc:
            self.statusBar().showMessage(f"{title_prefix} Plot failed: {exc}")
            return
    try:
        raw = self._read_raw_fif(raw_path, preload=True, verbose="ERROR")
        if autosave_step in _PLOT_CHANGE_TRACKED_STEPS:
            assert isinstance(tracked_open_state, dict)
            stale_reason = _tracked_plot_open_stale_reason(
                raw_path=raw_path,
                step=autosave_step,
                open_state=tracked_open_state,
            )
            if stale_reason is not None:
                raw.close()
                raise RuntimeError(
                    f"{raw_path.name} {stale_reason}; the plot was not opened."
                )
        previous_plot_size = os.environ.get("MNE_BROWSE_RAW_SIZE")
        os.environ["MNE_BROWSE_RAW_SIZE"] = _preproc_plot_figsize_env_value(self)
        try:
            browser = raw.plot(block=False, title=f"{title_prefix}: {raw_path.name}")
        finally:
            if previous_plot_size is None:
                os.environ.pop("MNE_BROWSE_RAW_SIZE", None)
            else:
                os.environ["MNE_BROWSE_RAW_SIZE"] = previous_plot_size
        if autosave_step in _PLOT_CHANGE_TRACKED_STEPS:
            assert isinstance(tracked_open_state, dict)
            stale_reason = _tracked_plot_open_stale_reason(
                raw_path=raw_path,
                step=autosave_step,
                open_state=tracked_open_state,
            )
            if stale_reason is not None:
                _request_close_registered_browser({"browser": browser})
                raw.close()
                raise RuntimeError(
                    f"{raw_path.name} {stale_reason}; the plot was discarded."
                )
        _resize_preproc_plot_window(browser)
        if autosave_step is not None:
            self._attach_plot_autosave(
                browser=browser,
                raw=raw,
                raw_path=raw_path,
                step=autosave_step,
                title_prefix=title_prefix,
                tracked_open_state=tracked_open_state,
            )
        else:
            _track_mne_browser(
                self,
                browser=browser,
                raw=raw,
                raw_path=raw_path,
                step=None,
                title_prefix=title_prefix,
            )
    except Exception as exc:
        self.statusBar().showMessage(f"{title_prefix} Plot failed: {exc}")


def active_mne_browser_tracking_tokens(window: Any) -> set[int]:
    registry = getattr(window, "_active_mne_browsers", None)
    if not isinstance(registry, dict):
        return set()
    tokens: set[int] = set()
    for entry in registry.values():
        for candidate_key in ("browser", "qt_object", "figure", "manager_window"):
            if entry.get(candidate_key) is not None:
                candidate = entry[candidate_key]
                tokens.add(id(candidate))
    return tokens


def active_mne_browser_widget_tokens(window: Any) -> set[int]:
    registry = getattr(window, "_active_mne_browsers", None)
    if not isinstance(registry, dict):
        return set()
    tokens: set[int] = set()
    for entry in registry.values():
        for candidate_key in ("browser", "qt_object", "figure", "manager_window"):
            candidate = entry.get(candidate_key)
            if isinstance(candidate, QWidget):
                tokens.add(id(candidate))
    return tokens
