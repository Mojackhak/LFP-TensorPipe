"""Tensor run orchestration MainWindow methods."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from uuid import uuid4

from lfptensorpipe.desktop_runtime import (
    TENSOR_WORKER_FLAG,
    TENSOR_WORKER_MODULE,
    build_worker_command,
)
from lfptensorpipe.app.tensor.cancellation import (
    BUILD_TENSOR_CANCELLED_MESSAGE,
    backfill_cancelled_build_tensor_run,
)
from lfptensorpipe.gui.shell.common import (
    QAction,
    Any,
    QApplication,
    QAbstractButton,
    RecordContext,
    TENSOR_COMMON_BASIC_METRIC_KEYS,
    build_tensor_metric_notch_payload,
)

TENSOR_RUN_TERMINATE_TIMEOUT_S = 2.0
TENSOR_RUN_SUCCESS_LABEL = "Build Tensor"
TENSOR_STOP_LABEL = "Stop"
TRGC_BUSY_SUFFIX = "This may take several hours."


class MainWindowTensorRunMixin:
    @staticmethod
    def _tensor_run_busy_suffix(selected_metrics: list[str]) -> str | None:
        return TRGC_BUSY_SUFFIX if "trgc" in selected_metrics else None

    def _selected_tensor_metrics_snapshot(self) -> list[str]:
        selected: list[str] = []
        for metric_key, checkbox in self._tensor_metric_checks.items():
            if checkbox.isChecked():
                selected.append(metric_key)
        return selected

    def _selected_tensor_metrics(self) -> list[str]:
        selected: list[str] = []
        for metric_key, checkbox in self._tensor_metric_checks.items():
            if checkbox.isEnabled() and checkbox.isChecked():
                selected.append(metric_key)
        return selected

    def _collect_tensor_pairs_for_metrics(
        self,
        selected_metrics: list[str],
    ) -> dict[str, list[tuple[str, str]]]:
        pairs_map: dict[str, list[tuple[str, str]]] = {}
        for metric_key in selected_metrics:
            mode = self._tensor_metric_pair_mode(metric_key)
            if mode is None:
                continue
            pairs_map[metric_key] = [
                tuple(pair)
                for pair in self._tensor_selected_pairs_by_metric.get(metric_key, ())
            ]
        return pairs_map

    def _collect_tensor_bands(self, value: Any) -> list[dict[str, Any]]:
        bands = self._normalize_tensor_bands_rows(value)
        if not bands:
            raise ValueError("At least one band is required.")
        return [dict(item) for item in bands]

    def _collect_tensor_runtime_metric_params(
        self, context: RecordContext, selected_metrics: list[str]
    ) -> dict[str, dict[str, Any]]:
        self._commit_active_tensor_panel_to_params()
        self._sync_tensor_selector_maps_into_metric_params()
        metric_params_map: dict[str, dict[str, Any]] = {}
        for metric_key in selected_metrics:
            params = dict(self._tensor_metric_params.get(metric_key, {}))
            params.update(
                build_tensor_metric_notch_payload(
                    params.get("notches"),
                    params.get("notch_widths"),
                )
            )
            if metric_key in TENSOR_COMMON_BASIC_METRIC_KEYS:
                low_freq = float(params.get("low_freq_hz", 0.0))
                high_freq = float(params.get("high_freq_hz", 0.0))
                step_hz = float(params.get("freq_step_hz", 0.0))
                if low_freq <= 0.0:
                    raise ValueError(
                        f"{self._tensor_metric_display_name(metric_key)} low freq must be > 0."
                    )
                if high_freq <= low_freq:
                    raise ValueError(
                        f"{self._tensor_metric_display_name(metric_key)} high freq must be greater than low freq."
                    )
                if step_hz <= 0.0:
                    raise ValueError(
                        f"{self._tensor_metric_display_name(metric_key)} step must be > 0."
                    )
                valid, message, _ = self._validate_tensor_frequency_params_runtime(
                    context,
                    low_freq=low_freq,
                    high_freq=high_freq,
                    step_hz=step_hz,
                )
                if not valid:
                    raise ValueError(
                        f"{self._tensor_metric_display_name(metric_key)}: {message}"
                    )
            if metric_key in {"psi", "burst"}:
                bands = self._collect_tensor_bands(params.get("bands"))
                params["bands"] = bands
            if metric_key == "periodic_aperiodic":
                freq_range = params.get("freq_range_hz")
                if not isinstance(freq_range, (list, tuple)) or len(freq_range) != 2:
                    raise ValueError("Periodic/APeriodic freq range must be provided.")
                try:
                    range_lo = float(freq_range[0])
                    range_hi = float(freq_range[1])
                except Exception as exc:  # noqa: BLE001
                    raise ValueError(
                        "Periodic/APeriodic freq range must be numeric."
                    ) from exc
                if range_hi <= range_lo:
                    raise ValueError(
                        "Periodic/APeriodic freq range must satisfy high > low."
                    )
                low = float(params.get("low_freq_hz", 0.0))
                high = float(params.get("high_freq_hz", 0.0))
                if low < range_lo or high > range_hi:
                    raise ValueError(
                        "Periodic/APeriodic low/high must stay within SpecParam freq range."
                    )
                step_hz = float(params.get("freq_step_hz", 0.0))
                valid_range, message_range, _ = (
                    self._validate_tensor_frequency_params_runtime(
                        context,
                        low_freq=range_lo,
                        high_freq=range_hi,
                        step_hz=step_hz,
                    )
                )
                if not valid_range:
                    raise ValueError(
                        "Periodic/APeriodic SpecParam freq range is invalid: "
                        f"{message_range}"
                    )
                if bool(params.get("freq_smooth_enabled", True)):
                    sigma = params.get("freq_smooth_sigma")
                    if sigma is not None and float(sigma) <= 0.0:
                        raise ValueError(
                            "Periodic/APeriodic freq smooth sigma must be > 0."
                        )
                if bool(params.get("time_smooth_enabled", True)):
                    kernel_size = params.get("time_smooth_kernel_size")
                    if kernel_size is not None and int(kernel_size) < 1:
                        raise ValueError(
                            "Periodic/APeriodic time smooth kernel size must be >= 1."
                        )
            if metric_key == "psi" and not params.get("bands"):
                raise ValueError("PSI requires at least one configured band.")
            if metric_key == "burst" and not params.get("bands"):
                raise ValueError("Burst requires at least one configured band.")
            if metric_key in self._tensor_selected_channels_by_metric:
                params["selected_channels"] = list(
                    self._tensor_selected_channels_by_metric.get(metric_key, ())
                )
            if metric_key in self._tensor_selected_pairs_by_metric:
                params["selected_pairs"] = [
                    [a, b]
                    for a, b in self._tensor_selected_pairs_by_metric.get(
                        metric_key, ()
                    )
                ]
            metric_params_map[metric_key] = params
        return metric_params_map

    def _collect_tensor_runtime_params(
        self,
        context: RecordContext,
    ) -> tuple[list[str], bool, dict[str, dict[str, Any]]]:
        selected_metrics = self._selected_tensor_metrics()
        if not selected_metrics:
            raise ValueError("Select at least one metric.")
        mask_edge_effects = (
            bool(self._tensor_mask_edge_checkbox.isChecked())
            if self._tensor_mask_edge_checkbox is not None
            else True
        )
        metric_params_map = self._collect_tensor_runtime_metric_params(
            context, selected_metrics
        )
        return selected_metrics, mask_edge_effects, metric_params_map

    def _tensor_run_is_active(self) -> bool:
        return isinstance(self._tensor_run_state, dict)

    def _refresh_tensor_controls_for_active_run(self) -> None:
        for checkbox in self._tensor_metric_checks.values():
            checkbox.setEnabled(False)
        for button in self._tensor_metric_name_buttons.values():
            button.setEnabled(False)
        for widget in self._tensor_basic_param_widgets.values():
            widget.setEnabled(False)
        if self._tensor_channels_button is not None:
            self._tensor_channels_button.setEnabled(False)
        if self._tensor_pairs_button is not None:
            self._tensor_pairs_button.setEnabled(False)
        if self._tensor_advance_button is not None:
            self._tensor_advance_button.setEnabled(False)
        if self._tensor_mask_edge_checkbox is not None:
            self._tensor_mask_edge_checkbox.setEnabled(False)
        if self._tensor_import_button is not None:
            self._tensor_import_button.setEnabled(False)
        if self._tensor_export_button is not None:
            self._tensor_export_button.setEnabled(False)
        if self._tensor_run_button is not None:
            self._tensor_run_button.setText(TENSOR_STOP_LABEL)
            self._tensor_run_button.setEnabled(True)

    def _start_tensor_run_busy(self, label: str, *, suffix: str | None = None) -> None:
        self._busy_label = label
        self._busy_suffix = str(suffix).strip() if suffix else None
        self._busy_frame_idx = 0
        self._busy_timer.start()
        self._render_busy_message()
        app = QApplication.instance()
        if app is not None:
            app.processEvents()

    def _stop_tensor_run_busy(self) -> None:
        self._busy_timer.stop()
        self._busy_label = None
        self._busy_suffix = None
        self._busy_frame_idx = 0

    def _set_tensor_run_ui_lock(self, lock: bool) -> None:
        if lock:
            self._tensor_run_locked_buttons = []
            exempt_buttons = {
                self._tensor_run_button,
            }
            for button in self.findChildren(QAbstractButton):
                try:
                    if button in exempt_buttons or not button.isEnabled():
                        continue
                    button.setEnabled(False)
                    self._tensor_run_locked_buttons.append(button)
                except RuntimeError:
                    continue

            self._tensor_run_locked_actions = []
            for action in self.findChildren(QAction):
                try:
                    if not action.isEnabled():
                        continue
                    action.setEnabled(False)
                    self._tensor_run_locked_actions.append(action)
                except RuntimeError:
                    continue
            return

        for button in self._tensor_run_locked_buttons:
            try:
                button.setEnabled(True)
            except RuntimeError:
                continue
        self._tensor_run_locked_buttons = []

        for action in self._tensor_run_locked_actions:
            try:
                action.setEnabled(True)
            except RuntimeError:
                continue
        self._tensor_run_locked_actions = []

    def _tensor_worker_env(self) -> dict[str, str]:
        env = dict(os.environ)
        pythonpath_entries: list[str] = []
        for entry in sys.path:
            token = str(entry).strip()
            if token and token not in pythonpath_entries:
                pythonpath_entries.append(token)
        for entry in env.get("PYTHONPATH", "").split(os.pathsep):
            token = entry.strip()
            if token and token not in pythonpath_entries:
                pythonpath_entries.append(token)
        if pythonpath_entries:
            env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
        return env

    def _tensor_temp_json_path(self, stem: str) -> Path:
        fd, raw_path = tempfile.mkstemp(prefix=f"lfptensorpipe_{stem}_", suffix=".json")
        os.close(fd)
        path = Path(raw_path)
        path.unlink(missing_ok=True)
        return path

    def _write_tensor_worker_request(
        self, payload: dict[str, Any]
    ) -> tuple[Path, Path]:
        request_path = self._tensor_temp_json_path("tensor_request")
        result_path = self._tensor_temp_json_path("tensor_result")
        request_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return request_path, result_path

    def _read_tensor_worker_result(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None
        return payload if isinstance(payload, dict) else None

    def _cleanup_tensor_run_files(self, state: dict[str, Any] | None) -> None:
        if not isinstance(state, dict):
            return
        for key in ("request_path", "result_path"):
            path = state.get(key)
            if not isinstance(path, Path):
                continue
            path.unlink(missing_ok=True)

    def _launch_tensor_run_process(
        self,
        *,
        context: RecordContext,
        selected_metrics: list[str],
        metric_params_map: dict[str, dict[str, Any]],
        mask_edge_effects: bool,
    ) -> None:
        run_id = uuid4().hex
        request_payload = {
            "context": {
                "project_root": str(context.project_root),
                "subject": context.subject,
                "record": context.record,
            },
            "selected_metrics": list(selected_metrics),
            "metric_params_map": metric_params_map,
            "mask_edge_effects": bool(mask_edge_effects),
            "run_id": run_id,
        }
        request_path, result_path = self._write_tensor_worker_request(request_payload)
        try:
            process = subprocess.Popen(
                build_worker_command(
                    module_name=TENSOR_WORKER_MODULE,
                    embedded_flag=TENSOR_WORKER_FLAG,
                    worker_args=[
                        "--request",
                        str(request_path),
                        "--result",
                        str(result_path),
                    ],
                    python_exec=sys.executable,
                ),
                cwd=str(Path.cwd()),
                env=self._tensor_worker_env(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            request_path.unlink(missing_ok=True)
            result_path.unlink(missing_ok=True)
            raise

        self._tensor_run_state = {
            "process": process,
            "request_path": request_path,
            "result_path": result_path,
            "context": context,
            "selected_metrics": list(selected_metrics),
            "metric_params_map": dict(metric_params_map),
            "mask_edge_effects": bool(mask_edge_effects),
            "run_id": run_id,
            "stop_requested": False,
            "kill_deadline_monotonic": None,
        }
        self._start_tensor_run_busy(
            TENSOR_RUN_SUCCESS_LABEL,
            suffix=self._tensor_run_busy_suffix(selected_metrics),
        )
        self._set_tensor_run_ui_lock(True)
        self._refresh_tensor_controls()
        self._tensor_run_poll_timer.start()

    def _finish_tensor_run(
        self,
        *,
        ok: bool,
        message: str,
        cancelled: bool,
    ) -> None:
        state = self._tensor_run_state
        self._tensor_run_poll_timer.stop()
        self._tensor_run_state = None
        self._cleanup_tensor_run_files(state)
        self._set_tensor_run_ui_lock(False)
        self._stop_tensor_run_busy()
        self._refresh_stage_states_from_context()
        self._refresh_tensor_controls()
        prefix = "Build Tensor cancelled" if cancelled else "Build Tensor OK"
        if not cancelled and not ok:
            prefix = "Build Tensor failed"
        self.statusBar().showMessage(f"{prefix}: {message}")
        self._post_step_action_sync(reason="tensor_run")

    def _request_stop_tensor_run(self) -> None:
        state = self._tensor_run_state
        if not isinstance(state, dict):
            return
        process = state.get("process")
        if process is None:
            return
        try:
            if process.poll() is not None:
                return
        except Exception:  # noqa: BLE001
            return
        if bool(state.get("stop_requested")):
            try:
                process.kill()
            except Exception:  # noqa: BLE001
                pass
            return
        state["stop_requested"] = True
        state["kill_deadline_monotonic"] = (
            time.monotonic() + TENSOR_RUN_TERMINATE_TIMEOUT_S
        )
        self._start_tensor_run_busy("Stopping Build Tensor")
        try:
            process.terminate()
        except Exception:  # noqa: BLE001
            pass

    def _poll_tensor_run_process(self) -> None:
        state = self._tensor_run_state
        if not isinstance(state, dict):
            self._tensor_run_poll_timer.stop()
            return
        process = state.get("process")
        if process is None:
            self._finish_tensor_run(
                ok=False,
                message="Build Tensor process handle is missing.",
                cancelled=False,
            )
            return
        try:
            exit_code = process.poll()
        except Exception:  # noqa: BLE001
            exit_code = None
        if exit_code is None:
            deadline = state.get("kill_deadline_monotonic")
            if (
                bool(state.get("stop_requested"))
                and isinstance(deadline, (int, float))
                and time.monotonic() >= float(deadline)
            ):
                try:
                    process.kill()
                except Exception:  # noqa: BLE001
                    pass
                state["kill_deadline_monotonic"] = None
            return

        result_path = state.get("result_path")
        result_payload = (
            self._read_tensor_worker_result(result_path)
            if isinstance(result_path, Path)
            else None
        )
        if bool(state.get("stop_requested")):
            metric_statuses = backfill_cancelled_build_tensor_run(
                state["context"],
                selected_metrics=state["selected_metrics"],
                metric_params_map=state["metric_params_map"],
                mask_edge_effects=bool(state["mask_edge_effects"]),
                run_id=str(state["run_id"]),
                message=BUILD_TENSOR_CANCELLED_MESSAGE,
            )
            if any(status == "cancelled" for status in metric_statuses.values()):
                self._finish_tensor_run(
                    ok=False,
                    message=BUILD_TENSOR_CANCELLED_MESSAGE,
                    cancelled=True,
                )
                return
        if isinstance(result_payload, dict):
            ok = bool(result_payload.get("ok", False))
            message = str(result_payload.get("message", "")).strip()
            if not message:
                message = "Build Tensor worker exited without a message."
            self._finish_tensor_run(ok=ok, message=message, cancelled=False)
            return

        self._finish_tensor_run(
            ok=False,
            message=f"Build Tensor worker exited unexpectedly (code {exit_code}).",
            cancelled=False,
        )

    def _shutdown_tensor_run(self) -> None:
        state = self._tensor_run_state
        if not isinstance(state, dict):
            return
        process = state.get("process")
        self._tensor_run_poll_timer.stop()
        try:
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=TENSOR_RUN_TERMINATE_TIMEOUT_S)
                except Exception:  # noqa: BLE001
                    try:
                        process.kill()
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        process.wait(timeout=1.0)
                    except Exception:  # noqa: BLE001
                        pass
                backfill_cancelled_build_tensor_run(
                    state["context"],
                    selected_metrics=state["selected_metrics"],
                    metric_params_map=state["metric_params_map"],
                    mask_edge_effects=bool(state["mask_edge_effects"]),
                    run_id=str(state["run_id"]),
                    message=BUILD_TENSOR_CANCELLED_MESSAGE,
                )
        finally:
            self._cleanup_tensor_run_files(state)
            self._tensor_run_state = None
            self._set_tensor_run_ui_lock(False)
            self._stop_tensor_run_busy()

    def _on_tensor_run(self) -> None:
        if self._tensor_run_is_active():
            self._request_stop_tensor_run()
            return
        if self._busy_label is not None:
            self.statusBar().showMessage(
                f"{self._busy_label} is running; duplicate action ignored."
            )
            return

        context = self._record_context()
        if context is None:
            self.statusBar().showMessage(
                "Run Build Tensor unavailable: select project/subject/record."
            )
            return
        if self._stage_states.get("preproc") != "green":
            self._show_warning(
                "Run Build Tensor",
                "Preprocess Signal must be green before Build Tensor can run.",
            )
            return

        try:
            (
                selected_metrics,
                mask_edge_effects,
                metric_params_map,
            ) = self._collect_tensor_runtime_params(context)
        except Exception as exc:  # noqa: BLE001
            self._show_warning(
                "Run Build Tensor",
                f"Invalid tensor parameters:\n{exc}",
            )
            self.statusBar().showMessage(
                f"Build Tensor failed: invalid parameters ({exc})"
            )
            return

        warnings_by_metric: dict[str, list[str]] = {}
        for metric_key in selected_metrics:
            warnings = self._tensor_metric_notch_warnings(
                context,
                metric_key,
                dict(metric_params_map.get(metric_key, {})),
            )
            if warnings:
                warnings_by_metric[metric_key] = warnings
        if warnings_by_metric and not self._confirm_tensor_preflight_notch_warnings(
            warnings_by_metric
        ):
            self.statusBar().showMessage("Build Tensor cancelled before launch.")
            return

        try:
            self._launch_tensor_run_process(
                context=context,
                selected_metrics=selected_metrics,
                metric_params_map=metric_params_map,
                mask_edge_effects=mask_edge_effects,
            )
        except Exception as exc:  # noqa: BLE001
            self._show_warning(
                "Run Build Tensor",
                f"Failed to start Build Tensor:\n{exc}",
            )
            self.statusBar().showMessage(f"Build Tensor failed to start: {exc}")
