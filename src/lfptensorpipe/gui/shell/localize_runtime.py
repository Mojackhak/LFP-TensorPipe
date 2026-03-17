"""Localize runtime and config MainWindow methods."""

from __future__ import annotations

from lfptensorpipe.app.localize.paths import normalize_localize_path_payload
from lfptensorpipe.gui.shell.common import (
    Any,
    DISABLE_MATLAB_WARMUP_ENV,
    LOCALIZE_PATH_CONFIG_FILENAME,
    LOCALIZE_PATH_FIELD_LABELS,
    QApplication,
    QDialog,
    QProgressDialog,
    Qt,
    is_stale_context_message,
    os,
    matlab_runtime_status,
    reset_matlab_runtime,
    time,
    warmup_matlab_async,
)


class MainWindowLocalizeRuntimeMixin:
    def _load_paths_config_values(self) -> dict[str, str]:
        payload = self._config_store.read_yaml(
            LOCALIZE_PATH_CONFIG_FILENAME,
            default={},
        )
        source, changed = normalize_localize_path_payload(payload)
        if changed:
            self._config_store.write_yaml(LOCALIZE_PATH_CONFIG_FILENAME, source)
        values: dict[str, str] = {}
        for key in LOCALIZE_PATH_FIELD_LABELS:
            values[key] = str(source.get(key, "")).strip()
        return values

    def _save_paths_config_values(self, values: dict[str, str]) -> None:
        payload = self._config_store.read_yaml(
            LOCALIZE_PATH_CONFIG_FILENAME,
            default={},
        )
        if not isinstance(payload, dict):
            payload = {}
        payload.pop("matlab_engine_path", None)
        payload.update(values)
        self._config_store.write_yaml(LOCALIZE_PATH_CONFIG_FILENAME, payload)

    def _on_settings_configs(self) -> None:
        dialog = self._create_paths_config_dialog(
            current_paths=self._load_paths_config_values(),
            parent=self,
        )
        if dialog.exec() != QDialog.Accepted or dialog.selected_paths is None:
            return
        self._save_paths_config_values(dialog.selected_paths)
        self._localize_paths = self._load_localize_paths_runtime(self._config_store)
        self._localize_matlab_failures_shown = set()
        reset_matlab_runtime(paths=self._localize_paths)
        self._refresh_localize_controls()
        self.statusBar().showMessage("Path settings saved to app storage.")

    def _refresh_localize_matlab_status(self) -> None:
        state, message = matlab_runtime_status()
        label_map = {
            "idle": "Idle",
            "starting": "Starting",
            "ready": "Ready",
            "failed": "Failed",
        }
        label = label_map.get(state, state.title())
        text = f"MATLAB: {label}"
        if message and state == "failed":
            text = f"{text} ({message})"
        if self._localize_matlab_status_label is not None:
            self._localize_matlab_status_label.setText(text)
            tooltip = "MATLAB runtime status. Actions auto-connect when needed."
            if message:
                tooltip = f"{tooltip} Details: {message}"
            self._localize_matlab_status_label.setToolTip(tooltip)
            if state == "ready":
                self._localize_matlab_status_label.setStyleSheet("color: #1f7a1f;")
            elif state == "starting":
                self._localize_matlab_status_label.setStyleSheet("color: #8a6d00;")
            elif state == "failed":
                self._localize_matlab_status_label.setStyleSheet("color: #a22d2d;")
            else:
                self._localize_matlab_status_label.setStyleSheet("color: #666666;")
        self._refresh_localize_action_state()

    def _poll_localize_matlab_status(self) -> None:
        self._refresh_localize_matlab_status()

    def _schedule_matlab_warmup(self) -> None:
        disabled = os.environ.get(DISABLE_MATLAB_WARMUP_ENV, "").strip().lower()
        if disabled in {"1", "true", "yes", "on"}:
            return
        state, _ = matlab_runtime_status()
        if state != "idle":
            return
        warmup_matlab_async(self._localize_paths)
        self._refresh_localize_matlab_status()

    def _wait_future_with_cancel(
        self,
        *,
        future: Any,
        title: str,
        label: str,
        timeout_s: float,
    ) -> tuple[bool, str, bool]:
        dialog = QProgressDialog(label, "Cancel", 0, 0, self)
        dialog.setWindowTitle(title)
        dialog.setWindowModality(Qt.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setValue(0)
        dialog.show()
        app = QApplication.instance()
        started = time.monotonic()
        while not future.done():
            if dialog.wasCanceled():
                dialog.close()
                return False, "Cancelled.", True
            elapsed = time.monotonic() - started
            if elapsed >= timeout_s:
                dialog.close()
                message = f"MATLAB startup timed out after {int(timeout_s)}s."
                return False, message, False
            if app is not None:
                app.processEvents()
            time.sleep(0.05)
        dialog.close()
        try:
            payload = future.result()
        except Exception as exc:  # noqa: BLE001
            return False, f"{exc}", False
        if isinstance(payload, tuple) and len(payload) >= 2:
            ok = bool(payload[0])
            message = str(payload[1])
            return ok, message, False
        return True, "", False

    def _ensure_matlab_ready_for_action(self, action_name: str) -> bool:
        state, _ = matlab_runtime_status()
        if state == "ready":
            return True
        future = warmup_matlab_async(self._localize_paths)
        ok, message, cancelled = self._wait_future_with_cancel(
            future=future,
            title=f"{action_name}: MATLAB",
            label="Connecting to MATLAB...",
            timeout_s=60.0,
        )
        self._refresh_localize_matlab_status()
        if cancelled:
            self.statusBar().showMessage(
                f"{action_name} cancelled while MATLAB startup continues in background."
            )
            return False
        if not ok:
            if is_stale_context_message(message):
                self.statusBar().showMessage(
                    f"{action_name} cancelled: request superseded by newer context."
                )
                return False
            token = message.strip()
            if token and token not in self._localize_matlab_failures_shown:
                self._localize_matlab_failures_shown.add(token)
                self._show_warning("Localize MATLAB", message)
            self.statusBar().showMessage(f"{action_name} unavailable: {message}")
            return False
        return True
