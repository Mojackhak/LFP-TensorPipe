"""Record-parameter UI-state IO MainWindow methods."""

from __future__ import annotations

from lfptensorpipe.gui.shell.common import (
    Any,
    PathResolver,
    RecordContext,
    read_run_log,
    read_ui_state,
    write_ui_state,
)


class MainWindowRecordParamsStoreMixin:
    def _load_record_params_payload(
        self, context: RecordContext
    ) -> tuple[bool, dict[str, Any], str]:
        resolver = PathResolver(context)
        path = resolver.record_ui_state_path(create=False)
        if not path.is_file():
            legacy_path = resolver.lfp_root / "lfptensorpipe_log.json"
            if not legacy_path.is_file():
                return True, {}, ""
            try:
                legacy_payload = read_run_log(legacy_path)
            except Exception as exc:  # noqa: BLE001
                return False, {}, str(exc)
            if not isinstance(legacy_payload, dict):
                return True, {}, ""
            params = legacy_payload.get("params", {})
            if not isinstance(params, dict):
                return (
                    False,
                    {},
                    "Legacy record params log field `params` must be a dict.",
                )
            return True, params, ""
        try:
            payload = read_ui_state(path)
        except Exception as exc:  # noqa: BLE001
            return False, {}, str(exc)
        if not isinstance(payload, dict):
            return True, {}, ""
        return True, payload, ""

    def _write_record_params_payload(
        self,
        context: RecordContext,
        *,
        params: dict[str, Any],
        reason: str,
    ) -> bool:
        _ = reason
        path = PathResolver(context).record_ui_state_path(create=True)
        try:
            write_ui_state(path, params)
            return True
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(f"参数日志写入失败: {exc}")
            return False
