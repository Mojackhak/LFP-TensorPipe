"""Preprocess defaults and parameter-normalization MainWindow methods."""

from __future__ import annotations

from lfptensorpipe.gui.shell.common import (
    Any,
    PREPROC_ECG_DEFAULTS_KEY,
    PREPROC_FILTER_BASIC_DEFAULTS_KEY,
    PREPROC_FILTER_DEFAULTS_KEY,
    PREPROC_VIZ_PSD_DEFAULTS_KEY,
    PREPROC_VIZ_TFR_DEFAULTS_KEY,
    default_filter_advance_params,
    default_preproc_filter_basic_params,
    default_preproc_viz_psd_params,
    default_preproc_viz_tfr_params,
    normalize_filter_advance_params,
    normalize_ecg_method_params,
    normalize_ecg_params_by_method,
    normalize_preproc_filter_basic_params,
    normalize_preproc_viz_psd_params,
    normalize_preproc_viz_tfr_params,
    set_control_validation_error,
)


class MainWindowPreprocDefaultsMixin:
    def _show_preproc_params_warning_once(self, message: str) -> None:
        shown = getattr(self, "_preproc_params_warnings_shown", set())
        if message in shown:
            return
        shown.add(message)
        self._preproc_params_warnings_shown = shown
        self.statusBar().showMessage(message)

    def _show_ecg_params_warning_once(self, message: str) -> None:
        self._show_preproc_params_warning_once(message)

    def _load_ecg_advance_defaults(self) -> dict[str, dict[str, Any]]:
        payload = self._config_store.read_yaml("preproc.yml", default={})
        raw_params: Any = None
        if isinstance(payload, dict) and PREPROC_ECG_DEFAULTS_KEY in payload:
            raw_params = payload.get(PREPROC_ECG_DEFAULTS_KEY)
        ok, normalized, message = normalize_ecg_params_by_method(raw_params)
        if not ok:
            self._show_ecg_params_warning_once(
                f"Invalid ECG Advance defaults were replaced in memory: {message}"
            )
        return normalized

    def _save_ecg_method_defaults(
        self,
        method: str,
        params: dict[str, Any],
    ) -> None:
        normalized_method = str(method).strip().lower()
        ok, normalized, message = normalize_ecg_method_params(
            normalized_method,
            params,
        )
        if not ok:
            raise ValueError(message)
        defaults = self._load_ecg_advance_defaults()
        defaults[normalized_method] = normalized
        payload = self._config_store.read_yaml("preproc.yml", default={})
        if not isinstance(payload, dict):
            payload = {}
        payload[PREPROC_ECG_DEFAULTS_KEY] = defaults
        self._config_store.write_yaml("preproc.yml", payload)

    def _load_filter_advance_defaults(self) -> dict[str, Any]:
        payload = self._config_store.read_yaml("preproc.yml", default={})
        raw_params: dict[str, Any] | None = None
        if isinstance(payload, dict):
            node = payload.get(PREPROC_FILTER_DEFAULTS_KEY)
            if isinstance(node, dict):
                raw_params = node
        ok, normalized, message = normalize_filter_advance_params(raw_params)
        if ok:
            return normalized
        self._show_preproc_params_warning_once(
            f"Invalid Filter Advance defaults were replaced in memory: {message}"
        )
        return default_filter_advance_params()

    def _load_filter_basic_defaults(self) -> dict[str, Any]:
        payload = self._config_store.read_yaml("preproc.yml", default={})
        raw_params: dict[str, Any] | None = None
        if isinstance(payload, dict):
            node = payload.get(PREPROC_FILTER_BASIC_DEFAULTS_KEY)
            if isinstance(node, dict):
                raw_params = node
        ok, normalized, message = normalize_preproc_filter_basic_params(raw_params)
        if ok:
            return normalized
        self._show_preproc_params_warning_once(
            f"Invalid Filter defaults were replaced in memory: {message}"
        )
        return default_preproc_filter_basic_params()

    def _save_filter_advance_defaults(self, params: dict[str, Any]) -> None:
        ok, normalized, message = normalize_filter_advance_params(params)
        if not ok:
            raise ValueError(message)
        payload = self._config_store.read_yaml("preproc.yml", default={})
        if not isinstance(payload, dict):
            payload = {}
        payload[PREPROC_FILTER_DEFAULTS_KEY] = normalized
        self._config_store.write_yaml("preproc.yml", payload)

    def _save_filter_basic_defaults(self, params: dict[str, Any]) -> None:
        ok, normalized, message = normalize_preproc_filter_basic_params(params)
        if not ok:
            raise ValueError(message)
        payload = self._config_store.read_yaml("preproc.yml", default={})
        if not isinstance(payload, dict):
            payload = {}
        payload[PREPROC_FILTER_BASIC_DEFAULTS_KEY] = normalized
        self._config_store.write_yaml("preproc.yml", payload)

    @staticmethod
    def _format_filter_notches(values: list[float]) -> str:
        return ",".join(f"{float(item):g}" for item in values)

    @staticmethod
    def _format_optional_filter_frequency(value: float | None) -> str:
        return "" if value is None else f"{float(value):g}"

    def _apply_filter_basic_params_to_fields(self, params: dict[str, Any]) -> None:
        defaults = default_preproc_filter_basic_params()
        candidate = dict(defaults)
        if isinstance(params, dict):
            candidate.update({key: params[key] for key in defaults if key in params})
        if self._preproc_filter_notches_edit is not None:
            raw_notches = candidate["notches"]
            if isinstance(raw_notches, (list, tuple)):
                text = ",".join(str(item) for item in raw_notches)
            elif raw_notches is None:
                text = ""
            else:
                text = str(raw_notches)
            self._preproc_filter_notches_edit.setText(text)
        if self._preproc_filter_low_freq_edit is not None:
            value = candidate["l_freq"]
            self._preproc_filter_low_freq_edit.setText(
                "" if value is None else str(value)
            )
        if self._preproc_filter_high_freq_edit is not None:
            value = candidate["h_freq"]
            self._preproc_filter_high_freq_edit.setText(
                "" if value is None else str(value)
            )
        self._refresh_preproc_filter_basic_validation()

    def _refresh_preproc_filter_basic_validation(self) -> tuple[bool, str]:
        edits = (
            self._preproc_filter_notches_edit,
            self._preproc_filter_low_freq_edit,
            self._preproc_filter_high_freq_edit,
        )
        for edit in edits:
            set_control_validation_error(edit, None)
        params = {
            "notches": (
                self._preproc_filter_notches_edit.text()
                if self._preproc_filter_notches_edit is not None
                else ""
            ),
            "l_freq": (
                self._preproc_filter_low_freq_edit.text()
                if self._preproc_filter_low_freq_edit is not None
                else ""
            ),
            "h_freq": (
                self._preproc_filter_high_freq_edit.text()
                if self._preproc_filter_high_freq_edit is not None
                else ""
            ),
        }
        valid, _, message = normalize_preproc_filter_basic_params(params)
        if valid:
            return True, ""
        lowered = message.lower()
        if "notch" in lowered:
            targets = (self._preproc_filter_notches_edit,)
        elif "l_freq" in lowered and "h_freq" not in lowered:
            targets = (self._preproc_filter_low_freq_edit,)
        elif "h_freq" in lowered and "l_freq" not in lowered:
            targets = (self._preproc_filter_high_freq_edit,)
        else:
            targets = (
                self._preproc_filter_low_freq_edit,
                self._preproc_filter_high_freq_edit,
            )
        for edit in targets:
            set_control_validation_error(edit, message)
        return False, message

    def _load_preproc_viz_psd_defaults(self) -> dict[str, Any]:
        payload = self._config_store.read_yaml("preproc.yml", default={})
        raw_params: dict[str, Any] | None = None
        if isinstance(payload, dict):
            node = payload.get(PREPROC_VIZ_PSD_DEFAULTS_KEY)
            if isinstance(node, dict):
                raw_params = node
        ok, normalized, message = normalize_preproc_viz_psd_params(raw_params)
        if ok:
            return normalized
        self._show_preproc_params_warning_once(
            f"Invalid PSD defaults were replaced in memory: {message}"
        )
        return default_preproc_viz_psd_params()

    def _save_preproc_viz_psd_defaults(self, params: dict[str, Any]) -> None:
        ok, normalized, message = normalize_preproc_viz_psd_params(params)
        if not ok:
            raise ValueError(message)
        payload = self._config_store.read_yaml("preproc.yml", default={})
        if not isinstance(payload, dict):
            payload = {}
        payload[PREPROC_VIZ_PSD_DEFAULTS_KEY] = normalized
        self._config_store.write_yaml("preproc.yml", payload)

    def _load_preproc_viz_tfr_defaults(self) -> dict[str, Any]:
        payload = self._config_store.read_yaml("preproc.yml", default={})
        raw_params: dict[str, Any] | None = None
        if isinstance(payload, dict):
            node = payload.get(PREPROC_VIZ_TFR_DEFAULTS_KEY)
            if isinstance(node, dict):
                raw_params = node
        ok, normalized, message = normalize_preproc_viz_tfr_params(raw_params)
        if ok:
            return normalized
        self._show_preproc_params_warning_once(
            f"Invalid TFR defaults were replaced in memory: {message}"
        )
        return default_preproc_viz_tfr_params()

    def _save_preproc_viz_tfr_defaults(self, params: dict[str, Any]) -> None:
        ok, normalized, message = normalize_preproc_viz_tfr_params(params)
        if not ok:
            raise ValueError(message)
        payload = self._config_store.read_yaml("preproc.yml", default={})
        if not isinstance(payload, dict):
            payload = {}
        payload[PREPROC_VIZ_TFR_DEFAULTS_KEY] = normalized
        self._config_store.write_yaml("preproc.yml", payload)

    @staticmethod
    def _parse_filter_notches(text: str) -> list[float]:
        parts = [item.strip() for item in text.split(",") if item.strip()]
        if not parts:
            return []
        values = [float(item) for item in parts]
        if any(value <= 0.0 for value in values):
            raise ValueError("Notches must be positive numbers.")
        return values

    def _collect_filter_runtime_params(
        self,
    ) -> tuple[list[float], float | None, float | None]:
        notches_text = (
            self._preproc_filter_notches_edit.text()
            if self._preproc_filter_notches_edit is not None
            else ""
        )
        low_text = (
            self._preproc_filter_low_freq_edit.text()
            if self._preproc_filter_low_freq_edit is not None
            else ""
        )
        high_text = (
            self._preproc_filter_high_freq_edit.text()
            if self._preproc_filter_high_freq_edit is not None
            else ""
        )
        valid, normalized, message = normalize_preproc_filter_basic_params(
            {
                "notches": notches_text,
                "l_freq": low_text,
                "h_freq": high_text,
            }
        )
        if not valid:
            raise ValueError(message)
        return normalized["notches"], normalized["l_freq"], normalized["h_freq"]
