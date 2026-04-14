"""Preprocess defaults and parameter-normalization MainWindow methods."""

from __future__ import annotations

from lfptensorpipe.gui.shell.common import (
    Any,
    PREPROC_FILTER_BASIC_DEFAULTS_KEY,
    PREPROC_FILTER_DEFAULTS_KEY,
    PREPROC_VIZ_PSD_DEFAULTS_KEY,
    PREPROC_VIZ_TFR_DEFAULTS_KEY,
    default_filter_advance_params,
    default_preproc_filter_basic_params,
    default_preproc_viz_psd_params,
    default_preproc_viz_tfr_params,
    normalize_filter_advance_params,
    normalize_preproc_filter_basic_params,
    normalize_preproc_viz_psd_params,
    normalize_preproc_viz_tfr_params,
)


class MainWindowPreprocDefaultsMixin:
    def _load_filter_advance_defaults(self) -> dict[str, Any]:
        payload = self._config_store.read_yaml("preproc.yml", default={})
        raw_params: dict[str, Any] | None = None
        if isinstance(payload, dict):
            node = payload.get(PREPROC_FILTER_DEFAULTS_KEY)
            if isinstance(node, dict):
                raw_params = node
        ok, normalized, _ = normalize_filter_advance_params(raw_params)
        if ok:
            return normalized
        return default_filter_advance_params()

    def _load_filter_basic_defaults(self) -> dict[str, Any]:
        payload = self._config_store.read_yaml("preproc.yml", default={})
        raw_params: dict[str, Any] | None = None
        if isinstance(payload, dict):
            node = payload.get(PREPROC_FILTER_BASIC_DEFAULTS_KEY)
            if isinstance(node, dict):
                raw_params = node
        ok, normalized, _ = normalize_preproc_filter_basic_params(raw_params)
        if ok:
            return normalized
        return default_preproc_filter_basic_params()

    def _save_filter_advance_defaults(self, params: dict[str, Any]) -> None:
        payload = self._config_store.read_yaml("preproc.yml", default={})
        if not isinstance(payload, dict):
            payload = {}
        payload[PREPROC_FILTER_DEFAULTS_KEY] = params
        self._config_store.write_yaml("preproc.yml", payload)

    def _save_filter_basic_defaults(self, params: dict[str, Any]) -> None:
        ok, normalized, _ = normalize_preproc_filter_basic_params(params)
        payload = self._config_store.read_yaml("preproc.yml", default={})
        if not isinstance(payload, dict):
            payload = {}
        payload[PREPROC_FILTER_BASIC_DEFAULTS_KEY] = (
            normalized if ok else default_preproc_filter_basic_params()
        )
        self._config_store.write_yaml("preproc.yml", payload)

    @staticmethod
    def _format_filter_notches(values: list[float]) -> str:
        return ",".join(f"{float(item):g}" for item in values)

    def _apply_filter_basic_params_to_fields(self, params: dict[str, Any]) -> None:
        ok, normalized, _ = normalize_preproc_filter_basic_params(params)
        if not ok:
            normalized = default_preproc_filter_basic_params()
        if self._preproc_filter_notches_edit is not None:
            self._preproc_filter_notches_edit.setText(
                self._format_filter_notches(normalized["notches"])
            )
        if self._preproc_filter_low_freq_edit is not None:
            self._preproc_filter_low_freq_edit.setText(f"{normalized['l_freq']:g}")
        if self._preproc_filter_high_freq_edit is not None:
            self._preproc_filter_high_freq_edit.setText(f"{normalized['h_freq']:g}")

    def _load_preproc_viz_psd_defaults(self) -> dict[str, Any]:
        payload = self._config_store.read_yaml("preproc.yml", default={})
        raw_params: dict[str, Any] | None = None
        if isinstance(payload, dict):
            node = payload.get(PREPROC_VIZ_PSD_DEFAULTS_KEY)
            if isinstance(node, dict):
                raw_params = node
        ok, normalized, _ = normalize_preproc_viz_psd_params(raw_params)
        if ok:
            return normalized
        return default_preproc_viz_psd_params()

    def _save_preproc_viz_psd_defaults(self, params: dict[str, Any]) -> None:
        payload = self._config_store.read_yaml("preproc.yml", default={})
        if not isinstance(payload, dict):
            payload = {}
        payload[PREPROC_VIZ_PSD_DEFAULTS_KEY] = params
        self._config_store.write_yaml("preproc.yml", payload)

    def _load_preproc_viz_tfr_defaults(self) -> dict[str, Any]:
        payload = self._config_store.read_yaml("preproc.yml", default={})
        raw_params: dict[str, Any] | None = None
        if isinstance(payload, dict):
            node = payload.get(PREPROC_VIZ_TFR_DEFAULTS_KEY)
            if isinstance(node, dict):
                raw_params = node
        ok, normalized, _ = normalize_preproc_viz_tfr_params(raw_params)
        if ok:
            return normalized
        return default_preproc_viz_tfr_params()

    def _save_preproc_viz_tfr_defaults(self, params: dict[str, Any]) -> None:
        payload = self._config_store.read_yaml("preproc.yml", default={})
        if not isinstance(payload, dict):
            payload = {}
        payload[PREPROC_VIZ_TFR_DEFAULTS_KEY] = params
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

    def _collect_filter_runtime_params(self) -> tuple[list[float], float, float]:
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
        notches = self._parse_filter_notches(notches_text)
        low_freq = float(low_text.strip())
        high_freq = float(high_text.strip())
        if low_freq < 0.0:
            raise ValueError("Low freq must be >= 0.")
        if high_freq <= low_freq:
            raise ValueError("High freq must be greater than Low freq.")
        return notches, low_freq, high_freq
