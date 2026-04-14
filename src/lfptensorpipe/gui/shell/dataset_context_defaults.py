"""Dataset-context defaults MainWindow methods."""

from __future__ import annotations

from lfptensorpipe.gui.shell.common import (
    RECORD_IMPORT_DEFAULTS_KEY,
    RECORD_IMPORT_LAST_TYPE_KEY,
    RECORD_IMPORT_TYPES,
)


class MainWindowDatasetContextDefaultsMixin:
    def _load_record_import_last_type(self) -> str:
        payload = self._config_store.read_yaml("recent_projects.yml", default={})
        if not isinstance(payload, dict):
            return RECORD_IMPORT_TYPES[0]
        defaults = payload.get(RECORD_IMPORT_DEFAULTS_KEY)
        if not isinstance(defaults, dict):
            return RECORD_IMPORT_TYPES[0]
        value = str(defaults.get(RECORD_IMPORT_LAST_TYPE_KEY, "")).strip()
        if value in RECORD_IMPORT_TYPES:
            return value
        return RECORD_IMPORT_TYPES[0]

    def _save_record_import_last_type(self, import_type: str) -> None:
        normalized = str(import_type).strip()
        if normalized not in RECORD_IMPORT_TYPES:
            return
        payload = self._config_store.read_yaml("recent_projects.yml", default={})
        if not isinstance(payload, dict):
            payload = {}
        defaults = payload.get(RECORD_IMPORT_DEFAULTS_KEY)
        if not isinstance(defaults, dict):
            defaults = {}
        defaults[RECORD_IMPORT_LAST_TYPE_KEY] = normalized
        payload[RECORD_IMPORT_DEFAULTS_KEY] = defaults
        self._config_store.write_yaml("recent_projects.yml", payload)
