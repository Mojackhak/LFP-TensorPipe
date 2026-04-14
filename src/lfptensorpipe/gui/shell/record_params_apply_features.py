"""Record-parameter features apply MainWindow methods."""

from __future__ import annotations

from lfptensorpipe.gui.shell.common import (
    Any,
    _nested_get,
)


class MainWindowRecordParamsApplyFeaturesMixin:
    def _apply_record_params_features_snapshot(self, snapshot: dict[str, Any]) -> int:
        current_slug = self._current_features_paradigm_slug()
        current_dirty_params: dict[str, Any] | None = None
        if any(key.startswith("features.") for key in self._record_param_dirty_keys):
            if isinstance(current_slug, str) and current_slug:
                current_dirty_params = self._collect_current_features_trial_params()

        trial_params_by_slug = self._normalize_features_trial_params_map(
            _nested_get(snapshot, ("features", "trial_params_by_slug"))
        )
        if not trial_params_by_slug:
            legacy_slug, legacy_params = self._legacy_features_trial_params_snapshot(
                snapshot
            )
            if isinstance(legacy_slug, str) and isinstance(legacy_params, dict):
                trial_params_by_slug[legacy_slug] = legacy_params

        self._features_trial_params_by_slug = trial_params_by_slug
        if (
            current_dirty_params is not None
            and isinstance(current_slug, str)
            and current_slug
        ):
            self._features_trial_params_by_slug[current_slug] = current_dirty_params

        return self._restore_features_trial_params(
            current_slug,
            respect_dirty_keys=True,
        )
