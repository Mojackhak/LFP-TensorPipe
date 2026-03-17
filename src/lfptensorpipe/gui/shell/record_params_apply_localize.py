"""Record-parameter localize apply MainWindow methods."""

from __future__ import annotations

from lfptensorpipe.gui.shell.common import (
    Any,
    _nested_get,
)


class MainWindowRecordParamsApplyLocalizeMixin:
    def _apply_record_params_localize_snapshot(self, snapshot: dict[str, Any]) -> int:
        skipped = 0

        if "localize.atlas" not in self._record_param_dirty_keys:
            atlas = _nested_get(snapshot, ("localize", "atlas"))
            selected_regions = _nested_get(snapshot, ("localize", "selected_regions"))
            if isinstance(atlas, str):
                self._localize_selected_atlas = (
                    atlas if atlas in self._localize_available_atlases else None
                )
            else:
                self._localize_selected_atlas = None
            if self._localize_selected_atlas is not None:
                self._localize_selected_regions = (
                    self._normalize_localize_selected_regions(
                        self._localize_selected_atlas,
                        selected_regions,
                    )
                )
            else:
                self._localize_selected_regions = ()
        else:
            skipped += 1

        if "localize.match" not in self._record_param_dirty_keys:
            match_payload = _nested_get(snapshot, ("localize", "match"))
            if isinstance(match_payload, dict):
                self._localize_match_payload = dict(match_payload)
                self._localize_match_completed = bool(
                    self._localize_match_payload.get("completed", False)
                )
        else:
            skipped += 1

        return skipped
