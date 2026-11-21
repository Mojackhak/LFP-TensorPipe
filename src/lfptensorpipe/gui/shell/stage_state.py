"""Shared stage-state refresh MainWindow methods."""

from __future__ import annotations

from lfptensorpipe.app.alignment.epoch_view import load_alignment_epoch_picks
from lfptensorpipe.app.runlog_store import run_log_read_snapshot
from lfptensorpipe.gui.shell.common import (
    PathResolver,
    STAGE_SPECS,
    scan_stage_states,
)


class MainWindowStageStateMixin:
    def _empty_stage_states(self) -> dict[str, str]:
        return {spec.key: "gray" for spec in STAGE_SPECS}

    def _derive_entry_stage_states(
        self,
        raw_stage_states: dict[str, str],
    ) -> dict[str, str]:
        normalized = self._empty_stage_states()
        for key, value in raw_stage_states.items():
            if key in normalized and value in {"gray", "yellow", "green"}:
                normalized[key] = value

        entry_states = self._empty_stage_states()
        spec_by_key = {spec.key: spec for spec in STAGE_SPECS}
        for spec in STAGE_SPECS:
            raw_state = normalized.get(spec.key, "gray")
            if raw_state == "gray":
                entry_states[spec.key] = "gray"
                continue

            upstream_key = spec.prerequisite_key
            upstream_green = True
            while isinstance(upstream_key, str):
                if entry_states.get(upstream_key) != "green":
                    upstream_green = False
                    break
                upstream_spec = spec_by_key.get(upstream_key)
                upstream_key = (
                    upstream_spec.prerequisite_key
                    if upstream_spec is not None
                    else None
                )
            entry_states[spec.key] = raw_state if upstream_green else "yellow"
        return entry_states

    def _set_stage_state_maps(self, raw_stage_states: dict[str, str]) -> None:
        normalized = self._empty_stage_states()
        for key, value in raw_stage_states.items():
            if key in normalized and value in {"gray", "yellow", "green"}:
                normalized[key] = value
        self._stage_raw_states = dict(normalized)
        self._stage_states = self._derive_entry_stage_states(normalized)

    def _current_trial_stage_states(self) -> dict[str, str]:
        context = self._record_context()
        slug_getter = getattr(self, "_shared_stage_trial_slug", None)
        slug = slug_getter() if callable(slug_getter) else None
        if context is None or not isinstance(slug, str) or not slug.strip():
            return {"alignment": "gray", "features": "gray"}

        resolver = PathResolver(context)
        alignment_state = self._alignment_display_state(resolver, slug)
        features_state = self._extract_features_indicator_state_runtime(
            resolver,
            paradigm_slug=slug,
        )
        if alignment_state != "green" and features_state != "gray":
            features_state = "yellow"
        return {
            "alignment": alignment_state,
            "features": features_state,
        }

    def _alignment_display_state(self, resolver: PathResolver, slug: str) -> str:
        paradigm = next(
            (p for p in self._alignment_paradigms if p["slug"] == slug), None
        )
        if paradigm is None:
            return self._alignment_trial_stage_state_runtime(
                resolver, paradigm_slug=slug
            )
        if (
            self._current_alignment_paradigm_slug() == slug
            and self._alignment_epoch_rows
        ):
            picks = self._selected_alignment_epoch_indices()
        else:
            picks = load_alignment_epoch_picks(resolver.context, paradigm_slug=slug)
        return self._alignment_epoch_inspector_state_runtime(
            resolver, paradigm=paradigm, picked_epoch_indices=picks
        )

    def _refresh_trial_stage_controls(self) -> None:
        """Project the edited trial onto navigation and Features without rescanning the record."""
        with run_log_read_snapshot():
            self._set_stage_state_maps(
                {**self._stage_raw_states, **self._current_trial_stage_states()}
            )
            self._refresh_stage_controls()
            self._refresh_features_trial_list_states(
                only_slug=self._shared_stage_trial_slug()
            )
            self._refresh_features_controls()

    def _refresh_stage_states_from_context(self) -> None:
        if (
            self._current_project is None
            or self._current_subject is None
            or self._current_record is None
        ):
            return
        raw_states = scan_stage_states(
            self._current_project, self._current_subject, self._current_record
        )
        raw_states.update(self._current_trial_stage_states())
        self._set_stage_state_maps(raw_states)
        refresh_features_trials = getattr(
            self, "_refresh_features_trial_list_states", None
        )
        if callable(refresh_features_trials):
            refresh_features_trials()
        self._refresh_stage_controls()
        self._refresh_tensor_controls()
        self._refresh_alignment_controls()
        self._refresh_features_indicators()
        self._refresh_features_controls()
