"""Tensor selector and pair-state MainWindow methods."""

from __future__ import annotations

from lfptensorpipe.gui.shell.common import (
    Any,
    PathResolver,
    RecordContext,
    TENSOR_CHANNEL_METRIC_KEYS,
    TENSOR_DIRECTED_METRIC_KEYS,
    TENSOR_UNDIRECTED_METRIC_KEYS,
    preproc_step_raw_path,
)


class MainWindowTensorStateSelectorsMixin:
    def _sync_tensor_selector_maps_into_metric_params(self) -> None:
        for metric_key, channels in self._tensor_selected_channels_by_metric.items():
            params = dict(self._tensor_metric_params.get(metric_key, {}))
            params["selected_channels"] = [str(item) for item in channels]
            self._tensor_metric_params[metric_key] = params
        for metric_key, pairs in self._tensor_selected_pairs_by_metric.items():
            params = dict(self._tensor_metric_params.get(metric_key, {}))
            params["selected_pairs"] = [[str(a), str(b)] for a, b in pairs]
            self._tensor_metric_params[metric_key] = params

    @staticmethod
    def _coerce_tensor_channels(value: Any) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)):
            return ()
        deduped: list[str] = []
        for item in value:
            channel = str(item).strip()
            if not channel or channel in deduped:
                continue
            deduped.append(channel)
        return tuple(deduped)

    @classmethod
    def _coerce_tensor_pairs(
        cls,
        value: Any,
        *,
        directed: bool,
    ) -> tuple[tuple[str, str], ...]:
        if not isinstance(value, (list, tuple)):
            return ()
        parsed: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for token in value:
            pair = cls._parse_tensor_pair_token(token)
            if pair is None:
                continue
            try:
                normalized = cls._normalize_tensor_pair(
                    pair[0], pair[1], directed=directed
                )
            except Exception:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            parsed.append(normalized)
        return tuple(parsed)

    @staticmethod
    def _tensor_metric_requires_channel_selector(metric_key: str) -> bool:
        return metric_key in TENSOR_CHANNEL_METRIC_KEYS

    @staticmethod
    def _tensor_metric_pair_mode(metric_key: str) -> str | None:
        if metric_key in TENSOR_UNDIRECTED_METRIC_KEYS:
            return "undirected"
        if metric_key in TENSOR_DIRECTED_METRIC_KEYS:
            return "directed"
        return None

    @staticmethod
    def _parse_tensor_pair_token(value: Any) -> tuple[str, str] | None:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            source = str(value[0]).strip()
            target = str(value[1]).strip()
            if source and target:
                return source, target
            return None
        if not isinstance(value, str):
            return None
        token = value.strip()
        if not token:
            return None
        if token.startswith("(") and token.endswith(")"):
            body = token[1:-1]
            parts = [part.strip() for part in body.split(",", maxsplit=1)]
            if len(parts) == 2 and parts[0] and parts[1]:
                return parts[0], parts[1]
            return None
        if "-" in token:
            source, target = token.split("-", maxsplit=1)
            source = source.strip()
            target = target.strip()
            if source and target:
                return source, target
        return None

    @staticmethod
    def _normalize_tensor_pair(
        source: str,
        target: str,
        *,
        directed: bool,
    ) -> tuple[str, str]:
        src = str(source).strip()
        dst = str(target).strip()
        if not src or not dst:
            raise ValueError("Pair channels cannot be empty.")
        if src == dst:
            raise ValueError("Self-pairs are not allowed.")
        if directed:
            return src, dst
        return tuple(sorted((src, dst)))  # type: ignore[return-value]

    @classmethod
    def _filter_tensor_pairs(
        cls,
        pairs: tuple[tuple[str, str], ...] | list[tuple[str, str]],
        *,
        available_channels: tuple[str, ...],
        directed: bool,
    ) -> tuple[tuple[str, str], ...]:
        allowed = set(available_channels)
        deduped: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for source, target in pairs:
            try:
                pair = cls._normalize_tensor_pair(source, target, directed=directed)
            except Exception:
                continue
            if pair[0] not in allowed or pair[1] not in allowed:
                continue
            if pair in seen:
                continue
            seen.add(pair)
            deduped.append(pair)
        return tuple(deduped)

    @staticmethod
    def _format_pair_button_text(
        selected_pairs: tuple[tuple[str, str], ...],
        available_channels: tuple[str, ...],
        *,
        directed: bool,
    ) -> str:
        count = len(available_channels)
        if directed:
            total = count * max(0, count - 1)
        else:
            total = count * max(0, count - 1) // 2
        return f"Select Pairs ({len(selected_pairs)}/{total})"

    def _refresh_tensor_pair_button_text(self) -> None:
        if self._tensor_pairs_button is None:
            return
        metric_key = self._tensor_active_metric_key
        directed = metric_key in TENSOR_DIRECTED_METRIC_KEYS
        selected = self._tensor_selected_pairs_by_metric.get(metric_key, ())
        self._tensor_pairs_button.setText(
            self._format_pair_button_text(
                selected,
                self._tensor_available_channels,
                directed=directed,
            )
        )

    def _refresh_tensor_channel_state(self, context: RecordContext | None) -> None:
        self._ensure_tensor_metric_state_from_defaults(context)
        if context is None:
            self._tensor_available_channels = ()
            for key in list(self._tensor_selected_channels_by_metric.keys()):
                self._tensor_selected_channels_by_metric[key] = ()
            for key in list(self._tensor_selected_pairs_by_metric.keys()):
                self._tensor_selected_pairs_by_metric[key] = ()
            if self._tensor_channels_button is not None:
                self._tensor_channels_button.setText("Select Channels (0/0)")
                self._tensor_channels_button.setEnabled(False)
            if self._tensor_pairs_button is not None:
                self._tensor_pairs_button.setText("Select Pairs (0/0)")
                self._tensor_pairs_button.setEnabled(False)
            return

        resolver = PathResolver(context)
        raw_path = preproc_step_raw_path(resolver, "finish")
        channels = (
            tuple(self._read_channel_names_from_raw(raw_path))
            if raw_path.exists()
            else ()
        )
        previous_channels = self._tensor_available_channels
        self._tensor_available_channels = channels

        if not channels:
            for key in list(self._tensor_selected_channels_by_metric.keys()):
                self._tensor_selected_channels_by_metric[key] = ()
            for key in list(self._tensor_selected_pairs_by_metric.keys()):
                self._tensor_selected_pairs_by_metric[key] = ()
        else:
            for metric_key in TENSOR_CHANNEL_METRIC_KEYS:
                current = self._tensor_selected_channels_by_metric.get(metric_key, ())
                filtered = tuple(item for item in current if item in set(channels))
                if filtered:
                    self._tensor_selected_channels_by_metric[metric_key] = filtered
                    continue
                if channels != previous_channels:
                    self._tensor_selected_channels_by_metric[metric_key] = (
                        self._tensor_default_selected_channels_for_metric(
                            metric_key,
                            available_channels=channels,
                        )
                    )
            for metric_key in TENSOR_UNDIRECTED_METRIC_KEYS:
                source = self._tensor_selected_pairs_by_metric.get(metric_key, ())
                if channels != previous_channels and not source:
                    source = self._tensor_default_selected_pairs_for_metric(
                        metric_key,
                        directed=False,
                        available_channels=channels,
                    )
                self._tensor_selected_pairs_by_metric[metric_key] = (
                    self._filter_tensor_pairs(
                        source,
                        available_channels=channels,
                        directed=False,
                    )
                )
            for metric_key in TENSOR_DIRECTED_METRIC_KEYS:
                source = self._tensor_selected_pairs_by_metric.get(metric_key, ())
                if channels != previous_channels and not source:
                    source = self._tensor_default_selected_pairs_for_metric(
                        metric_key,
                        directed=True,
                        available_channels=channels,
                    )
                self._tensor_selected_pairs_by_metric[metric_key] = (
                    self._filter_tensor_pairs(
                        source,
                        available_channels=channels,
                        directed=True,
                    )
                )

        if self._tensor_channels_button is not None:
            selected_channels = self._tensor_selected_channels_by_metric.get(
                self._tensor_active_metric_key, ()
            )
            self._tensor_channels_button.setText(
                self._format_channel_button_text(
                    "Select Channels",
                    selected_channels,
                    channels,
                )
            )
        self._refresh_tensor_pair_button_text()
