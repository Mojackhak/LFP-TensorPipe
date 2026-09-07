"""Tensor selector and pair-state MainWindow methods."""

from __future__ import annotations

from lfptensorpipe.app.tensor.selectors import (
    coerce_tensor_channels,
    coerce_tensor_pairs,
    filter_tensor_pairs,
    load_tensor_channel_inventory,
    normalize_tensor_pair,
    parse_tensor_pair_token,
    tensor_metric_pair_mode,
)
from lfptensorpipe.gui.shell.common import (
    TENSOR_CHANNEL_METRIC_KEYS,
    TENSOR_DIRECTED_METRIC_KEYS,
    TENSOR_UNDIRECTED_METRIC_KEYS,
    PathResolver,
    RecordContext,
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

    _coerce_tensor_channels = staticmethod(coerce_tensor_channels)

    _coerce_tensor_pairs = staticmethod(coerce_tensor_pairs)

    @staticmethod
    def _tensor_metric_requires_channel_selector(metric_key: str) -> bool:
        return metric_key in TENSOR_CHANNEL_METRIC_KEYS

    _tensor_metric_pair_mode = staticmethod(tensor_metric_pair_mode)

    _parse_tensor_pair_token = staticmethod(parse_tensor_pair_token)

    _normalize_tensor_pair = staticmethod(normalize_tensor_pair)

    _filter_tensor_pairs = staticmethod(filter_tensor_pairs)

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
        if context is None:
            self._tensor_channel_inventory = None
            self._tensor_channel_inventory_path = None
            self._tensor_available_channels = ()
            for key in list(self._tensor_selected_channels_by_metric.keys()):
                self._tensor_selected_channels_by_metric[key] = ()
            for key in list(self._tensor_selected_pairs_by_metric.keys()):
                self._tensor_selected_pairs_by_metric[key] = ()
            inventory_tooltip = (
                "Select a record to view available Tensor channels and pairs."
            )
            if self._tensor_channels_button is not None:
                self._tensor_channels_button.setText("Select Channels (0/0)")
                self._tensor_channels_button.setEnabled(False)
                self._tensor_channels_button.setToolTip(inventory_tooltip)
            if self._tensor_pairs_button is not None:
                self._tensor_pairs_button.setText("Select Pairs (0/0)")
                self._tensor_pairs_button.setEnabled(False)
                self._tensor_pairs_button.setToolTip(inventory_tooltip)
            return

        resolver = PathResolver(context)
        raw_path = preproc_step_raw_path(resolver, "finish")
        try:
            inventory = (
                load_tensor_channel_inventory(raw_path) if raw_path.exists() else None
            )
        except Exception:
            inventory = None
        channels = inventory.usable_channels if inventory is not None else ()
        previous_inventory = self._tensor_channel_inventory
        previous_inventory_path = self._tensor_channel_inventory_path
        self._tensor_channel_inventory = inventory
        self._tensor_channel_inventory_path = raw_path
        self._tensor_available_channels = channels
        self._ensure_tensor_metric_state_from_defaults(context)
        restore_defaults_for_inventory = inventory is not None and (
            previous_inventory is None or previous_inventory_path != raw_path
        )

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
                if restore_defaults_for_inventory:
                    self._tensor_selected_channels_by_metric[metric_key] = (
                        self._tensor_default_selected_channels_for_metric(
                            metric_key,
                            available_channels=channels,
                        )
                    )
                else:
                    self._tensor_selected_channels_by_metric[metric_key] = ()
            for metric_key in TENSOR_UNDIRECTED_METRIC_KEYS:
                source = self._tensor_selected_pairs_by_metric.get(metric_key, ())
                if restore_defaults_for_inventory and not source:
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
                if restore_defaults_for_inventory and not source:
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
        if inventory is None:
            inventory_tooltip = "Preprocess Finish channel inventory is unavailable."
        elif inventory.bad_channels:
            inventory_tooltip = (
                f"Usable Finish channels: {len(inventory.usable_channels)}/"
                f"{len(inventory.all_channels)}. Excluded bad channel(s): "
                + ", ".join(inventory.bad_channels)
                + "."
            )
        else:
            inventory_tooltip = (
                f"Usable Finish channels: {len(inventory.usable_channels)}/"
                f"{len(inventory.all_channels)}."
            )
        if self._tensor_channels_button is not None:
            self._tensor_channels_button.setToolTip(inventory_tooltip)
        if self._tensor_pairs_button is not None:
            self._tensor_pairs_button.setToolTip(inventory_tooltip)
        self._sync_tensor_selector_maps_into_metric_params()
        self._refresh_tensor_pair_button_text()
