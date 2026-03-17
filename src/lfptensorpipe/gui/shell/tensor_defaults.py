"""Tensor defaults mixin facade."""

from __future__ import annotations

from .tensor_defaults_bands import (
    _load_tensor_bands_defaults as _load_tensor_bands_defaults_impl,
    _load_tensor_metric_bands_defaults as _load_tensor_metric_bands_defaults_impl,
    _normalize_tensor_bands_rows as _normalize_tensor_bands_rows_impl,
    _save_tensor_metric_bands_defaults as _save_tensor_metric_bands_defaults_impl,
)
from .tensor_defaults_metric import (
    _default_tensor_metric_params as _default_tensor_metric_params_impl,
    _load_tensor_metric_default_params as _load_tensor_metric_default_params_impl,
    _load_tensor_metric_defaults_payload as _load_tensor_metric_defaults_payload_impl,
    _save_tensor_metric_default_params as _save_tensor_metric_default_params_impl,
    _tensor_effective_metric_defaults as _tensor_effective_metric_defaults_impl,
    _tensor_metric_default_override_node as _tensor_metric_default_override_node_impl,
    _tensor_prepare_metric_default_payload as _tensor_prepare_metric_default_payload_impl,
    _tensor_supported_methods as _tensor_supported_methods_impl,
)
from .tensor_defaults_selectors import (
    _tensor_default_selected_channels_for_metric as _tensor_default_selected_channels_for_metric_impl,
    _tensor_default_selected_pairs_for_metric as _tensor_default_selected_pairs_for_metric_impl,
    _tensor_load_default_channels as _tensor_load_default_channels_impl,
    _tensor_load_default_pairs as _tensor_load_default_pairs_impl,
    _tensor_read_selector_defaults_payload as _tensor_read_selector_defaults_payload_impl,
    _tensor_save_default_channels as _tensor_save_default_channels_impl,
    _tensor_save_default_pairs as _tensor_save_default_pairs_impl,
)


class MainWindowTensorDefaultsMixin:
    @staticmethod
    def _normalize_tensor_bands_rows(value):
        return _normalize_tensor_bands_rows_impl(value)

    def _load_tensor_bands_defaults(self):
        return _load_tensor_bands_defaults_impl(self)

    def _load_tensor_metric_bands_defaults(self, metric_key: str):
        return _load_tensor_metric_bands_defaults_impl(self, metric_key)

    def _save_tensor_metric_bands_defaults(self, metric_key: str, bands):
        _save_tensor_metric_bands_defaults_impl(self, metric_key, bands)

    @staticmethod
    def _tensor_supported_methods():
        return _tensor_supported_methods_impl()

    def _default_tensor_metric_params(self, metric_key: str, *, context):
        return _default_tensor_metric_params_impl(self, metric_key, context=context)

    def _load_tensor_metric_defaults_payload(self):
        return _load_tensor_metric_defaults_payload_impl(self)

    def _load_tensor_metric_default_params(self, metric_key: str, *, context):
        return _load_tensor_metric_default_params_impl(
            self, metric_key, context=context
        )

    def _save_tensor_metric_default_params(self, metric_key: str, params):
        _save_tensor_metric_default_params_impl(self, metric_key, params)

    def _tensor_metric_default_override_node(self, metric_key: str):
        return _tensor_metric_default_override_node_impl(self, metric_key)

    def _tensor_default_selected_channels_for_metric(
        self, metric_key: str, *, available_channels
    ):
        return _tensor_default_selected_channels_for_metric_impl(
            self,
            metric_key,
            available_channels=available_channels,
        )

    def _tensor_default_selected_pairs_for_metric(
        self, metric_key: str, *, directed: bool, available_channels
    ):
        return _tensor_default_selected_pairs_for_metric_impl(
            self,
            metric_key,
            directed=directed,
            available_channels=available_channels,
        )

    def _tensor_effective_metric_defaults(
        self, metric_key: str, *, context, available_channels=None
    ):
        return _tensor_effective_metric_defaults_impl(
            self,
            metric_key,
            context=context,
            available_channels=available_channels,
        )

    def _tensor_prepare_metric_default_payload(self, metric_key: str, payload):
        return _tensor_prepare_metric_default_payload_impl(self, metric_key, payload)

    def _tensor_read_selector_defaults_payload(self):
        return _tensor_read_selector_defaults_payload_impl(self)

    def _tensor_load_default_channels(self):
        return _tensor_load_default_channels_impl(self)

    def _tensor_load_default_pairs(self, *, directed: bool):
        return _tensor_load_default_pairs_impl(self, directed=directed)

    def _tensor_save_default_channels(self, channels):
        _tensor_save_default_channels_impl(self, channels)

    def _tensor_save_default_pairs(self, pairs, *, directed: bool):
        _tensor_save_default_pairs_impl(self, pairs, directed=directed)
