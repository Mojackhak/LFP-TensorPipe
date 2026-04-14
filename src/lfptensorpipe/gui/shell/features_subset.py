"""Features subset and file-selection MainWindow methods."""

from __future__ import annotations

from lfptensorpipe.gui.shell.common import (
    Any,
    FEATURE_DERIVED_TYPES,
    QComboBox,
    QLineEdit,
    np,
    pd,
)


class MainWindowFeaturesSubsetMixin:
    def _selected_features_file(self) -> dict[str, Any] | None:
        if self._features_available_table is None:
            return None
        row = self._features_available_table.currentRow()
        if row < 0 or row >= len(self._features_filtered_files):
            return None
        return self._features_filtered_files[row]

    @staticmethod
    def _default_plot_labels_for_derived_type(
        derived_type: str,
    ) -> tuple[str | None, str | None]:
        token = derived_type.strip().lower()
        if token == "spectral":
            return "Frequency (Hz)", None
        if token == "trace":
            return "Percent / Time", None
        if token == "raw":
            return "Percent / Time", "Frequency (Hz)"
        return None, None

    @staticmethod
    def _resolve_plot_label(edit: QLineEdit | None, default: str | None) -> str | None:
        if edit is None:
            return default
        token = edit.text().strip()
        if token:
            return token
        return default

    def _apply_features_plot_label_placeholders(self) -> None:
        selected = self._selected_features_file()
        derived_type = (
            str(selected.get("derived_type", "")).strip().lower()
            if isinstance(selected, dict)
            else ""
        )
        x_default, y_default = self._default_plot_labels_for_derived_type(derived_type)
        if self._features_x_label_edit is not None:
            self._features_x_label_edit.setPlaceholderText(x_default or "")
        if self._features_y_label_edit is not None:
            self._features_y_label_edit.setPlaceholderText(y_default or "")
        if self._features_cbar_label_edit is not None:
            self._features_cbar_label_edit.setPlaceholderText("")
            self._features_cbar_label_edit.setEnabled(derived_type == "raw")

    def _on_features_available_selection_changed(self) -> None:
        self._apply_features_plot_label_placeholders()
        self._refresh_features_controls()

    @staticmethod
    def _parse_derived_type_from_stem(stem: str) -> str:
        token = stem.strip()
        if "-" not in token:
            return ""
        derived = token.rsplit("-", 1)[-1].strip().lower()
        return derived if derived in FEATURE_DERIVED_TYPES else ""

    @staticmethod
    def _parse_reducer_from_stem(stem: str, derived_type: str) -> str:
        token = stem.strip()
        suffix = f"-{derived_type}"
        if not derived_type or not token.lower().endswith(suffix):
            return ""
        reducer = token[: -len(suffix)].strip().lower()
        return reducer or ""

    @staticmethod
    def _replace_combo_values(
        combo: QComboBox | None,
        values: list[str],
        *,
        selected: str | None = None,
    ) -> None:
        if combo is None:
            return
        target = combo.currentData() if selected is None else selected
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("All", "")
        for value in values:
            combo.addItem(value, value)
        idx = combo.findData(target)
        if idx < 0:
            idx = 0
        combo.setCurrentIndex(idx)
        combo.setEnabled(bool(values))
        combo.blockSignals(False)

    @staticmethod
    def _subset_unique_values(payload: pd.DataFrame, column: str) -> list[str]:
        if column not in payload.columns:
            return []
        out: list[str] = []
        seen: set[str] = set()
        for item in payload[column].tolist():
            token = str(item).strip()
            if not token or token in seen:
                continue
            seen.add(token)
            out.append(token)
        return sorted(out, key=str.lower)

    @staticmethod
    def _region_names_from_payload(payload: pd.DataFrame) -> list[str]:
        out: list[str] = []
        for col in payload.columns:
            token = str(col)
            if not token.endswith("_in"):
                continue
            series = payload[col]
            valid = False
            truthy = False
            if pd.api.types.is_bool_dtype(series):
                valid = True
                truthy = bool(series.fillna(False).astype(bool).any())
            else:
                numeric = pd.to_numeric(series, errors="coerce")
                finite = numeric.dropna().to_numpy(dtype=float)
                if finite.size and np.all(np.isin(finite, [0.0, 1.0])):
                    valid = True
                    truthy = bool((finite > 0.0).any())
            if valid and truthy:
                out.append(token[:-3])
        out.sort(key=str.lower)
        return out

    def _selected_features_payload(self) -> pd.DataFrame | None:
        selected = self._selected_features_file()
        if selected is None:
            return None
        try:
            payload = self._load_pickle(selected["path"])
        except Exception:
            return None
        if not isinstance(payload, pd.DataFrame):
            return None
        return payload

    def _clear_features_subset_options(self) -> None:
        self._replace_combo_values(
            self._features_subset_band_combo,
            [],
            selected="",
        )
        self._replace_combo_values(
            self._features_subset_channel_combo,
            [],
            selected="",
        )
        self._replace_combo_values(
            self._features_subset_region_combo,
            [],
            selected="",
        )

    def _current_features_subset_selection(self) -> dict[str, str]:
        return {
            "band": (
                str(self._features_subset_band_combo.currentData() or "").strip()
                if self._features_subset_band_combo is not None
                else ""
            ),
            "channel": (
                str(self._features_subset_channel_combo.currentData() or "").strip()
                if self._features_subset_channel_combo is not None
                else ""
            ),
            "region": (
                str(self._features_subset_region_combo.currentData() or "").strip()
                if self._features_subset_region_combo is not None
                else ""
            ),
        }

    def _normalize_features_subset_selection(
        self,
        selection: dict[str, Any] | None,
    ) -> dict[str, str]:
        source = selection if isinstance(selection, dict) else {}
        return {
            "band": str(source.get("band", "")).strip(),
            "channel": str(source.get("channel", "")).strip(),
            "region": str(source.get("region", "")).strip(),
        }

    def _filter_features_payload_by_subset(
        self,
        payload: pd.DataFrame,
        *,
        band: str = "",
        channel: str = "",
        region: str = "",
    ) -> pd.DataFrame:
        out = payload
        if band and "Band" in out.columns:
            out = out[out["Band"].astype(str).str.strip() == band]
        if channel and "Channel" in out.columns:
            out = out[out["Channel"].astype(str).str.strip() == channel]
        if region:
            region_col = f"{region}_in"
            if region_col in out.columns:
                keep = self._coerce_numeric_bool_series(out[region_col])
                out = out[keep.to_numpy(dtype=bool)]
        return out

    def _available_features_subset_bands(
        self,
        payload: pd.DataFrame,
        *,
        channel: str = "",
        region: str = "",
    ) -> list[str]:
        filtered = self._filter_features_payload_by_subset(
            payload,
            channel=channel,
            region=region,
        )
        return self._subset_unique_values(filtered, "Band")

    def _available_features_subset_channels(
        self,
        payload: pd.DataFrame,
        *,
        band: str = "",
        region: str = "",
    ) -> list[str]:
        filtered = self._filter_features_payload_by_subset(
            payload,
            band=band,
            region=region,
        )
        return self._subset_unique_values(filtered, "Channel")

    def _available_features_subset_regions(
        self,
        payload: pd.DataFrame,
        *,
        band: str = "",
        channel: str = "",
    ) -> list[str]:
        filtered = self._filter_features_payload_by_subset(
            payload,
            band=band,
            channel=channel,
        )
        return self._region_names_from_payload(filtered)

    def _sync_features_subset_options(
        self,
        *,
        preferred_selection: dict[str, Any] | None = None,
    ) -> None:
        payload = self._selected_features_payload()
        if payload is None:
            self._clear_features_subset_options()
            return

        target = self._normalize_features_subset_selection(
            preferred_selection
            if preferred_selection is not None
            else self._current_features_subset_selection()
        )
        for _ in range(3):
            band_values = self._available_features_subset_bands(
                payload,
                channel=target["channel"],
                region=target["region"],
            )
            channel_values = self._available_features_subset_channels(
                payload,
                band=target["band"],
                region=target["region"],
            )
            region_values = self._available_features_subset_regions(
                payload,
                band=target["band"],
                channel=target["channel"],
            )

            normalized = {
                "band": target["band"] if target["band"] in band_values else "",
                "channel": (
                    target["channel"] if target["channel"] in channel_values else ""
                ),
                "region": target["region"] if target["region"] in region_values else "",
            }

            self._replace_combo_values(
                self._features_subset_band_combo,
                band_values,
                selected=normalized["band"],
            )
            self._replace_combo_values(
                self._features_subset_channel_combo,
                channel_values,
                selected=normalized["channel"],
            )
            self._replace_combo_values(
                self._features_subset_region_combo,
                region_values,
                selected=normalized["region"],
            )

            if normalized == target:
                break
            target = normalized

    def _refresh_features_subset_options(self) -> None:
        self._sync_features_subset_options()

    def _on_features_subset_changed(self, _index: int) -> None:
        self._refresh_features_subset_options()

    @staticmethod
    def _coerce_numeric_bool_series(series: pd.Series) -> pd.Series:
        if pd.api.types.is_bool_dtype(series):
            return series.fillna(False).astype(bool)
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.fillna(0.0).astype(float) > 0.0

    def _apply_features_subset_filters(self, payload: pd.DataFrame) -> pd.DataFrame:
        subset = self._current_features_subset_selection()
        return self._filter_features_payload_by_subset(
            payload.copy(),
            band=subset["band"],
            channel=subset["channel"],
            region=subset["region"],
        )
