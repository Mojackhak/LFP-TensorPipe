"""Features plotting and export MainWindow methods."""

from __future__ import annotations
from lfptensorpipe.gui.shell.common import (
    Any,
    FEATURE_PLOT_COLORMAPS,
    Path,
    PathResolver,
    QDialog,
    np,
    normalize_feature_plot_transform_mode,
    pd,
)


class MainWindowFeaturesPlottingMixin:
    def _features_plot_advance_capabilities(
        self,
        payload: pd.DataFrame,
        derived_type: str,
    ) -> tuple[bool, bool, bool]:
        allow_x_log = True
        allow_y_log = True
        allow_normalize = derived_type != "scalar"
        sample: Any = None
        if "Value" in payload.columns:
            for item in payload["Value"].tolist():
                if isinstance(item, (pd.Series, pd.DataFrame)):
                    sample = item
                    break
                if item is not None and not (
                    isinstance(item, float) and np.isnan(item)
                ):
                    sample = item
                    break
        if isinstance(sample, pd.DataFrame):
            x_has_str = any(
                isinstance(v, (str, np.str_)) for v in sample.columns.tolist()
            )
            y_has_str = any(
                isinstance(v, (str, np.str_)) for v in sample.index.tolist()
            )
            allow_x_log = not x_has_str
            allow_y_log = not y_has_str
        elif isinstance(sample, pd.Series):
            x_has_str = any(
                isinstance(v, (str, np.str_)) for v in sample.index.tolist()
            )
            allow_x_log = not x_has_str
            allow_y_log = True
        else:
            allow_x_log = False
            allow_y_log = False
            allow_normalize = False
        return allow_x_log, allow_y_log, allow_normalize

    def _apply_features_plot_advance(
        self,
        payload: pd.DataFrame,
        *,
        derived_type: str,
    ) -> pd.DataFrame:
        processed = payload.copy()
        transform_mode = normalize_feature_plot_transform_mode(
            self._features_plot_advance_params.get("transform_mode", "none")
        )
        normalize_mode = str(
            self._features_plot_advance_params.get("normalize_mode", "none")
        ).strip()
        baseline_mode = str(
            self._features_plot_advance_params.get("baseline_mode", "mean")
        ).strip()
        baseline_ranges = self._features_plot_advance_params.get(
            "baseline_percent_ranges",
            [],
        )
        if transform_mode != "none":
            processed = self._transform_dataframe(
                processed,
                value_col="Value",
                mode=transform_mode,
                out_col=None,
                drop_empty=True,
            )
        has_nested_value = False
        if "Value" in processed.columns:
            for item in processed["Value"].tolist():
                if isinstance(item, (pd.Series, pd.DataFrame)):
                    has_nested_value = True
                    break
        if derived_type == "scalar":
            normalize_mode = "none"
        if normalize_mode != "none" and has_nested_value:
            if not isinstance(baseline_ranges, list) or not baseline_ranges:
                raise ValueError("Plot normalization requires baseline percent ranges.")
            processed = self._normalize_dataframe_by_baseline(
                processed,
                baseline=baseline_ranges,
                value_col="Value",
                mode_baseline=baseline_mode,  # type: ignore[arg-type]
                mode=normalize_mode,  # type: ignore[arg-type]
                slice_mode="percent",
            )
        return processed

    @staticmethod
    def _flatten_cell_for_xlsx(value: Any) -> str:
        if isinstance(value, pd.DataFrame):
            return value.to_json(orient="split")
        if isinstance(value, pd.Series):
            return value.to_json(orient="split")
        if value is None:
            return ""
        if isinstance(value, float) and np.isnan(value):
            return ""
        return str(value)

    def _save_plot_data_xlsx(self, payload: pd.DataFrame, path: Path) -> None:
        xlsx_df = payload.copy()
        if "Value" in xlsx_df.columns:
            xlsx_df["Value"] = xlsx_df["Value"].map(self._flatten_cell_for_xlsx)
        xlsx_df.to_excel(path, index=False)

    def _tighten_features_plot_figure(
        self,
        fig: Any,
        *,
        pad_in: float = 0.03,
    ) -> None:
        canvas = getattr(fig, "canvas", None)
        if canvas is None:
            return
        try:
            from matplotlib.transforms import Bbox

            canvas.draw()
            renderer = canvas.get_renderer()
            tight_bbox = fig.get_tightbbox(renderer)
            if tight_bbox is None:
                return
            bbox = Bbox.from_extents(
                float(tight_bbox.x0),
                float(tight_bbox.y0),
                float(tight_bbox.x1),
                float(tight_bbox.y1),
            )
            if (
                not np.all(np.isfinite([bbox.x0, bbox.y0, bbox.x1, bbox.y1]))
                or bbox.width <= 0.0
                or bbox.height <= 0.0
            ):
                return

            fig_w_in, fig_h_in = fig.get_size_inches()
            new_w_in = float(bbox.width) + 2.0 * float(pad_in)
            new_h_in = float(bbox.height) + 2.0 * float(pad_in)
            if new_w_in <= 0.0 or new_h_in <= 0.0:
                return

            for ax in fig.axes:
                pos = ax.get_position()
                ax.set_position(
                    [
                        ((pos.x0 * fig_w_in) - bbox.x0 + pad_in) / new_w_in,
                        ((pos.y0 * fig_h_in) - bbox.y0 + pad_in) / new_h_in,
                        (pos.width * fig_w_in) / new_w_in,
                        (pos.height * fig_h_in) / new_h_in,
                    ]
                )

            for text in getattr(fig, "texts", []):
                if text.get_transform() is not fig.transFigure:
                    continue
                x_frac, y_frac = text.get_position()
                text.set_position(
                    (
                        ((float(x_frac) * fig_w_in) - bbox.x0 + pad_in) / new_w_in,
                        ((float(y_frac) * fig_h_in) - bbox.y0 + pad_in) / new_h_in,
                    )
                )

            for legend in getattr(fig, "legends", []):
                anchor_box = legend.get_bbox_to_anchor()
                if anchor_box is None:
                    continue
                anchor_box = anchor_box.transformed(fig.transFigure.inverted())
                legend.set_bbox_to_anchor(
                    Bbox.from_bounds(
                        ((float(anchor_box.x0) * fig_w_in) - bbox.x0 + pad_in)
                        / new_w_in,
                        ((float(anchor_box.y0) * fig_h_in) - bbox.y0 + pad_in)
                        / new_h_in,
                        (float(anchor_box.width) * fig_w_in) / new_w_in,
                        (float(anchor_box.height) * fig_h_in) / new_h_in,
                    ),
                    transform=fig.transFigure,
                )

            fig.set_size_inches(new_w_in, new_h_in, forward=True)
            if hasattr(canvas, "draw_idle"):
                canvas.draw_idle()
        except Exception:
            return

    def _on_features_plot_advance(self) -> None:
        selected = self._selected_features_file()
        if selected is None:
            self._show_warning("Plot Advance", "Select one feature first.")
            return
        derived_type = str(selected.get("derived_type", "")).strip().lower()
        try:
            payload = self._load_pickle(Path(selected["path"]))
            if not isinstance(payload, pd.DataFrame):
                raise ValueError("Selected feature payload is not a DataFrame.")
        except Exception as exc:  # noqa: BLE001
            self._show_warning(
                "Plot Advance",
                f"Failed to read selected feature:\n{exc}",
            )
            return
        allow_x_log, allow_y_log, allow_normalize = (
            self._features_plot_advance_capabilities(
                payload,
                derived_type,
            )
        )

        def _save_plot_advance_defaults(params: dict[str, Any]) -> None:
            self._save_features_plot_advance_defaults(params)
            self.statusBar().showMessage("Plot advance defaults saved.")

        dialog = self._create_features_plot_advance_dialog(
            session_params=dict(self._features_plot_advance_params),
            default_params=self._load_features_plot_advance_defaults(),
            allow_x_log=allow_x_log,
            allow_y_log=allow_y_log,
            allow_normalize=allow_normalize,
            set_default_callback=_save_plot_advance_defaults,
            parent=self,
        )
        if dialog.exec() != QDialog.Accepted or dialog.selected_params is None:
            return
        self._features_plot_advance_params = dict(dialog.selected_params)
        self.statusBar().showMessage("Plot advance params updated.")
        self._mark_record_param_dirty("features.plot_advance")

    def _on_features_plot_export(self) -> None:
        if self._features_last_plot_figure is None or not isinstance(
            self._features_last_plot_data, pd.DataFrame
        ):
            self._show_warning("Export", "Run Plot first.")
            return
        selected = self._selected_features_file()
        if isinstance(selected, dict) and "path" in selected:
            default_dir = Path(str(selected["path"])).parent
        else:
            context = self._record_context()
            slug = self._current_features_paradigm_slug()
            if context is None or not isinstance(slug, str):
                default_dir = Path.cwd()
            else:
                default_dir = PathResolver(context).features_root / slug
        default_name = self._features_last_plot_name or "features-plot"
        fig_path_text, _ = self._save_file_name(
            "Export Plot Figure",
            str((default_dir / f"{default_name}.png").resolve()),
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;All files (*)",
        )
        if not fig_path_text:
            return
        fig_path = Path(fig_path_text)
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._features_last_plot_figure.savefig(
                fig_path, dpi=300, bbox_inches="tight"
            )
            data_pkl = fig_path.with_suffix(".pkl")
            data_xlsx = fig_path.with_suffix(".xlsx")
            self._save_pickle(self._features_last_plot_data, data_pkl)
            self._save_plot_data_xlsx(self._features_last_plot_data, data_xlsx)
            self.statusBar().showMessage(
                f"Exported figure and plot data: {fig_path.name}, {data_pkl.name}, {data_xlsx.name}"
            )
        except Exception as exc:  # noqa: BLE001
            self._show_warning("Export", f"Export failed:\n{exc}")

    def _on_features_plot(self) -> None:
        selected = self._selected_features_file()
        if selected is None:
            self._show_warning(
                "Plot",
                "Select exactly one available feature file first.",
            )
            return
        if not self._enable_plots:
            return
        path = Path(selected["path"])
        try:
            payload = self._load_pickle(path)
            if not isinstance(payload, pd.DataFrame):
                raise ValueError("Selected file is not a DataFrame payload.")
            if "Value" not in payload.columns:
                raise ValueError("Selected file is missing required Value column.")
            derived_type = str(selected.get("derived_type", "")).strip().lower()
            if derived_type in {"raw", "trace"} and "Phase" not in payload.columns:
                payload = payload.copy()
                payload["Phase"] = "All"
            payload = self._apply_features_subset_filters(payload)
            if payload.empty:
                raise ValueError("No rows remain after subset filtering.")

            has_series = False
            has_df = False
            has_scalar = False
            for item in payload["Value"].tolist():
                if isinstance(item, pd.Series):
                    has_series = True
                elif isinstance(item, pd.DataFrame):
                    has_df = True
                elif np.isscalar(item) and not (
                    isinstance(item, float) and np.isnan(item)
                ):
                    has_scalar = True
                elif item is None or (isinstance(item, float) and np.isnan(item)):
                    continue
                else:
                    raise ValueError(f"Unsupported nested Value type: {type(item)!r}")
            if has_series and has_df:
                raise ValueError("Mixed nested Value types are not supported.")
            if sum([has_series, has_df, has_scalar]) > 1:
                raise ValueError("Mixed Value types are not supported.")
            if not has_series and not has_df and not has_scalar:
                raise ValueError("No plottable nested Value data found.")

            payload = self._apply_features_plot_advance(
                payload, derived_type=derived_type
            )
            allow_x_log, allow_y_log, _ = self._features_plot_advance_capabilities(
                payload,
                derived_type,
            )
            x_default, y_default = self._default_plot_labels_for_derived_type(
                derived_type
            )
            x_label = self._resolve_plot_label(self._features_x_label_edit, x_default)
            y_label = self._resolve_plot_label(self._features_y_label_edit, y_default)
            cbar_label = self._resolve_plot_label(self._features_cbar_label_edit, None)
            metric_key = str(selected.get("metric", "")).strip()
            plot_params = self._load_features_plot_params(metric_key, derived_type)
            x_log = (
                bool(self._features_plot_advance_params.get("x_log", False))
                and allow_x_log
            )
            y_log = (
                bool(self._features_plot_advance_params.get("y_log", False))
                and allow_y_log
            )
            colormap = str(
                self._features_plot_advance_params.get("colormap", "viridis")
            ).strip()
            if colormap not in FEATURE_PLOT_COLORMAPS:
                colormap = "viridis"
            palette_name = colormap
            if colormap == "cmcrameri.vik":
                try:
                    cmap_value = self._load_cmcrameri_vik()
                    palette_name = "cmc.vik"
                except Exception as exc:  # noqa: BLE001
                    raise ValueError(
                        "cmcrameri.vik requires cmcrameri to be installed."
                    ) from exc
            else:
                cmap_value = colormap

            common_plot_kwargs = {
                "title": None,
                "boxsize": plot_params["boxsize"],
                "title_fontsize": plot_params["font_size"],
                "axis_label_fontsize": plot_params["font_size"],
                "tick_label_fontsize": plot_params["tick_label_size"],
                "x_label_offset_mm": plot_params["x_label_offset_mm"],
                "y_label_offset_mm": plot_params["y_label_offset_mm"],
            }

            if has_series:
                fig = self._plot_single_effect_series(
                    payload,
                    value_col="Value",
                    x_label=x_label,
                    y_label=y_label,
                    x_log=x_log,
                    y_log=y_log,
                    line_palette=palette_name,
                    legend_loc=plot_params["legend_position"],
                    **common_plot_kwargs,
                )
            elif has_df:
                df_vmode = "sym" if colormap == "cmcrameri.vik" else "auto"
                fig = self._plot_single_effect_df(
                    payload,
                    value_col="Value",
                    x_label=x_label,
                    y_label=y_label,
                    colorbar_label=cbar_label,
                    x_log=x_log,
                    y_log=y_log,
                    cmap=cmap_value,
                    vmode=df_vmode,
                    colorbar_pad_mm=plot_params["colorbar_pad_mm"],
                    cbar_label_offset_mm=plot_params["cbar_label_offset_mm"],
                    **common_plot_kwargs,
                )
            else:
                scalar_x_var = (
                    "Phase"
                    if "Phase" in payload.columns
                    else ("Band" if "Band" in payload.columns else "Channel")
                )
                if scalar_x_var not in payload.columns:
                    payload = payload.copy()
                    payload["_Row"] = np.arange(len(payload), dtype=int)
                    scalar_x_var = "_Row"
                fig = self._plot_single_effect_scalar(
                    payload,
                    value_col="Value",
                    x_var=scalar_x_var,
                    x_label=x_label,
                    y_label=y_label,
                    x_log=x_log,
                    y_log=y_log,
                    jitter_palette=palette_name,
                    fill_palette=palette_name,
                    legend_loc=plot_params["legend_position"],
                    **common_plot_kwargs,
                )
            self._features_last_plot_figure = fig
            self._features_last_plot_data = payload
            self._features_last_plot_name = str(path.stem)
            self._tighten_features_plot_figure(fig)
            self._refresh_features_controls()
            fig.show()
        except Exception as exc:  # noqa: BLE001
            self._show_warning("Plot", f"Plot failed:\n{exc}")
