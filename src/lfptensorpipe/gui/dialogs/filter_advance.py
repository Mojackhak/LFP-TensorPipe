"""Filter-advance dialog."""

from __future__ import annotations

from .common import *  # noqa: F403


class FilterAdvanceDialog(QDialog):
    """Advance dialog for preprocess filter parameters."""

    def __init__(
        self,
        *,
        session_params: dict[str, Any],
        default_params: dict[str, Any],
        set_default_callback: Callable[[dict[str, Any]], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Filter Advance")
        self.setModal(True)
        self.resize(520, 250)
        self._selected_action: str | None = None
        self._selected_params: dict[str, Any] | None = None
        self._default_params = default_params
        self._set_default_callback = set_default_callback
        self._restore_callback: Callable[[], None] | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)

        self._notch_widths_edit = QLineEdit()
        self._epoch_dur_edit = QLineEdit()
        self._p2p_thresh_edit = QLineEdit()
        self._autoreject_factor_edit = QLineEdit()
        self._isolate_bad_boundaries_check = QCheckBox()
        self._mark_filter_edges_check = QCheckBox()

        notch_widths_tooltip = (
            "Notch filter bandwidth (Hz) used for each notch in Filter > Notches. "
            "Use one value (e.g., 2) or a comma-separated list. Values must be > 0."
        )
        epoch_dur_tooltip = (
            "Epoch length in seconds for bad-segment detection. Smaller values catch "
            "short artifacts; larger values are smoother. Must be > 0."
        )
        p2p_tooltip = (
            "Peak-to-peak amplitude range in Volts: min,max. Epochs outside this "
            "range are marked BAD. Leave blank to disable only fixed peak-to-peak "
            "rejection; AutoReject remains active. Otherwise require two finite "
            "values with 0 <= min < max (e.g., 1e-6,1e-3)."
        )
        autoreject_tooltip = (
            "Multiplier for AutoReject channel thresholds. Higher values are more "
            "tolerant (fewer rejections); lower values are stricter. Must be > 0."
        )
        isolate_bad_boundaries_tooltip = (
            "On (default): filter every valid interval between BAD/EDGE boundaries "
            "independently so reviewed BAD values cannot enter an adjacent filter "
            "input. Off: filter the reviewed Raw continuously."
        )
        mark_filter_edges_tooltip = (
            "Mark the exact combined band-pass and notch FIR support at every "
            "filtered interval edge as EDGE_filter. Off (default) accepts MNE "
            "padding results without adding these annotations."
        )

        notch_widths_label = QLabel("notch widths")
        notch_widths_label.setToolTip(notch_widths_tooltip)
        self._notch_widths_edit.setToolTip(notch_widths_tooltip)
        form.addRow(notch_widths_label, self._notch_widths_edit)

        epoch_dur_label = QLabel("epoch duration")
        epoch_dur_label.setToolTip(epoch_dur_tooltip)
        self._epoch_dur_edit.setToolTip(epoch_dur_tooltip)
        form.addRow(epoch_dur_label, self._epoch_dur_edit)

        p2p_label = QLabel("peak-to-peak threshold (min, max)")
        p2p_label.setToolTip(p2p_tooltip)
        self._p2p_thresh_edit.setToolTip(p2p_tooltip)
        form.addRow(p2p_label, self._p2p_thresh_edit)

        autoreject_label = QLabel("autoreject correct factor")
        autoreject_label.setToolTip(autoreject_tooltip)
        self._autoreject_factor_edit.setToolTip(autoreject_tooltip)
        form.addRow(autoreject_label, self._autoreject_factor_edit)

        isolate_bad_boundaries_label = QLabel("isolate BAD boundaries")
        isolate_bad_boundaries_label.setToolTip(isolate_bad_boundaries_tooltip)
        self._isolate_bad_boundaries_check.setToolTip(isolate_bad_boundaries_tooltip)
        form.addRow(
            isolate_bad_boundaries_label,
            self._isolate_bad_boundaries_check,
        )

        mark_filter_edges_label = QLabel("mark filter edges")
        mark_filter_edges_label.setToolTip(mark_filter_edges_tooltip)
        self._mark_filter_edges_check.setToolTip(mark_filter_edges_tooltip)
        form.addRow(mark_filter_edges_label, self._mark_filter_edges_check)
        root.addLayout(form)

        button_row = QWidget()
        button_layout = QHBoxLayout(button_row)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(6)
        save_button = QPushButton("Save")
        default_button = QPushButton("Set as Default")
        restore_button = QPushButton("Restore Defaults")
        cancel_button = QPushButton("Cancel")

        default_button.setToolTip(
            "Save current Advance values and Filter basic values (Notches, Low freq, "
            "High freq) as defaults."
        )
        restore_button.setToolTip(
            "Restore saved defaults for Advance values and Filter basic values "
            "(Notches, Low freq, High freq)."
        )
        save_button.setToolTip(
            "Apply these advanced filter parameters to the current session."
        )
        cancel_button.setToolTip("Close without changing session values.")

        save_button.clicked.connect(lambda: self._on_submit("save"))
        default_button.clicked.connect(lambda: self._on_submit("set_default"))
        restore_button.clicked.connect(self._on_restore_defaults)
        cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(save_button)
        button_layout.addWidget(default_button)
        button_layout.addWidget(restore_button)
        button_layout.addWidget(cancel_button)
        button_layout.addStretch(1)
        root.addWidget(button_row)

        self._apply_to_fields(session_params)
        self._isolate_bad_boundaries_check.toggled.connect(
            self._sync_mark_filter_edges_enabled
        )
        self._mark_filter_edges_check.toggled.connect(self._refresh_validation)
        for edit in (
            self._notch_widths_edit,
            self._epoch_dur_edit,
            self._p2p_thresh_edit,
            self._autoreject_factor_edit,
        ):
            edit.editingFinished.connect(self._refresh_validation)
        self._refresh_validation()

    @property
    def selected_action(self) -> str | None:
        return self._selected_action

    @property
    def selected_params(self) -> dict[str, Any] | None:
        return self._selected_params

    def _show_warning(self, title: str, message: str) -> int:
        return QMessageBox.warning(self, title, message)

    def set_restore_callback(self, callback: Callable[[], None] | None) -> None:
        self._restore_callback = callback

    def _on_restore_defaults(self) -> None:
        self._apply_to_fields(self._default_params)
        self._refresh_validation()
        if self._restore_callback is not None:
            self._restore_callback()

    @staticmethod
    def _stringify_notch_widths(value: float | list[float]) -> str:
        if isinstance(value, list):
            return ", ".join(f"{item:g}" for item in value)
        return f"{float(value):g}"

    @staticmethod
    def _draft_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return f"{value:g}"
        return str(value)

    def _apply_to_fields(self, params: dict[str, Any]) -> None:
        notch_value = params.get("notch_widths", 2.0)
        p2p_value = params.get("p2p_thresh", [1e-6, 1e-3])
        if p2p_value is None:
            p2p_text = ""
        elif isinstance(p2p_value, (list, tuple)) and len(p2p_value) == 2:
            p2p_text = ", ".join(self._draft_text(item) for item in p2p_value)
        else:
            p2p_text = str(p2p_value)

        if isinstance(notch_value, list):
            notch_text = ", ".join(self._draft_text(item) for item in notch_value)
        else:
            notch_text = self._draft_text(notch_value)
        self._notch_widths_edit.setText(notch_text)
        self._epoch_dur_edit.setText(self._draft_text(params.get("epoch_dur", 1.0)))
        self._p2p_thresh_edit.setText(p2p_text)
        self._autoreject_factor_edit.setText(
            self._draft_text(params.get("autoreject_correct_factor", 1.5))
        )
        self._isolate_bad_boundaries_check.setChecked(
            params.get("isolate_bad_boundaries", True) is True
        )
        self._mark_filter_edges_check.setChecked(
            params.get("mark_filter_edges", False) is True
        )
        self._sync_mark_filter_edges_enabled(
            self._isolate_bad_boundaries_check.isChecked()
        )

    def _sync_mark_filter_edges_enabled(self, isolated: bool) -> None:
        self._mark_filter_edges_check.setEnabled(bool(isolated))
        if not isolated:
            self._mark_filter_edges_check.setChecked(False)
        self._refresh_validation()

    @staticmethod
    def _parse_notch_widths(text: str) -> float | list[float]:
        parts = [item.strip() for item in text.split(",") if item.strip()]
        if not parts:
            raise ValueError("notch_widths cannot be empty.")
        if len(parts) == 1:
            return float(parts[0])
        return [float(item) for item in parts]

    @staticmethod
    def _parse_p2p_thresh(text: str) -> list[float] | None:
        parts = [item.strip() for item in text.split(",") if item.strip()]
        if not parts:
            return None
        if len(parts) != 2:
            raise ValueError(
                "p2p_thresh must be empty or provided as two numbers: min,max."
            )
        return [float(parts[0]), float(parts[1])]

    @staticmethod
    def _draft_number(text: str) -> float | str | None:
        token = text.strip()
        if not token:
            return None
        try:
            value = float(token)
        except Exception:
            return token
        return value if np.isfinite(value) else token

    @classmethod
    def _draft_number_list(
        cls,
        text: str,
        *,
        empty_value: Any,
    ) -> Any:
        token = text.strip()
        if not token:
            return empty_value
        parts = [item.strip() for item in token.split(",")]
        if any(not item for item in parts):
            return token
        parsed = [cls._draft_number(item) for item in parts]
        if any(not isinstance(item, float) for item in parsed):
            return token
        return parsed[0] if len(parsed) == 1 else parsed

    def _collect_draft_params(self) -> dict[str, Any]:
        return {
            "notch_widths": self._draft_number_list(
                self._notch_widths_edit.text(),
                empty_value=None,
            ),
            "epoch_dur": self._draft_number(self._epoch_dur_edit.text()),
            "p2p_thresh": self._draft_number_list(
                self._p2p_thresh_edit.text(),
                empty_value=None,
            ),
            "autoreject_correct_factor": self._draft_number(
                self._autoreject_factor_edit.text()
            ),
            "isolate_bad_boundaries": (self._isolate_bad_boundaries_check.isChecked()),
            "mark_filter_edges": self._mark_filter_edges_check.isChecked(),
        }

    def _collect_params(self) -> dict[str, Any]:
        candidate = self._collect_draft_params()
        valid, normalized, message = normalize_filter_advance_params(candidate)
        if not valid:
            raise ValueError(message)
        return normalized

    def _refresh_validation(self) -> None:
        controls = (
            self._notch_widths_edit,
            self._epoch_dur_edit,
            self._p2p_thresh_edit,
            self._autoreject_factor_edit,
        )
        for control in controls:
            set_control_validation_error(control, None)
        valid, _, message = normalize_filter_advance_params(
            self._collect_draft_params()
        )
        if valid:
            return
        lowered = message.lower()
        if "notch_width" in lowered:
            targets = (self._notch_widths_edit,)
        elif "epoch_dur" in lowered:
            targets = (self._epoch_dur_edit,)
        elif "p2p" in lowered:
            targets = (self._p2p_thresh_edit,)
        elif "autoreject" in lowered:
            targets = (self._autoreject_factor_edit,)
        else:
            targets = controls
        for control in targets:
            set_control_validation_error(control, message)

    def _on_submit(self, action: str) -> None:
        if action == "set_default":
            try:
                params = self._collect_params()
            except Exception as exc:  # noqa: BLE001
                self._refresh_validation()
                self._show_warning("Filter Advance", f"Invalid parameters:\n{exc}")
                return
            if self._set_default_callback is not None:
                try:
                    self._set_default_callback(dict(params))
                except Exception as exc:  # noqa: BLE001
                    self._show_warning(
                        self.windowTitle(), f"Set as default failed:\n{exc}"
                    )
                    return
            self._default_params = dict(params)
            self._selected_action = action
            self._selected_params = params
            return
        params = self._collect_draft_params()
        self._selected_action = action
        self._selected_params = params
        self.accept()
