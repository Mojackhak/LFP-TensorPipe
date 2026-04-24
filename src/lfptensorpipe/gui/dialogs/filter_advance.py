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
        self.resize(520, 220)
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
            "range are marked BAD Must satisfy 0 <= min < max (e.g., 1e-6,1e-3)."
        )
        autoreject_tooltip = (
            "Multiplier for AutoReject channel thresholds. Higher values are more "
            "tolerant (fewer rejections); lower values are stricter. Must be > 0."
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
        if self._restore_callback is not None:
            self._restore_callback()

    @staticmethod
    def _stringify_notch_widths(value: float | list[float]) -> str:
        if isinstance(value, list):
            return ", ".join(f"{item:g}" for item in value)
        return f"{float(value):g}"

    def _apply_to_fields(self, params: dict[str, Any]) -> None:
        notch_value = params.get("notch_widths", 2.0)
        p2p_value = params.get("p2p_thresh", [1e-6, 1e-3])
        if not isinstance(p2p_value, (list, tuple)) or len(p2p_value) != 2:
            p2p_value = [1e-6, 1e-3]

        self._notch_widths_edit.setText(self._stringify_notch_widths(notch_value))
        self._epoch_dur_edit.setText(f"{float(params.get('epoch_dur', 1.0)):g}")
        self._p2p_thresh_edit.setText(
            f"{float(p2p_value[0]):g}, {float(p2p_value[1]):g}"
        )
        self._autoreject_factor_edit.setText(
            f"{float(params.get('autoreject_correct_factor', 1.5)):g}"
        )

    @staticmethod
    def _parse_notch_widths(text: str) -> float | list[float]:
        parts = [item.strip() for item in text.split(",") if item.strip()]
        if not parts:
            raise ValueError("notch_widths cannot be empty.")
        if len(parts) == 1:
            return float(parts[0])
        return [float(item) for item in parts]

    @staticmethod
    def _parse_p2p_thresh(text: str) -> list[float]:
        parts = [item.strip() for item in text.split(",") if item.strip()]
        if len(parts) != 2:
            raise ValueError("p2p_thresh must be provided as two numbers: min,max.")
        return [float(parts[0]), float(parts[1])]

    def _collect_params(self) -> dict[str, Any]:
        candidate = {
            "notch_widths": self._parse_notch_widths(self._notch_widths_edit.text()),
            "epoch_dur": float(self._epoch_dur_edit.text().strip()),
            "p2p_thresh": self._parse_p2p_thresh(self._p2p_thresh_edit.text()),
            "autoreject_correct_factor": float(
                self._autoreject_factor_edit.text().strip()
            ),
        }
        valid, normalized, message = normalize_filter_advance_params(candidate)
        if not valid:
            raise ValueError(message)
        return normalized

    def _on_submit(self, action: str) -> None:
        try:
            params = self._collect_params()
        except Exception as exc:  # noqa: BLE001
            self._show_warning("Filter Advance", f"Invalid parameters:\n{exc}")
            return
        if action == "set_default":
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
        self._selected_action = action
        self._selected_params = params
        self.accept()
