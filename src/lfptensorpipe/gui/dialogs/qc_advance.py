"""QC advance dialog."""

from __future__ import annotations

from .common import *  # noqa: F403


class QcAdvanceDialog(QDialog):
    """Advance dialog for preprocess Visualization QC PSD/TFR params."""

    def __init__(
        self,
        *,
        mode: str,
        session_params: dict[str, Any],
        default_params: dict[str, Any],
        set_default_callback: Callable[[dict[str, Any]], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if mode not in {"psd", "tfr"}:
            raise ValueError(f"Unsupported QC mode: {mode}")
        self._mode = mode
        self.setWindowTitle("PSD Advance" if mode == "psd" else "TFR Advance")
        self.setModal(True)
        self.resize(460, 220)
        self._selected_action: str | None = None
        self._selected_params: dict[str, Any] | None = None
        self._default_params = dict(default_params)
        self._set_default_callback = set_default_callback

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)

        self._form = form
        self._fields = {}
        defaults = (
            default_preproc_viz_psd_params()
            if mode == "psd"
            else default_preproc_viz_tfr_params()
        )
        labels = dict(
            method="Method",
            fmin="Low freq",
            fmax="High freq",
            tmin="Start time (s)",
            tmax="Stop time (s)",
            average="Average channels",
            exclude_bad="Exclude BAD/EDGE",
            n_fft="n_fft",
            n_freqs="n_freqs",
            spacing="Frequency spacing",
            decim="decim",
            cycles="Morlet cycles",
            bandwidth="Bandwidth (Hz)",
            window_length_s="Window length (s)",
        )
        order = [
            "method",
            "fmin",
            "fmax",
            "tmin",
            "tmax",
            "exclude_bad",
            "average",
            "n_fft",
            "spacing",
            "n_freqs",
            "cycles",
            "window_length_s",
            "bandwidth",
            "decim",
        ]
        for key in order:
            if key not in defaults:
                continue
            if key in ("method", "spacing", "average", "exclude_bad"):
                edit = QComboBox()
                choices = (
                    (
                        ["welch", "multitaper", "morlet"]
                        if mode == "psd"
                        else ["morlet", "multitaper"]
                    )
                    if key == "method"
                    else (["linear", "log"] if key == "spacing" else [True, False])
                )
                for value in choices:
                    edit.addItem(str(value), value)
            else:
                edit = QLineEdit()
                if key in ("tmin", "tmax", "cycles"):
                    edit.setPlaceholderText("Auto")
            form.addRow(labels[key], edit)
            self._fields[key] = edit
        for key in ("fmin", "fmax", "n_fft", "n_freqs", "decim"):
            setattr(self, f"_{key}_edit", self._fields.get(key))
        self._average_combo = self._fields["average"]

        root.addLayout(form)

        button_row = QWidget()
        button_layout = QHBoxLayout(button_row)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(6)
        save_button = QPushButton("Save")
        default_button = QPushButton("Set as Default")
        restore_button = QPushButton("Restore Defaults")
        cancel_button = QPushButton("Cancel")
        save_button.setToolTip(
            "Apply these visualization parameters to the current session."
        )
        default_button.setToolTip("Save these visualization parameters as defaults.")
        restore_button.setToolTip("Restore saved visualization defaults.")
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
        for edit in self._fields.values():
            if isinstance(edit, QLineEdit):
                edit.editingFinished.connect(self._refresh_validation)
            else:
                edit.currentIndexChanged.connect(self._refresh_validation)
        self._refresh_validation()

    @property
    def selected_action(self) -> str | None:
        return self._selected_action

    @property
    def selected_params(self) -> dict[str, Any] | None:
        return self._selected_params

    def _show_warning(self, title: str, message: str) -> int:
        return QMessageBox.warning(self, title, message)

    def _apply_to_fields(self, params: dict[str, Any]) -> None:
        defaults = (
            default_preproc_viz_psd_params()
            if self._mode == "psd"
            else default_preproc_viz_tfr_params()
        )
        for key, edit in self._fields.items():
            value = params.get(key, defaults[key])
            edit.blockSignals(True)
            if isinstance(edit, QComboBox):
                index = edit.findData(value)
                if index < 0:
                    edit.addItem(str(value), value)
                    index = edit.count() - 1
                edit.setCurrentIndex(index)
            else:
                edit.setText(self._draft_text(value))
            edit.blockSignals(False)

    def _on_restore_defaults(self) -> None:
        self._apply_to_fields(self._default_params)
        self._refresh_validation()

    @staticmethod
    def _draft_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return f"{value:g}"
        return str(value)

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

    def _collect_draft_params(self) -> dict[str, Any]:
        return {
            key: (
                edit.currentData()
                if isinstance(edit, QComboBox)
                else self._draft_number(edit.text())
            )
            for key, edit in self._fields.items()
        }

    def _collect_params(self) -> dict[str, Any]:
        candidate = self._collect_draft_params()
        if self._mode == "psd":
            valid, normalized, message = normalize_preproc_viz_psd_params(candidate)
        else:
            valid, normalized, message = normalize_preproc_viz_tfr_params(candidate)
        if not valid:
            raise ValueError(message)
        return normalized

    def _refresh_validation(self) -> None:
        method = self._fields["method"].currentData()
        for key, control in self._fields.items():
            active = True
            if key == "n_fft":
                active = method == "welch"
            elif key in ("spacing", "n_freqs"):
                active = method != "welch"
            elif key == "cycles":
                active = method == "morlet"
            elif key in ("bandwidth", "window_length_s"):
                active = method == "multitaper"
            self._form.setRowVisible(control, active)
            set_control_validation_error(control, None)
        candidate = self._collect_draft_params()
        normalize = (
            normalize_preproc_viz_psd_params
            if self._mode == "psd"
            else normalize_preproc_viz_tfr_params
        )
        valid, _, message = normalize(candidate)
        if not valid:
            targets = [
                control for key, control in self._fields.items() if key in message
            ]
            for control in targets or [self._fields["method"]]:
                set_control_validation_error(control, message)

    def _on_submit(self, action: str) -> None:
        if action == "set_default":
            try:
                params = self._collect_params()
            except Exception as exc:  # noqa: BLE001
                self._refresh_validation()
                self._show_warning(self.windowTitle(), f"Invalid parameters:\n{exc}")
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
