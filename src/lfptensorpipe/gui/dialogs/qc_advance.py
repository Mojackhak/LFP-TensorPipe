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

        self._fmin_edit = QLineEdit()
        self._fmax_edit = QLineEdit()
        self._fmin_edit.setToolTip("Lower bound for the plot frequency range in Hz.")
        self._fmax_edit.setToolTip("Upper bound for the plot frequency range in Hz.")
        form.addRow("Low freq", self._fmin_edit)
        form.addRow("High freq", self._fmax_edit)

        self._n_fft_edit: QLineEdit | None = None
        self._average_combo: QComboBox | None = None
        self._n_freqs_edit: QLineEdit | None = None
        self._decim_edit: QLineEdit | None = None

        if self._mode == "psd":
            self._n_fft_edit = QLineEdit()
            self._average_combo = QComboBox()
            self._average_combo.addItem("True", True)
            self._average_combo.addItem("False", False)
            self._n_fft_edit.setToolTip("FFT length in samples for PSD.")
            self._average_combo.setToolTip("Average PSD across selected channels.")
            form.addRow("n_fft", self._n_fft_edit)
            form.addRow("average", self._average_combo)
        else:
            self._n_freqs_edit = QLineEdit()
            self._decim_edit = QLineEdit()
            self._n_freqs_edit.setToolTip("Number of frequencies for TFR.")
            self._decim_edit.setToolTip("Decimation factor for TFR computation.")
            form.addRow("n_freqs", self._n_freqs_edit)
            form.addRow("decim", self._decim_edit)

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
        edits = [self._fmin_edit, self._fmax_edit]
        if self._mode == "psd":
            edits.append(self._n_fft_edit)
        else:
            edits.extend([self._n_freqs_edit, self._decim_edit])
        for edit in edits:
            if edit is not None:
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

    def _apply_to_fields(self, params: dict[str, Any]) -> None:
        self._fmin_edit.setText(self._draft_text(params.get("fmin", 1.0)))
        self._fmax_edit.setText(self._draft_text(params.get("fmax", 200.0)))
        if self._mode == "psd":
            if self._n_fft_edit is not None:
                self._n_fft_edit.setText(self._draft_text(params.get("n_fft", 1024)))
            if self._average_combo is not None:
                target = bool(params.get("average", True))
                index = self._average_combo.findData(target)
                self._average_combo.setCurrentIndex(index if index >= 0 else 0)
        else:
            if self._n_freqs_edit is not None:
                self._n_freqs_edit.setText(self._draft_text(params.get("n_freqs", 40)))
            if self._decim_edit is not None:
                self._decim_edit.setText(self._draft_text(params.get("decim", 4)))

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
        if self._mode == "psd":
            return {
                "fmin": self._draft_number(self._fmin_edit.text()),
                "fmax": self._draft_number(self._fmax_edit.text()),
                "n_fft": self._draft_number(
                    self._n_fft_edit.text() if self._n_fft_edit is not None else ""
                ),
                "average": bool(
                    self._average_combo.currentData()
                    if self._average_combo is not None
                    else True
                ),
            }
        return {
            "fmin": self._draft_number(self._fmin_edit.text()),
            "fmax": self._draft_number(self._fmax_edit.text()),
            "n_freqs": self._draft_number(
                self._n_freqs_edit.text() if self._n_freqs_edit is not None else ""
            ),
            "decim": self._draft_number(
                self._decim_edit.text() if self._decim_edit is not None else ""
            ),
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
        controls = [self._fmin_edit, self._fmax_edit]
        controls.extend(
            [self._n_fft_edit]
            if self._mode == "psd"
            else [self._n_freqs_edit, self._decim_edit]
        )
        for control in controls:
            set_control_validation_error(control, None)
        candidate = self._collect_draft_params()
        if self._mode == "psd":
            valid, _, message = normalize_preproc_viz_psd_params(candidate)
        else:
            valid, _, message = normalize_preproc_viz_tfr_params(candidate)
        if valid:
            return
        lowered = message.lower()
        if "frequency" in lowered or "fmin" in lowered or "fmax" in lowered:
            targets = [self._fmin_edit, self._fmax_edit]
        elif "n_fft" in lowered:
            targets = [self._n_fft_edit]
        elif "n_freqs" in lowered and "decim" not in lowered:
            targets = [self._n_freqs_edit]
        elif "decim" in lowered and "n_freqs" not in lowered:
            targets = [self._decim_edit]
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
