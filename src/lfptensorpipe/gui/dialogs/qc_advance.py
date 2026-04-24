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

    @property
    def selected_action(self) -> str | None:
        return self._selected_action

    @property
    def selected_params(self) -> dict[str, Any] | None:
        return self._selected_params

    def _show_warning(self, title: str, message: str) -> int:
        return QMessageBox.warning(self, title, message)

    def _apply_to_fields(self, params: dict[str, Any]) -> None:
        self._fmin_edit.setText(f"{float(params.get('fmin', 1.0)):g}")
        self._fmax_edit.setText(f"{float(params.get('fmax', 200.0)):g}")
        if self._mode == "psd":
            if self._n_fft_edit is not None:
                self._n_fft_edit.setText(str(int(params.get("n_fft", 1024))))
            if self._average_combo is not None:
                target = bool(params.get("average", True))
                index = self._average_combo.findData(target)
                self._average_combo.setCurrentIndex(index if index >= 0 else 0)
        else:
            if self._n_freqs_edit is not None:
                self._n_freqs_edit.setText(str(int(params.get("n_freqs", 40))))
            if self._decim_edit is not None:
                self._decim_edit.setText(str(int(params.get("decim", 4))))

    def _on_restore_defaults(self) -> None:
        self._apply_to_fields(self._default_params)

    def _collect_params(self) -> dict[str, Any]:
        if self._mode == "psd":
            candidate = {
                "fmin": float(self._fmin_edit.text().strip()),
                "fmax": float(self._fmax_edit.text().strip()),
                "n_fft": int(
                    (self._n_fft_edit.text() if self._n_fft_edit else "").strip()
                ),
                "average": bool(
                    self._average_combo.currentData()
                    if self._average_combo is not None
                    else True
                ),
            }
            valid, normalized, message = normalize_preproc_viz_psd_params(candidate)
        else:
            candidate = {
                "fmin": float(self._fmin_edit.text().strip()),
                "fmax": float(self._fmax_edit.text().strip()),
                "n_freqs": int(
                    (
                        self._n_freqs_edit.text()
                        if self._n_freqs_edit is not None
                        else ""
                    ).strip()
                ),
                "decim": int(
                    (
                        self._decim_edit.text() if self._decim_edit is not None else ""
                    ).strip()
                ),
            }
            valid, normalized, message = normalize_preproc_viz_tfr_params(candidate)
        if not valid:
            raise ValueError(message)
        return normalized

    def _on_submit(self, action: str) -> None:
        try:
            params = self._collect_params()
        except Exception as exc:  # noqa: BLE001
            self._show_warning(self.windowTitle(), f"Invalid parameters:\n{exc}")
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
