"""Edit an Epoch Inspector QC scheme without changing the active trial until Save."""

from copy import deepcopy

from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lfptensorpipe.app.alignment.inspector_config import (
    LETTERS,
    SIGNAL,
    normalize_scheme,
    rule_defaults,
    tokens,
)
from .epoch_review_controls import Channels, Frequency, number


class EpochQCDialog(QDialog):
    def __init__(self, parent, data, config, bands, load_default, save_default):
        super().__init__(parent)
        self.setWindowTitle("Epoch Inspector · QC Advance")
        self.data, self.bands = data, bands
        self.load_default, self.save_default = load_default, save_default
        self.scheme = deepcopy(config)
        self.active = -1
        layout = QGridLayout(self)
        left, right = QGroupBox("Rule configuration"), QGroupBox("Parameters")
        form, self.params = QFormLayout(left), QFormLayout(right)
        self.metric_combo, self.method, self.match, self.time = (
            QComboBox() for _ in range(4)
        )
        self.method.addItems(["MAD", "Valid coverage"])
        self.match.addItems(["Any", "All"])
        self.time.addItems(["All", "Custom"])
        for metric in data.metrics:
            self.metric_combo.addItem(
                "Preprocessed Signal" if metric == SIGNAL else metric, metric
            )
        self.channels, self.frequency = Channels(), Frequency()
        self.tlo, self.thi = number(data.bounds[0]), number(data.bounds[1])
        time_box = QWidget()
        time_row = QHBoxLayout(time_box)
        time_row.setContentsMargins(0, 0, 0, 0)
        for w in (self.time, self.tlo, self.thi):
            time_row.addWidget(w)
        for label, control in (
            ("Metric", self.metric_combo),
            ("Channels", self.channels),
            ("Channel match", self.match),
            (f"Time window ({data.unit})", time_box),
            ("Frequency", self.frequency),
            ("Method", self.method),
        ):
            form.addRow(label, control)
        self.threshold = number(6, 0)
        self.minimum = QSpinBox()
        self.minimum.setRange(1, 2**31 - 1)
        self.baseline = number(0, 0)
        self.duration = number(0, 0)
        self.duration.setSpecialValueText(" ")
        self.area, self.coverage = number(5, 0, 100), number(80, 0, 100)
        for label, w in (
            ("MAD threshold", self.threshold),
            (f"Baseline window ({data.unit})", self.baseline),
            ("Min. valid epochs", self.minimum),
            (f"Min. excursion duration ({data.unit})", self.duration),
            ("Min. cluster area (%)", self.area),
            ("Min. valid coverage (%)", self.coverage),
        ):
            self.params.addRow(label, w)
        layout.addWidget(left, 0, 0)
        layout.addWidget(right, 0, 1)

        configured = QGroupBox("Configured rules")
        table_layout = QVBoxLayout(configured)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["ID", "Metric", "Method", "Channels"])
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setMaximumHeight(140)
        table_layout.addWidget(self.table)
        buttons = QHBoxLayout()
        self.add = QPushButton("Add")
        self.update_rule = QPushButton("Update")
        self.remove = QPushButton("Remove")
        for btn, fn in (
            (self.add, self._add),
            (self.update_rule, self._update),
            (self.remove, self._remove),
        ):
            buttons.addWidget(btn)
            btn.clicked.connect(fn)
        buttons.addStretch()
        table_layout.addLayout(buttons)
        layout.addWidget(configured, 1, 0, 1, 2)

        expression = QGroupBox("Final expression")
        expr_layout = QHBoxLayout(expression)
        self.expression = QLineEdit()
        expr_layout.addWidget(self.expression)
        for name in ("AND", "OR", "(", ")", "⌫"):
            btn = QPushButton(name)
            btn.setMaximumWidth(54)
            btn.clicked.connect(lambda _, token=name: self._insert(token))
            expr_layout.addWidget(btn)
        layout.addWidget(expression, 2, 0, 1, 2)
        self.message = QLabel()
        self.message.setWordWrap(True)
        self.message.setStyleSheet("color: #a23434")
        layout.addWidget(self.message, 3, 0, 1, 2)
        footer = QHBoxLayout()
        for name, fn in (
            ("Set as Default", self._set_default),
            ("Restore Default", self._restore),
            ("Cancel", self.reject),
            ("Save", self._save),
        ):
            if name == "Cancel":
                footer.addStretch()
            btn = QPushButton(name)
            btn.clicked.connect(fn)
            footer.addWidget(btn)
        layout.addLayout(footer, 4, 0, 1, 2)
        self.table.cellClicked.connect(self._clicked)
        self.metric_combo.currentIndexChanged.connect(self._metric_changed)
        self.method.currentTextChanged.connect(self._method_changed)
        self.time.currentTextChanged.connect(self._time_changed)
        self.channels.changed.connect(self._channel_changed)
        self._restore_ui()

    def _time_changed(self):
        custom = self.time.currentText() == "Custom"
        self.tlo.setVisible(custom)
        self.thi.setVisible(custom)

    def _channel_changed(self):
        self.match.setEnabled(len(self.channels.values()) > 1)

    def _method_changed(self):
        metric = self.metric_combo.currentData()
        signal = metric == SIGNAL
        mad = self.method.currentText() == "MAD"
        for field in (
            self.threshold,
            self.minimum,
            self.baseline,
            self.duration,
            self.area,
        ):
            self.params.setRowVisible(field, mad)
        self.params.setRowVisible(self.coverage, not mad)
        self.baseline.setEnabled(signal)
        self.duration.setEnabled(signal)
        self.area.setEnabled(not signal)
        self.threshold.setValue(6 if signal else 4)
        self.minimum.setValue(3)
        self.baseline.setValue(0)
        self.duration.setValue(0)
        self.area.setValue(5)
        self.coverage.setValue(80)

    def _metric_changed(self):
        metric = self.metric_combo.currentData()
        available = metric in self.data.metrics
        self.channels.populate(self.data.channels(metric) if available else [], "All")
        grid = self.data.grid(metric) if available and metric != SIGNAL else None
        self.frequency.populate(grid, self.bands, "All")
        self._channel_changed()
        self._method_changed()

    def _load(self, index):
        self.active = index
        r = self.scheme["rules"][index]
        metric = r["metric"]
        pos = self.metric_combo.findData(metric)
        if pos < 0:
            self.metric_combo.addItem(f"{metric} (unavailable)", metric)
            pos = self.metric_combo.count() - 1
        self.metric_combo.setCurrentIndex(pos)
        self._metric_changed()
        self.method.setCurrentText(r["method"])
        self._method_changed()
        available = metric in self.data.metrics
        names = self.data.channels(metric) if available else []
        # Preserve explicit missing names for the user to correct on import.
        listed = (
            names
            if r["channels"] == "All"
            else list(dict.fromkeys([*names, *r["channels"]]))
        )
        self.channels.populate(listed, r["channels"])
        self.match.setCurrentText(r["match"])
        self._channel_changed()
        self.time.setCurrentText("All" if r["time"] == "All" else "Custom")
        low, high = self.data.bounds if r["time"] == "All" else r["time"]
        self.tlo.setValue(low)
        self.thi.setValue(high)
        self._time_changed()
        grid = self.data.grid(metric) if available and metric != SIGNAL else None
        self.frequency.populate(grid, self.bands, r["frequency"])
        for key, w in (
            ("threshold", self.threshold),
            ("minimum_epochs", self.minimum),
            ("baseline", self.baseline),
            ("duration", self.duration),
            ("area", self.area),
            ("coverage", self.coverage),
        ):
            if key in r:
                w.setValue(0 if r[key] is None else r[key])
        self.initial_editor = self._draft()

    def _draft(self):
        metric, method = self.metric_combo.currentData(), self.method.currentText()
        r = rule_defaults(metric, method)
        names = self.channels.values()
        r.update(
            channels="All" if self.channels.all_mode else names,
            match=self.match.currentText(),
            frequency=self.frequency.value(),
        )
        if self.time.currentText() != "All":
            r["time"] = [self.tlo.value(), self.thi.value()]
        if method == "MAD":
            r.update(
                threshold=self.threshold.value(), minimum_epochs=self.minimum.value()
            )
            if metric == SIGNAL:
                r.update(
                    baseline=self.baseline.value(),
                    duration=self.duration.value() or None,
                )
            else:
                r["area"] = self.area.value()
        else:
            r["coverage"] = self.coverage.value()
        if r["time"] != "All" or r.get("baseline") or r.get("duration"):
            r["axis"] = self.data.axis
        return r

    def _clicked(self, row, col):
        self._load(row)
        if col == 0:
            self._insert(LETTERS[row])

    def _insert(self, token):
        if token == "⌫":
            parts = tokens(self.expression.text())
            self.expression.setText(" ".join(parts[:-1]))
        else:
            self.expression.insert(f" {token} ")

    def _render(self):
        rules = self.scheme["rules"]
        self.table.setRowCount(len(rules))
        for i, r in enumerate(rules):
            for j, value in enumerate(
                (
                    LETTERS[i],
                    "Preprocessed Signal" if r["metric"] == SIGNAL else r["metric"],
                    r["method"],
                    (
                        r["channels"]
                        if r["channels"] == "All"
                        else ", ".join(r["channels"])
                    ),
                )
            ):
                self.table.setItem(i, j, QTableWidgetItem(value))
        self.add.setEnabled(len(rules) < 26)
        self.update_rule.setEnabled(bool(rules))
        self.remove.setEnabled(bool(rules))

    def _restore_ui(self):
        self.expression.setText(self.scheme["expression"])
        self._render()
        self._load(0)
        self.table.selectRow(0)
        self._show_issues()

    def _checked(self):
        if self.active >= 0 and self._draft() != self.initial_editor:
            raise ValueError("Apply the edited rule with Update or Add before saving.")
        config = normalize_scheme({**self.scheme, "expression": self.expression.text()})
        for i, r in enumerate(config["rules"]):
            try:
                self.data.validate_rule(r)
            except ValueError as exc:
                raise ValueError(f"Rule {LETTERS[i]}: {exc}") from exc
        return config

    def _show_issues(self):
        try:
            self._checked()
        except ValueError as exc:
            self.message.setText(str(exc))
        else:
            self.message.clear()

    def _store(self, new):
        try:
            r = normalize_scheme({"rules": [self._draft()], "expression": "A"})[
                "rules"
            ][0]
            self.data.validate_rule(r)
        except ValueError as exc:
            self.message.setText(str(exc))
            return
        if new:
            self.scheme["rules"].append(r)
            self.active = len(self.scheme["rules"]) - 1
        else:
            self.scheme["rules"][self.active] = r
        self._render()
        self._load(self.active)
        self.table.selectRow(self.active)
        self._show_issues()

    def _add(self):
        self._store(True)

    def _update(self):
        self._store(False)

    def _remove(self):
        i = self.active
        expr = tokens(self.expression.text())
        if LETTERS[i] in expr:
            self.message.setText(f"Remove {LETTERS[i]} from the expression first.")
            return
        self.scheme["rules"].pop(i)
        self.expression.setText(
            " ".join(
                (
                    LETTERS[LETTERS.index(t) - 1]
                    if len(t) == 1 and t in LETTERS[i + 1 :]
                    else t
                )
                for t in expr
            )
        )
        self._render()
        self.active = min(i, len(self.scheme["rules"]) - 1)
        if self.active >= 0:
            self._load(self.active)
            self.table.selectRow(self.active)
        self._show_issues()

    def _save(self):
        try:
            self.scheme = self._checked()
        except ValueError as exc:
            self.message.setText(str(exc))
            return
        self.accept()

    def _set_default(self):
        try:
            self.save_default(self._checked())
        except (ValueError, OSError) as exc:
            self.message.setText(str(exc))
            return
        self.message.setText("Default saved.")

    def _restore(self):
        self.scheme = deepcopy(self.load_default())
        self._restore_ui()
