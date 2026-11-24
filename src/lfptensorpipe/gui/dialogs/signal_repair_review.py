"""Reversible Signal Repair decisions attached to the existing Qt browser."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDockWidget,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
    QPushButton,
    QLabel,
)

from lfptensorpipe.preproc.repair_review import SignalRepairReview


def attach_signal_repair_review(browser, raw, intervals):
    """Attach one review dock; use window-local Qt browser integration only."""
    review = SignalRepairReview(raw, intervals)
    browser._signal_repair_review = review
    dock = QDockWidget("Repair review", browser)
    dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
    table = QTableWidget(len(review.rows), 6)
    table.setHorizontalHeaderLabels(
        ["Accept", "Channel", "Type", "Start (s)", "End (s)", "Samples"]
    )
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.setSortingEnabled(False)
    sfreq = raw.info["sfreq"]
    for row, item in enumerate(review.rows):
        check = QTableWidgetItem()
        check.setFlags(check.flags() | Qt.ItemIsUserCheckable)
        check.setCheckState(Qt.Checked if item["accepted"] else Qt.Unchecked)
        table.setItem(row, 0, check)
        values = [
            item["channel"],
            "gap" if item["kind"] == "gaps" else "peak",
            f'{item["start_sample"] / sfreq:.6f}',
            f'{item["stop_sample"] / sfreq:.6f}',
            str(item["stop_sample"] - item["start_sample"]),
        ]
        for column, value in enumerate(values, 1):
            table.setItem(row, column, QTableWidgetItem(value))
    table.resizeColumnsToContents()
    table.setMinimumWidth(sum(table.columnWidth(i) for i in range(6)) + 35)
    if not review.rows:
        message = QLabel(
            "No reversible repairs. Apply generates a new repair list; skipped candidates are excluded."
        )
        message.setWordWrap(True)
        dock.setWidget(message)
    else:
        dock.setWidget(table)
    browser.addDockWidget(Qt.RightDockWidgetArea, dock)

    def protect_description_actions(*_):
        annotation_dock = browser.mne.fig_annotation
        managed_labels = {
            ann[2] for item in review.rows for ann in review.annotations_for(item)
        }
        locked = annotation_dock.description_cmbx.currentText() in managed_labels
        browser.mne.current_description = (
            None if locked else annotation_dock.description_cmbx.currentText()
        )
        annotation_dock.description_cmbx.setToolTip(
            "Managed repair labels are controlled by the Repair review list."
            if locked
            else "Select a label for manual annotations."
        )
        for button in annotation_dock.findChildren(QPushButton):
            if button.text() in ("Remove Description", "Edit Description"):
                button.setEnabled(not locked)

    def lock_managed_regions():
        managed = {
            review.annotation_key(ann)
            for item in review.rows
            for ann in review.annotations_for(item)
        }
        for region in browser.mne.regions:
            start, stop = region.getRegion()
            annotation = [
                start + raw.first_time,
                stop - start,
                region.description,
                tuple(region.single_channel_annots),
            ]
            if review.annotation_key(annotation) in managed:
                movable = region.setMovable
                movable(False)
                region.setMovable = lambda _value, set_movable=movable: set_movable(
                    False
                )
                region.mouseClickEvent = lambda event: event.ignore()
                region.remove = lambda: None
                region.select = lambda _selected: None

    def refresh_regions():
        browser._setup_annotation_colors()
        for region in list(browser.mne.regions):
            region.removeSingleChannelAnnots.emit(region)
            browser._remove_region(region, from_annot=False)
        browser._update_annotation_segments()
        browser._init_annot_mode()
        lock_managed_regions()
        browser._update_regions_visible()
        browser.mne.overview_bar.update_annotations()
        protect_description_actions()

    def toggle(item):
        if item.column() != 0:
            return
        review.set_accepted(item.row(), item.checkState() == Qt.Checked)
        refresh_regions()
        browser._redraw(update_data=True)

    def locate(row, _column):
        item = review.rows[row]
        center = (item["start_sample"] + item["stop_sample"]) / (2 * sfreq)
        duration = browser.mne.duration
        start = max(0, min(center - duration / 2, raw.times[-1] - duration))
        browser.mne.viewbox.setXRange(start, start + duration, padding=0)

    table.itemChanged.connect(toggle)
    table.cellClicked.connect(locate)
    annotation_dock = browser.mne.fig_annotation
    reset_annotations = annotation_dock.reset

    def reset_and_protect():
        reset_annotations()
        protect_description_actions()

    annotation_dock.reset = reset_and_protect
    drag = browser.mne.viewbox.mouseDragEvent

    def protected_drag(event, axis=None):
        protect_description_actions()
        if browser.mne.annotation_mode and browser.mne.current_description is None:
            event.ignore()
            return
        return drag(event, axis=axis)

    browser.mne.viewbox.mouseDragEvent = protected_drag
    browser.mne.fig_annotation.description_cmbx.currentIndexChanged.connect(
        protect_description_actions
    )
    lock_managed_regions()
    protect_description_actions()
    return review
