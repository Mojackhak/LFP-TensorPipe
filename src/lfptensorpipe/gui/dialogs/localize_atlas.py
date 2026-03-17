"""Atlas and interested-region configuration dialog for Localize."""

from __future__ import annotations

from collections.abc import Callable

from lfptensorpipe.app.config_store import AppConfigStore

from .common import (
    QListWidgetItem,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
    Qt,
)
from lfptensorpipe.gui.dialogs import (
    checked_item_texts as _dialog_checked_item_texts,
    set_all_check_state as _dialog_set_all_check_state,
)

SPACE_LOCALIZE_DEFAULTS_KEY = "space_localize_defaults"


def _normalize_region_selection(
    region_names: tuple[str, ...],
    selected_regions: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    valid = {name for name in region_names}
    seen: set[str] = set()
    out: list[str] = []
    for raw_name in selected_regions or ():
        name = str(raw_name).strip()
        if not name or name not in valid or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return tuple(out)


class LocalizeAtlasDialog(QDialog):
    """Configure the atlas and interested regions for one record."""

    def __init__(
        self,
        *,
        space: str,
        atlas_names: tuple[str, ...],
        current_atlas: str,
        current_selected_regions: tuple[str, ...],
        region_loader: Callable[[str], tuple[str, ...]],
        config_store: AppConfigStore,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Configure Localize Atlas")
        self.resize(460, 520)
        self.setMinimumSize(460, 520)

        self._space = str(space).strip()
        self._atlas_names = tuple(
            str(item).strip() for item in atlas_names if str(item).strip()
        )
        self._region_loader = region_loader
        self._config_store = config_store
        self._selected_payload: dict[str, object] | None = None
        self._status_message = ""

        self._atlas_combo = QComboBox()
        self._region_list = QListWidget()
        self._status_label = QLabel("Status: 0/0 selected")
        self._save_button = QPushButton("Save")

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        root.addWidget(QLabel("Atlas"))
        self._atlas_combo.setToolTip("Atlas used by Localize Apply for this record.")
        for atlas_name in self._atlas_names:
            self._atlas_combo.addItem(atlas_name, atlas_name)
        self._atlas_combo.currentIndexChanged.connect(self._on_atlas_changed)
        root.addWidget(self._atlas_combo)

        root.addWidget(QLabel("Interested Regions"))
        self._region_list.setToolTip(
            "Select the atlas regions that should generate `{region}_in` columns."
        )
        self._region_list.itemChanged.connect(self._on_region_item_changed)
        root.addWidget(self._region_list, stretch=1)

        action_row = QWidget()
        action_layout = QHBoxLayout(action_row)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(6)
        select_all_button = QPushButton("Select All")
        clear_button = QPushButton("Clear")
        select_all_button.clicked.connect(self._on_select_all)
        clear_button.clicked.connect(self._on_clear)
        select_all_button.setToolTip("Select every region in the current atlas.")
        clear_button.setToolTip("Clear all selected regions.")
        action_layout.addWidget(select_all_button)
        action_layout.addWidget(clear_button)
        action_layout.addStretch(1)
        root.addWidget(action_row)

        self._status_label.setToolTip(
            "Show the current interested-region selection count for this atlas."
        )
        root.addWidget(self._status_label)

        footer = QWidget()
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(0, 0, 0, 0)
        footer_layout.setSpacing(6)
        set_default_button = QPushButton("Set as Default")
        restore_default_button = QPushButton("Restore Default")
        cancel_button = QPushButton("Cancel")
        self._save_button.clicked.connect(self._on_save)
        cancel_button.clicked.connect(self.reject)
        set_default_button.clicked.connect(self._on_set_default)
        restore_default_button.clicked.connect(self._on_restore_default)
        set_default_button.setToolTip(
            "Save the current atlas and interested regions as defaults for this space."
        )
        restore_default_button.setToolTip(
            "Restore the saved atlas and interested-region defaults for this space."
        )
        cancel_button.setToolTip("Close without saving atlas changes.")
        self._save_button.setToolTip(
            "Save the current atlas and interested-region selection for this record."
        )
        footer_layout.addWidget(set_default_button)
        footer_layout.addWidget(restore_default_button)
        footer_layout.addStretch(1)
        footer_layout.addWidget(cancel_button)
        footer_layout.addWidget(self._save_button)
        root.addWidget(footer)

        initial_index = self._atlas_combo.findData(current_atlas)
        if initial_index < 0:
            initial_index = 0
        self._atlas_combo.setCurrentIndex(initial_index)
        self._apply_region_selection(
            current_selected_regions,
            message="",
        )

    @property
    def selected_payload(self) -> dict[str, object] | None:
        return self._selected_payload

    def _current_atlas(self) -> str:
        atlas = self._atlas_combo.currentData()
        return str(atlas).strip() if isinstance(atlas, str) else ""

    def _current_region_names(self) -> tuple[str, ...]:
        atlas = self._current_atlas()
        if not atlas:
            return ()
        return tuple(self._region_loader(atlas))

    def _current_selected_regions(self) -> tuple[str, ...]:
        return tuple(_dialog_checked_item_texts(self._region_list))

    def _render_regions(self, selected_regions: tuple[str, ...]) -> None:
        region_names = self._current_region_names()
        selected = set(_normalize_region_selection(region_names, selected_regions))
        self._region_list.blockSignals(True)
        self._region_list.clear()
        for region_name in region_names:
            item = QListWidgetItem(region_name)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if region_name in selected else Qt.Unchecked)
            self._region_list.addItem(item)
        self._region_list.blockSignals(False)
        self._refresh_status()

    def _refresh_status(self) -> None:
        selected = len(self._current_selected_regions())
        total = len(self._current_region_names())
        suffix = f" {self._status_message}" if self._status_message else ""
        self._status_label.setText(f"Status: {selected}/{total} selected{suffix}")
        self._save_button.setEnabled(selected > 0)

    def _apply_region_selection(
        self,
        selected_regions: tuple[str, ...] | list[str] | None,
        *,
        message: str,
    ) -> None:
        region_names = self._current_region_names()
        normalized = _normalize_region_selection(region_names, selected_regions)
        if not normalized and region_names:
            normalized = tuple(region_names)
        self._status_message = message.strip()
        self._render_regions(normalized)

    def _load_defaults(self) -> dict[str, dict[str, object]]:
        payload = self._config_store.read_yaml("localization.yml", default={})
        if not isinstance(payload, dict):
            return {}
        raw_defaults = payload.get(SPACE_LOCALIZE_DEFAULTS_KEY, {})
        if not isinstance(raw_defaults, dict):
            return {}
        out: dict[str, dict[str, object]] = {}
        for raw_space, raw_entry in raw_defaults.items():
            space = str(raw_space).strip()
            if not space or not isinstance(raw_entry, dict):
                continue
            atlas = str(raw_entry.get("atlas", "")).strip()
            selected_regions = raw_entry.get("selected_regions", [])
            out[space] = {
                "atlas": atlas,
                "selected_regions": [
                    str(item).strip() for item in selected_regions if str(item).strip()
                ],
            }
        return out

    def _on_atlas_changed(self, _index: int) -> None:
        self._apply_region_selection((), message="")

    def _on_region_item_changed(self, _item: QListWidgetItem) -> None:
        if self._current_selected_regions():
            self._status_message = ""
        else:
            self._status_message = "At least one region must be selected."
        self._refresh_status()

    def _on_select_all(self) -> None:
        _dialog_set_all_check_state(self._region_list, checked=True)
        self._status_message = ""
        self._refresh_status()

    def _on_clear(self) -> None:
        _dialog_set_all_check_state(self._region_list, checked=False)
        self._status_message = "At least one region must be selected."
        self._refresh_status()

    def _on_set_default(self) -> None:
        atlas = self._current_atlas()
        selected_regions = self._current_selected_regions()
        if not atlas or not selected_regions:
            QMessageBox.warning(
                self,
                "Localize Atlas",
                "Select one atlas and at least one region before saving defaults.",
            )
            return
        payload = self._config_store.read_yaml("localization.yml", default={})
        if not isinstance(payload, dict):
            payload = {}
        defaults = self._load_defaults()
        defaults[self._space] = {
            "atlas": atlas,
            "selected_regions": list(selected_regions),
        }
        payload = dict(payload)
        payload[SPACE_LOCALIZE_DEFAULTS_KEY] = defaults
        self._config_store.write_yaml("localization.yml", payload)
        self._status_message = "Defaults saved."
        self._refresh_status()

    def _on_restore_default(self) -> None:
        defaults = self._load_defaults()
        entry = defaults.get(self._space)
        atlas = ""
        selected_regions: tuple[str, ...] = ()
        message = "No saved default; using first available atlas."
        if isinstance(entry, dict):
            atlas = str(entry.get("atlas", "")).strip()
            selected_regions = tuple(
                str(item).strip()
                for item in entry.get("selected_regions", [])
                if str(item).strip()
            )
            message = "Defaults restored."
        atlas_index = self._atlas_combo.findData(atlas) if atlas else -1
        if atlas_index < 0:
            atlas_index = 0
            if atlas:
                message = (
                    "Saved default atlas unavailable; using first available atlas."
                )
        self._atlas_combo.blockSignals(True)
        self._atlas_combo.setCurrentIndex(atlas_index)
        self._atlas_combo.blockSignals(False)
        self._apply_region_selection(selected_regions, message=message)

    def _on_save(self) -> None:
        atlas = self._current_atlas()
        selected_regions = self._current_selected_regions()
        if not atlas or not selected_regions:
            QMessageBox.warning(
                self,
                "Localize Atlas",
                "Select one atlas and at least one region before saving.",
            )
            return
        self._selected_payload = {
            "atlas": atlas,
            "selected_regions": list(selected_regions),
        }
        self.accept()
