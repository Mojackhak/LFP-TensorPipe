"""Localize match dialog."""

from __future__ import annotations

from lfptensorpipe.app.config_store import AppConfigStore

from .common import (
    Any,
    QAbstractItemView,
    QComboBox,
    QDialog,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QVBoxLayout,
    QWidget,
    Qt,
    _dialog_auto_channel_pair,
    _dialog_auto_contact_index_side,
)
from .localize_match_actions import (
    _on_auto_match as _on_auto_match_impl,
    _on_bind_update as _on_bind_update_impl,
    _on_contact_clicked as _on_contact_clicked_impl,
    _on_delete_mapping as _on_delete_mapping_impl,
    _on_reset_all as _on_reset_all_impl,
    _on_save as _on_save_impl,
    _on_set_default as _on_set_default_impl,
    _on_restore_default as _on_restore_default_impl,
)
from .localize_match_render import (
    _build_contact_maps as _build_contact_maps_impl,
    _clear_draft_endpoint as _clear_draft_endpoint_impl,
    _load_current_payload as _load_current_payload_impl,
    _on_channel_selected as _on_channel_selected_impl,
    _on_mapping_row_clicked as _on_mapping_row_clicked_impl,
    _on_rep_coord_changed as _on_rep_coord_changed_impl,
    _refresh_status as _refresh_status_impl,
    _render_channel_list as _render_channel_list_impl,
    _render_lead_columns as _render_lead_columns_impl,
    _render_mapping_table as _render_mapping_table_impl,
    _sync_draft_widgets as _sync_draft_widgets_impl,
)


class LocalizeMatchDialog(QDialog):
    """Record-channel to Lead-DBS contact matcher."""

    def __init__(
        self,
        *,
        channel_names: tuple[str, ...],
        lead_specs: list[dict[str, Any]],
        current_payload: dict[str, Any] | None = None,
        config_store: AppConfigStore | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Match: Record Channels ↔ Lead-DBS Contacts")
        self.resize(800, 600)
        self.setMinimumSize(800, 600)
        self._all_channels = tuple(str(item) for item in channel_names)
        self._lead_specs = list(lead_specs)
        self._mapping: dict[str, dict[str, str]] = {}
        self._selected_payload: dict[str, Any] | None = None
        self._active_channel: str | None = None
        self._draft_anode: str = ""
        self._draft_cathode: str = ""
        self._draft_rep_coord: str = "Mid"
        self._config_store = config_store

        self._contact_by_token: dict[str, dict[str, Any]] = {}
        self._lead_auto_tokens: list[dict[str, str]] = []
        self._lead_auto_sides: list[set[str]] = []
        self._build_contact_maps()

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        top = QWidget()
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(6)
        top_layout.addWidget(QLabel("Search channel"))
        self._search_edit = QLineEdit()
        self._search_edit.textChanged.connect(self._render_channel_list)
        self._search_edit.setToolTip("Filter unmapped channels by name.")
        top_layout.addWidget(self._search_edit, stretch=1)
        self._auto_match_button = QPushButton("Auto Match")
        self._auto_match_button.clicked.connect(self._on_auto_match)
        self._auto_match_button.setToolTip(
            "Auto-map channels when a unique contact pair is detected."
        )
        top_layout.addWidget(self._auto_match_button)
        self._reset_button = QPushButton("Reset")
        self._reset_button.clicked.connect(self._on_reset_all)
        self._reset_button.setToolTip("Clear all mappings for this record.")
        top_layout.addWidget(self._reset_button)
        root.addWidget(top)

        status_row = QWidget()
        status_layout = QHBoxLayout(status_row)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(6)
        self._status_label = QLabel("Status: 0/0 mapped")
        self._status_label.setToolTip(
            "Mapped channels / total channels. Save requires full mapping."
        )
        status_layout.addWidget(self._status_label)
        status_layout.addStretch(1)
        root.addWidget(status_row)

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(6)

        left_panel = QGroupBox("Record Channels")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.setSpacing(4)
        self._channel_list = QListWidget()
        self._channel_list.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._channel_list.itemSelectionChanged.connect(self._on_channel_selected)
        self._channel_list.setToolTip("Unmapped record channels.")
        left_layout.addWidget(self._channel_list, stretch=1)
        body_layout.addWidget(left_panel, stretch=2)

        center_panel = QGroupBox("Leads")
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(6, 6, 6, 6)
        center_layout.setSpacing(4)
        self._lead_scroll = QScrollArea()
        self._lead_scroll.setWidgetResizable(True)
        self._lead_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self._lead_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._lead_scroll_container = QWidget()
        self._lead_scroll_layout = QHBoxLayout(self._lead_scroll_container)
        self._lead_scroll_layout.setContentsMargins(0, 0, 0, 0)
        self._lead_scroll_layout.setSpacing(8)
        self._lead_scroll.setWidget(self._lead_scroll_container)
        center_layout.addWidget(self._lead_scroll, stretch=1)

        special_row = QWidget()
        special_layout = QHBoxLayout(special_row)
        special_layout.setContentsMargins(0, 0, 0, 0)
        special_layout.setSpacing(6)
        special_layout.addWidget(QLabel("Cathode-only"))
        case_button = QPushButton("Case")
        case_button.clicked.connect(lambda: self._on_contact_clicked("case", True))
        case_button.setToolTip("Cathode-only option. Select anode first.")
        ground_button = QPushButton("Ground")
        ground_button.clicked.connect(lambda: self._on_contact_clicked("ground", True))
        ground_button.setToolTip("Cathode-only option. Select anode first.")
        special_layout.addWidget(case_button)
        special_layout.addWidget(ground_button)
        special_layout.addStretch(1)
        center_layout.addWidget(special_row)
        body_layout.addWidget(center_panel, stretch=5)

        right_panel = QGroupBox("Binding Editor")
        right_layout = QGridLayout(right_panel)
        right_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.setHorizontalSpacing(6)
        right_layout.setVerticalSpacing(6)
        right_layout.addWidget(QLabel("Selected"), 0, 0)
        self._selected_channel_label = QLabel("-")
        self._selected_channel_label.setToolTip("Current record channel being edited.")
        right_layout.addWidget(self._selected_channel_label, 0, 1)
        right_layout.addWidget(QLabel("Anode"), 1, 0)
        self._anode_button = QPushButton("")
        self._anode_button.clicked.connect(lambda: self._clear_draft_endpoint("anode"))
        self._anode_button.setToolTip("Click to clear current anode selection.")
        right_layout.addWidget(self._anode_button, 1, 1)
        right_layout.addWidget(QLabel("Cathode"), 2, 0)
        self._cathode_button = QPushButton("")
        self._cathode_button.clicked.connect(
            lambda: self._clear_draft_endpoint("cathode")
        )
        self._cathode_button.setToolTip("Click to clear current cathode selection.")
        right_layout.addWidget(self._cathode_button, 2, 1)
        right_layout.addWidget(QLabel("Rep. coord"), 3, 0)
        self._rep_combo = QComboBox()
        self._rep_combo.addItems(["Anode", "Cathode", "Mid"])
        self._rep_combo.currentIndexChanged.connect(self._on_rep_coord_changed)
        self._rep_combo.setToolTip(
            "Representative coordinate exported for this channel."
        )
        right_layout.addWidget(self._rep_combo, 3, 1)
        self._bind_button = QPushButton("Bind/Update")
        self._bind_button.clicked.connect(self._on_bind_update)
        self._bind_button.setToolTip("Save mapping for selected channel.")
        right_layout.addWidget(self._bind_button, 4, 1, alignment=Qt.AlignRight)
        body_layout.addWidget(right_panel, stretch=3)

        root.addWidget(body, stretch=3)

        table_panel = QGroupBox("Mapping Table")
        table_layout = QVBoxLayout(table_panel)
        table_layout.setContentsMargins(6, 6, 6, 6)
        table_layout.setSpacing(4)
        self._mapping_table = QTableWidget(0, 5)
        self._mapping_table.setHorizontalHeaderLabels(
            ["channel", "Anode", "Cathode", "Rep. coord", "Action"]
        )
        self._mapping_table.verticalHeader().setVisible(False)
        self._mapping_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._mapping_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._mapping_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._mapping_table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._mapping_table.cellClicked.connect(self._on_mapping_row_clicked)
        self._mapping_table.setToolTip("Mapped channels for this record.")
        table_header = self._mapping_table.horizontalHeader()
        table_header.setSectionResizeMode(0, QHeaderView.Stretch)
        table_header.setSectionResizeMode(1, QHeaderView.Stretch)
        table_header.setSectionResizeMode(2, QHeaderView.Stretch)
        table_header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        table_header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        table_layout.addWidget(self._mapping_table, stretch=1)
        root.addWidget(table_panel, stretch=2)

        footer = QWidget()
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(0, 0, 0, 0)
        footer_layout.setSpacing(6)
        self._set_default_button = QPushButton("Set as Default")
        self._set_default_button.clicked.connect(self._on_set_default)
        self._set_default_button.setToolTip(
            "Save current committed mappings to app defaults."
        )
        self._restore_default_button = QPushButton("Restore Default")
        self._restore_default_button.clicked.connect(self._on_restore_default)
        self._restore_default_button.setToolTip(
            "Restore compatible saved mappings from app defaults."
        )
        footer_layout.addWidget(self._set_default_button)
        footer_layout.addWidget(self._restore_default_button)
        footer_layout.addStretch(1)
        cancel_button = QPushButton("Cancel")
        save_button = QPushButton("Save")
        cancel_button.clicked.connect(self.reject)
        save_button.clicked.connect(self._on_save)
        cancel_button.setToolTip("Close without saving mappings.")
        save_button.setToolTip("Save is allowed only when all channels are mapped.")
        footer_layout.addWidget(cancel_button)
        footer_layout.addWidget(save_button)
        root.addWidget(footer)

        self._render_lead_columns()
        self._load_current_payload(current_payload)
        self._render_mapping_table()
        self._render_channel_list()
        self._sync_draft_widgets()

    @property
    def selected_payload(self) -> dict[str, Any] | None:
        return self._selected_payload

    def _build_contact_maps(self) -> None:
        _build_contact_maps_impl(self)

    @staticmethod
    def _auto_contact_index_side(contact_name: str) -> tuple[str | None, str | None]:
        return _dialog_auto_contact_index_side(contact_name)

    @staticmethod
    def _auto_channel_pair(
        channel_name: str,
    ) -> tuple[str, str, str | None] | None:
        return _dialog_auto_channel_pair(channel_name)

    def _load_current_payload(self, payload: dict[str, Any] | None) -> None:
        _load_current_payload_impl(self, payload)

    def _render_lead_columns(self) -> None:
        _render_lead_columns_impl(self)

    def _render_channel_list(self) -> None:
        _render_channel_list_impl(self)

    def _render_mapping_table(self) -> None:
        _render_mapping_table_impl(self)

    def _refresh_status(self) -> None:
        _refresh_status_impl(self)

    def _on_channel_selected(self) -> None:
        _on_channel_selected_impl(self)

    def _on_mapping_row_clicked(self, row: int, _col: int) -> None:
        _on_mapping_row_clicked_impl(self, row, _col)

    def _on_contact_clicked(self, token: str, cathode_only: bool) -> None:
        _on_contact_clicked_impl(self, token, cathode_only)

    def _clear_draft_endpoint(self, endpoint: str) -> None:
        _clear_draft_endpoint_impl(self, endpoint)

    def _sync_draft_widgets(self) -> None:
        _sync_draft_widgets_impl(self)

    def _on_rep_coord_changed(self, _index: int) -> None:
        _on_rep_coord_changed_impl(self, _index)

    def _on_bind_update(self) -> None:
        _on_bind_update_impl(self)

    def _on_delete_mapping(self, channel: str) -> None:
        _on_delete_mapping_impl(self, channel)

    def _on_auto_match(self) -> None:
        _on_auto_match_impl(self)

    def _on_reset_all(self) -> None:
        _on_reset_all_impl(self)

    def _on_save(self) -> None:
        _on_save_impl(self)

    def _on_set_default(self) -> None:
        _on_set_default_impl(self)

    def _on_restore_default(self) -> None:
        _on_restore_default_impl(self)
