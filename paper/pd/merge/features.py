"""IPython entrypoint for merging PD feature tables."""

# %%
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_FILE = Path(__file__).resolve()
_REPO_ROOT = _FILE.parents[3]
_SRC_ROOT = _REPO_ROOT / "src"

for _path in (_REPO_ROOT, _SRC_ROOT):
    _path_text = str(_path)
    if _path.exists() and _path_text not in sys.path:
        sys.path.insert(0, _path_text)

from paper.pd.merge.core import (
    collect_feature_inventory,
    export_merge_tables,
    inventory_frame,
)
from paper.pd.specs import (
    DEFAULT_MERGE_SPEC,
    DEFAULT_PROJECT_ROOT,
    DEFAULT_STRICT_SELECTION,
)

# %%
PROJECT_ROOT = DEFAULT_PROJECT_ROOT
MERGE_SPEC = DEFAULT_MERGE_SPEC
STRICT_SELECTION = DEFAULT_STRICT_SELECTION

# %%
inventory = collect_feature_inventory(PROJECT_ROOT)
inventory_df = inventory_frame(inventory)
inventory_df.head()

# %%
report = export_merge_tables(
    PROJECT_ROOT,
    merge_spec=MERGE_SPEC,
    strict_selection=STRICT_SELECTION,
)
pd.DataFrame([report.as_dict()])

# %%
sorted(report.named_outputs)[:10]

# %%
report.missing_selections

# %%
report.load_errors

# %%
