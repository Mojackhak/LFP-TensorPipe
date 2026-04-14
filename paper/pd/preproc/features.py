"""IPython entrypoint for preprocessing PD paper tables."""

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

from paper.pd.preproc.core import (
    collect_preproc_sources,
    export_preprocessed_tables,
    preproc_source_frame,
)
from paper.pd.specs import (
    DEFAULT_NORMALIZE_SPEC,
    DEFAULT_PROJECT_ROOT,
    DEFAULT_SCALAR_NORMALIZE_SPEC,
    DEFAULT_TRANSFORM_MODE_CFG,
)

# %%
PROJECT_ROOT = DEFAULT_PROJECT_ROOT
TRANSFORM_MODE_CFG = DEFAULT_TRANSFORM_MODE_CFG
NORMALIZE_SPEC = DEFAULT_NORMALIZE_SPEC
SCALAR_NORMALIZE_SPEC = DEFAULT_SCALAR_NORMALIZE_SPEC
TABLE_NAMES = None

# %%
sources = collect_preproc_sources(PROJECT_ROOT, table_names=TABLE_NAMES)
source_df = preproc_source_frame(PROJECT_ROOT, sources)
source_df.head()

# %%
report = export_preprocessed_tables(
    PROJECT_ROOT,
    transform_mode_cfg=TRANSFORM_MODE_CFG,
    normalize_spec=NORMALIZE_SPEC,
    scalar_normalize_spec=SCALAR_NORMALIZE_SPEC,
    table_names=TABLE_NAMES,
)
pd.DataFrame([report.as_dict()])

# %%
sorted(path.as_posix() for path in report.summarized_outputs)[:10]

# %%
sorted(path.as_posix() for path in report.transformed_outputs)[:10]

# %%
sorted(path.as_posix() for path in report.normalized_outputs)[:10]

# %%
sorted(path.as_posix() for path in report.passthrough_outputs)[:10]

# %%
report.skipped_normalization

# %%
report.load_errors

# %%
