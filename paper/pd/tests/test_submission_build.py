from __future__ import annotations

from pathlib import Path

import pandas as pd

from paper.pd.submission.build_submission import (
    extract_comment_segments,
    normalize_subj_paradigm_frame,
    scalar_artifact_paths_from_pdf,
)


def test_normalize_subj_paradigm_frame_flattens_sheet() -> None:
    raw = pd.DataFrame(
        [
            {
                "Subject": None,
                "ID": None,
                "Gait": "Sit",
                "Unnamed: 3": "Stand",
                "Unnamed: 4": "Walk",
                "Unnamed: 5": "Turn",
                "Unnamed: 6": "Gait cycle",
                "Unnamed: 7": "FoG",
                "Med": "Med",
                "Pain": "Pain",
            },
            {
                "Subject": 1,
                "ID": "Foo",
                "Gait": "✓",
                "Unnamed: 3": "✓",
                "Unnamed: 4": "",
                "Unnamed: 5": "✓",
                "Unnamed: 6": "",
                "Unnamed: 7": "",
                "Med": "✓",
                "Pain": "",
            },
        ]
    )
    out = normalize_subj_paradigm_frame(raw)
    assert list(out.columns) == [
        "SubjectIndex",
        "SubjectCode",
        "SubjectName",
        "Sit",
        "Stand",
        "Walk",
        "Turn",
        "Cycle",
        "FoG",
        "Med",
        "Pain",
    ]
    assert out.loc[0, "SubjectCode"] == "sub-001"
    assert out.loc[0, "Sit"] == "Yes"
    assert out.loc[0, "Walk"] == "No"


def test_extract_comment_segments_splits_inline_comments() -> None:
    text = "Alpha [[COMMENT: fill clinical metadata]] beta"
    segments = extract_comment_segments(text)
    assert segments == [
        ("Alpha ", None),
        ("[?]", "fill clinical metadata"),
        (" beta", None),
    ]


def test_scalar_artifact_paths_from_pdf_maps_lmer_to_tukey_csvs() -> None:
    pdf = Path(
        "/tmp/med/burst/rate-scalar_summary_trans/Beta_low/lmer_model/Phase-Region_raw.pdf"
    )
    lmer_csv, rlmer_csv = scalar_artifact_paths_from_pdf(pdf)  # type: ignore[misc]
    assert lmer_csv.as_posix().endswith(
        "/lmer_model/Phase-Region_tukey.csv"
    )
    assert rlmer_csv.as_posix().endswith(
        "/rlmer_model/Phase-Region_tukey.csv"
    )
