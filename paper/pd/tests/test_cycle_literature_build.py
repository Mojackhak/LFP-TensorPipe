from __future__ import annotations

import pandas as pd

from paper.pd.submission.build_cycle_literature import (
    annotate_candidate,
    infer_candidate_cluster,
    map_phase_bucket,
)


def test_map_phase_bucket_marks_wrap_intervals_as_cycle_boundary() -> None:
    bucket, context, closest, crossed = map_phase_bucket(97, 4, True)
    assert bucket == "cycle boundary / contralateral heel-strike centered"
    assert "contralateral heel-strike centered" in context
    assert closest == "contralateral_heel_strike"
    assert crossed == ""


def test_map_phase_bucket_uses_majority_overlap_for_non_wrap_intervals() -> None:
    bucket, context, closest, crossed = map_phase_bucket(44, 52, False)
    assert bucket == "18-50%"
    assert "ipsilateral swing" in context
    assert closest == "ipsilateral_heel_strike"
    assert crossed == "ipsilateral_heel_strike"


def test_infer_candidate_cluster_covers_primary_and_extension_groups() -> None:
    assert infer_candidate_cluster("periodic", "SNr") == "SNr local periodic"
    assert infer_candidate_cluster("wpli", "SNr-STN") == "SNr-STN synchrony"
    assert infer_candidate_cluster("psi", "SNr->STN") == "SNr->STN directional"
    assert infer_candidate_cluster("periodic", "STN") == "STN local periodic extension"


def test_annotate_candidate_marks_snr_aperiodic_as_potentially_novel() -> None:
    row = pd.Series({"Metric": "aperiodic", "Band": "Offset", "Region": "SNr"})
    evidence = annotate_candidate(row)
    assert evidence.evidence_type == "analog"
    assert evidence.claim_strength == "potentially novel"
    assert "direct SNr aperiodic gait-cycle reports were not identified" in evidence.literature_consensus


def test_annotate_candidate_marks_stn_beta_as_direct_support() -> None:
    row = pd.Series({"Metric": "periodic", "Band": "Beta_low", "Region": "STN"})
    evidence = annotate_candidate(row)
    assert evidence.evidence_type == "direct"
    assert evidence.claim_strength == "strong support"
