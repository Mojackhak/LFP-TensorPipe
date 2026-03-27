"""Build cycle-candidate literature screening outputs without using manuscript files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from paper.pd.paths import resolve_project_root, summary_table_root

DEFAULT_SUBMISSION_ROOTNAME = "submission"
SUPPORT_PRIORITY = {"joint": 0, "rlmer": 1, "lmer": 2}
LANDMARKS = [
    ("contralateral_heel_strike", 0),
    ("ipsilateral_toe_off", 18),
    ("ipsilateral_heel_strike", 50),
    ("contralateral_toe_off", 68),
]
PHASE_BUCKETS = [
    ("0-18%", 0, 18, "contralateral heel-strike centered early stance"),
    ("18-50%", 18, 50, "ipsilateral swing"),
    ("50-68%", 50, 68, "post-ipsilateral heel-strike transition"),
    ("68-100%", 68, 100, "contralateral swing to next contralateral heel strike"),
]


@dataclass(frozen=True)
class LiteratureRef:
    """Curated reference used to annotate cycle candidates."""

    ref_id: str
    citation: str
    url: str
    scope: str


@dataclass(frozen=True)
class EvidenceSpec:
    """Evidence annotation assigned to one cycle candidate."""

    evidence_type: str
    claim_strength: str
    key_refs: tuple[str, ...]
    literature_consensus: str
    notes: str


def _submission_root(project_root: Path) -> Path:
    return project_root / DEFAULT_SUBMISSION_ROOTNAME


def _figure_table_root(project_root: Path) -> Path:
    return _submission_root(project_root) / "figure&table"


def _candidate_root(project_root: Path) -> Path:
    return summary_table_root(project_root) / "cycle" / "interval"


def candidate_paths(project_root: Path) -> dict[str, Path]:
    """Return the three current cycle candidate tables."""
    root = _candidate_root(project_root)
    return {
        "joint": root / "cycle_timepoint_joint_interval_candidates.csv",
        "lmer": root / "cycle_timepoint_lmer_interval_candidates.csv",
        "rlmer": root / "cycle_timepoint_rlmer_interval_candidates.csv",
    }


def literature_catalog() -> dict[str, LiteratureRef]:
    """Return the manually curated literature catalogue."""
    return {
        "morelli2023": LiteratureRef(
            "morelli2023",
            "Morelli A, et al. Gait-related brain activity in Parkinson's disease: a systematic review. npj Parkinsons Dis. 2023.",
            "https://pubmed.ncbi.nlm.nih.gov/36244929/",
            "Systematic review of invasive and non-invasive gait electrophysiology in PD.",
        ),
        "arnulfo2018": LiteratureRef(
            "arnulfo2018",
            "Arnulfo G, et al. Phase matters: A role for the subthalamic network during gait. PLoS One. 2018.",
            "https://pubmed.ncbi.nlm.nih.gov/29874298/",
            "Human PD STN recordings during gait with beta-band phase-locking changes.",
        ),
        "louie2022": LiteratureRef(
            "louie2022",
            "Louie DR, et al. Cortico-subthalamic field potentials support classification of the natural gait cycle in Parkinson's disease and reveal individualized spectral signatures. eNeuro. 2022.",
            "https://pubmed.ncbi.nlm.nih.gov/36270803/",
            "Human PD natural gait-cycle STN and cortico-subthalamic oscillatory signatures.",
        ),
        "yeh2024": LiteratureRef(
            "yeh2024",
            "Yeh CH, et al. Auditory cues modulate the short timescale dynamics of STN activity during stepping in Parkinson's disease. Brain Stimul. 2024.",
            "https://pubmed.ncbi.nlm.nih.gov/38636820/",
            "Human PD stepping study showing alpha and beta transient modulation in STN.",
        ),
        "chen2019": LiteratureRef(
            "chen2019",
            "Chen CC, et al. Subthalamic oscillations correlate with vulnerability to freezing of gait in patients with Parkinson's disease. Neurobiol Dis. 2019.",
            "https://pubmed.ncbi.nlm.nih.gov/31494286/",
            "Human PD recordings linking low-frequency ventral STN/SNr activity to gait vulnerability.",
        ),
        "gulberti2023": LiteratureRef(
            "gulberti2023",
            "Gulberti A, et al. Subthalamic and nigral neurons are differentially modulated during parkinsonian gait. Brain. 2023.",
            "https://pubmed.ncbi.nlm.nih.gov/36730026/",
            "Human PD microelectrode stepping study demonstrating gait-related nigral and STN modulation.",
        ),
        "thevathasan2012": LiteratureRef(
            "thevathasan2012",
            "Thevathasan W, et al. Alpha oscillations in the pedunculopontine nucleus correlate with gait performance in parkinsonism. Brain. 2012.",
            "https://pubmed.ncbi.nlm.nih.gov/22232591/",
            "Human locomotor alpha activity in a gait-relevant brainstem node.",
        ),
        "he2021": LiteratureRef(
            "he2021",
            "He S, et al. Gait-Phase Modulates Alpha and Beta Oscillations in the Pedunculopontine Nucleus. J Neurosci. 2021.",
            "https://pubmed.ncbi.nlm.nih.gov/34413208/",
            "Human gait-phase modulation of alpha/beta power and coherence in a locomotor control node.",
        ),
        "wagner2022": LiteratureRef(
            "wagner2022",
            "Wagner JR, et al. Combined Subthalamic and Nigral Stimulation Modulates Temporal Gait Coordination and Cortical Gait-Network Activity in Parkinson's Disease. Front Hum Neurosci. 2022.",
            "https://pubmed.ncbi.nlm.nih.gov/35295883/",
            "Human PD evidence that nigral stimulation alters temporal gait coordination.",
        ),
        "li2016": LiteratureRef(
            "li2016",
            "Li M, et al. The network of causal interactions for beta oscillations in the pedunculopontine nucleus, primary motor cortex, and subthalamic nucleus of walking parkinsonian rats. Exp Neurol. 2016.",
            "https://pubmed.ncbi.nlm.nih.gov/27163550/",
            "Animal locomotor study with directional beta-band causal flow from STN-centered networks.",
        ),
        "lamos2026": LiteratureRef(
            "lamos2026",
            "Lamos M, et al. Local Field Aperiodic Spectral Power Modulated by Deep Brain Stimulation in Parkinson's Disease. Mov Disord. 2026.",
            "https://pubmed.ncbi.nlm.nih.gov/41684330/",
            "Human STN aperiodic slope and offset during rest and a short gait task.",
        ),
        "guevara2025": LiteratureRef(
            "guevara2025",
            "Beyond beta rhythms: subthalamic aperiodic broadband power scales with Parkinson's disease severity-a cross-sectional multicentre study. EBioMedicine. 2025.",
            "https://pubmed.ncbi.nlm.nih.gov/41168073/",
            "Large multicentre STN aperiodic broadband study in PD.",
        ),
    }


def load_candidate_table(csv_path: Path, support_level: str) -> pd.DataFrame:
    """Load one candidate table and append support metadata."""
    frame = pd.read_csv(csv_path)
    frame = frame.loc[frame["Metric"].notna()].copy()
    frame["support_level"] = support_level
    frame["source_candidates_csv"] = str(csv_path)
    return frame


def interval_points(start_pct: int, end_pct: int, wraps_cycle: bool) -> list[int]:
    """Expand one interval into cycle bins using the exported half-open convention."""
    if wraps_cycle or end_pct < start_pct:
        return list(range(start_pct, 100)) + list(range(0, end_pct))
    return list(range(start_pct, end_pct))


def candidate_id_from_row(row: pd.Series) -> str:
    """Return a stable identifier for one candidate row."""
    return (
        f"{row['support_level']}:{row['Metric']}:{row['Band']}:{row['Region']}:"
        f"{row['direction']}:{int(row['start_pct'])}-{int(row['end_pct'])}"
    )


def infer_candidate_cluster(metric: str, region: str) -> str:
    """Map one candidate to a literature-search cluster."""
    metric_norm = str(metric).lower()
    region_norm = str(region)
    if region_norm == "SNr" and metric_norm == "periodic":
        return "SNr local periodic"
    if region_norm == "SNr" and metric_norm == "aperiodic":
        return "SNr local aperiodic"
    if region_norm == "SNr-STN" and metric_norm in {"coherence", "wpli", "ciplv"}:
        return "SNr-STN synchrony"
    if region_norm == "SNr->STN" and metric_norm in {"psi", "trgc"}:
        return "SNr->STN directional"
    if region_norm == "STN" and metric_norm == "periodic":
        return "STN local periodic extension"
    if region_norm == "STN" and metric_norm == "aperiodic":
        return "STN local aperiodic extension"
    return "Other cycle extension"


def _closest_landmark(points: list[int]) -> str:
    if not points:
        return "unknown"
    midpoint = points[len(points) // 2]
    best_name = "contralateral_heel_strike"
    best_distance = 10**9
    for name, position in LANDMARKS:
        distance = min((midpoint - position) % 100, (position - midpoint) % 100)
        if distance < best_distance:
            best_name = name
            best_distance = distance
    return best_name


def _landmarks_crossed(points: list[int]) -> list[str]:
    point_set = set(points)
    crossed: list[str] = []
    for name, position in LANDMARKS[1:]:
        if position in point_set:
            crossed.append(name)
    return crossed


def map_phase_bucket(start_pct: int, end_pct: int, wraps_cycle: bool) -> tuple[str, str, str, str]:
    """Map one interval to a unique gait-phase bucket and textual context."""
    if wraps_cycle or end_pct < start_pct:
        return (
            "cycle boundary / contralateral heel-strike centered",
            "cycle boundary / contralateral heel-strike centered",
            "contralateral_heel_strike",
            "",
        )
    points = interval_points(start_pct, end_pct, wraps_cycle)
    overlaps: list[tuple[int, str, str]] = []
    for bucket_id, lower, upper, label in PHASE_BUCKETS:
        bucket_points = set(range(lower, upper))
        overlap = len(bucket_points.intersection(points))
        overlaps.append((overlap, bucket_id, label))
    overlaps.sort(key=lambda item: (-item[0], item[1]))
    _, bucket_id, label = overlaps[0]
    closest = _closest_landmark(points)
    crossed = ", ".join(_landmarks_crossed(points))
    if crossed:
        context = f"{label}; crosses {crossed.replace('_', ' ')}"
    else:
        context = label
    return bucket_id, context, closest, crossed


def annotate_candidate(row: pd.Series) -> EvidenceSpec:
    """Assign literature evidence to one candidate row."""
    metric = str(row["Metric"]).lower()
    band = str(row["Band"])
    region = str(row["Region"])
    cluster = infer_candidate_cluster(metric, region)

    if cluster == "STN local periodic extension":
        if band in {"Theta", "Alpha", "Beta_low", "Beta_high"}:
            return EvidenceSpec(
                "direct",
                "strong support",
                ("louie2022", "arnulfo2018", "yeh2024"),
                "Human PD invasive gait and stepping studies consistently report phase-locked STN low-frequency and beta-band modulation during locomotion.",
                "Direct support exists for cycle-resolved STN modulation, although the exact phase windows and spectral peaks vary across tasks and cohorts.",
            )
        return EvidenceSpec(
            "partial",
            "discussion-only",
            ("louie2022", "arnulfo2018", "morelli2023"),
            "Human STN gait studies report broad 10-50 Hz and beta changes, but isolated gamma cycle intervals are not a consistent primary finding.",
            "Use STN gamma intervals as an extension rather than as a replication claim.",
        )

    if cluster == "STN local aperiodic extension":
        return EvidenceSpec(
            "analog",
            "potentially novel",
            ("lamos2026", "guevara2025", "morelli2023"),
            "Aperiodic STN slope and offset have been linked to PD state and short gait tasks, but phase-resolved gait-cycle intervals have not been directly reported.",
            "Treat STN aperiodic cycle intervals as an extension of recent aperiodic biomarker work rather than as a replicated gait-cycle phenomenon.",
        )

    if cluster == "SNr local periodic":
        if band in {"Theta", "Alpha"}:
            return EvidenceSpec(
                "partial",
                "discussion-only",
                ("chen2019", "gulberti2023", "wagner2022"),
                "Human studies implicate the nigral region in gait vulnerability and stepping-related modulation, especially in lower frequencies, but chronic SNr LFP cycle intervals remain sparse.",
                "This supports nigral locomotor relevance but not a direct replication of chronic SNr cycle-locked periodic intervals.",
            )
        return EvidenceSpec(
            "analog",
            "potentially novel",
            ("gulberti2023", "wagner2022", "morelli2023"),
            "Nigral gait-related modulation has been reported, but isolated chronic SNr gamma cycle intervals were not identified in the human PD literature reviewed here.",
            "Frame SNr gamma intervals as a potentially novel extension.",
        )

    if cluster == "SNr local aperiodic":
        return EvidenceSpec(
            "analog",
            "potentially novel",
            ("lamos2026", "guevara2025", "gulberti2023"),
            "Current aperiodic LFP reports are centered on STN and non-phase-resolved tasks; direct SNr aperiodic gait-cycle reports were not identified.",
            "SNr aperiodic intervals should be treated as potentially novel with only nearby STN aperiodic background support.",
        )

    if cluster == "SNr-STN synchrony":
        if band in {"Alpha", "Theta"}:
            return EvidenceSpec(
                "partial",
                "discussion-only",
                ("louie2022", "he2021", "thevathasan2012"),
                "Phase-resolved locomotor coherence has been demonstrated in STN-cortical and PPN-cortical networks at low frequencies, but direct SNr-STN synchrony during gait was not identified.",
                "Low-frequency SNr-STN synchrony intervals extend a broader locomotor-network literature rather than directly matching an existing SNr-STN report.",
            )
        return EvidenceSpec(
            "partial",
            "discussion-only",
            ("arnulfo2018", "louie2022", "yeh2024"),
            "Beta-band locomotor network modulation is well documented in STN-centered human recordings, but direct SNr-STN beta synchrony reports during gait are still lacking.",
            "Use beta synchrony candidates as a network extension with partial support only.",
        )

    if cluster == "SNr->STN directional":
        return EvidenceSpec(
            "analog",
            "potentially novel",
            ("li2016", "wagner2022", "morelli2023"),
            "Directed locomotor beta flow has been demonstrated in animal STN-centered walking networks, and nigral stimulation alters temporal gait coordination in humans, but direct human SNr->STN cycle directionality reports were not identified.",
            "Directional cycle candidates should be framed as potentially novel, mechanistically adjacent observations.",
        )

    return EvidenceSpec(
        "none",
        "potentially novel",
        ("morelli2023",),
        "No sufficiently close reports were identified for this candidate beyond the general PD gait-electrophysiology literature.",
        "Keep this candidate as potentially novel unless later targeted searches reveal a closer match.",
    )


def build_candidate_frame(project_root: Path) -> pd.DataFrame:
    """Load, normalize, and annotate the union of current cycle candidate tables."""
    frames = [
        load_candidate_table(path, support_level)
        for support_level, path in candidate_paths(project_root).items()
    ]
    frame = pd.concat(frames, ignore_index=True)
    frame["candidate_id"] = frame.apply(candidate_id_from_row, axis=1)
    frame["candidate_cluster"] = frame.apply(
        lambda row: infer_candidate_cluster(row["Metric"], row["Region"]),
        axis=1,
    )
    mapped = frame.apply(
        lambda row: map_phase_bucket(
            int(row["start_pct"]),
            int(row["end_pct"]),
            str(row["wraps_cycle"]).upper() == "TRUE",
        ),
        axis=1,
        result_type="expand",
    )
    mapped.columns = [
        "phase_bucket",
        "gait_event_context",
        "closest_landmark",
        "landmarks_crossed",
    ]
    frame = pd.concat([frame, mapped], axis=1)
    evidence = frame.apply(annotate_candidate, axis=1)
    frame["search_status"] = "reviewed"
    frame["evidence_type"] = evidence.map(lambda item: item.evidence_type)
    frame["claim_strength"] = evidence.map(lambda item: item.claim_strength)
    frame["literature_consensus"] = evidence.map(lambda item: item.literature_consensus)
    frame["notes"] = evidence.map(lambda item: item.notes)
    frame["key_ref_ids"] = evidence.map(lambda item: "; ".join(item.key_refs))

    catalog = literature_catalog()
    max_refs = 3
    for index in range(max_refs):
        slot = index + 1
        frame[f"key_paper_{slot}_citation"] = ""
        frame[f"key_paper_{slot}_url"] = ""
        frame[f"key_paper_{slot}_scope"] = ""
    for row_index, evidence_spec in enumerate(evidence):
        for index, ref_id in enumerate(evidence_spec.key_refs[:max_refs]):
            ref = catalog[ref_id]
            slot = index + 1
            frame.at[row_index, f"key_paper_{slot}_citation"] = ref.citation
            frame.at[row_index, f"key_paper_{slot}_url"] = ref.url
            frame.at[row_index, f"key_paper_{slot}_scope"] = ref.scope

    frame["support_priority"] = frame["support_level"].map(SUPPORT_PRIORITY)
    frame = frame.sort_values(
        by=["support_priority", "candidate_rank", "Metric", "Band", "Region"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)
    return frame


def build_phase_map(screening: pd.DataFrame) -> pd.DataFrame:
    """Build the compact candidate-to-phase workbook."""
    columns = [
        "candidate_id",
        "support_level",
        "candidate_rank",
        "candidate_cluster",
        "Metric",
        "Band",
        "Region",
        "direction",
        "start_pct",
        "end_pct",
        "span_pct",
        "wraps_cycle",
        "phase_bucket",
        "gait_event_context",
        "closest_landmark",
        "landmarks_crossed",
    ]
    return screening.loc[:, columns].copy()


def _candidate_labels(frame: pd.DataFrame) -> str:
    labels = (
        frame["support_level"]
        + " "
        + frame["Metric"].astype(str)
        + " "
        + frame["Band"].astype(str)
        + " "
        + frame["Region"].astype(str)
        + " "
        + frame["start_pct"].astype(int).astype(str)
        + "-"
        + frame["end_pct"].astype(int).astype(str)
        + "%"
    )
    return "; ".join(labels.tolist())


def build_summary_markdown(screening: pd.DataFrame) -> str:
    """Build the three-paragraph cycle literature summary."""
    direct = screening.loc[screening["evidence_type"] == "direct"].copy()
    partial = screening.loc[screening["evidence_type"] == "partial"].copy()
    novel = screening.loc[screening["claim_strength"] == "potentially novel"].copy()

    direct_text = (
        "Direct support in the current literature is concentrated in STN-centered human "
        "gait recordings rather than in SNr or SNr-STN recordings. The strongest direct "
        "matches in the present screening are "
        f"{_candidate_labels(direct) if not direct.empty else 'none'}, "
        "which align with prior reports of gait-cycle or stepping-related STN low-frequency "
        "and beta modulation and with gait-phase-sensitive STN/cortical coherence "
        f"([{literature_catalog()['louie2022'].url}]({literature_catalog()['louie2022'].url}), "
        f"[{literature_catalog()['arnulfo2018'].url}]({literature_catalog()['arnulfo2018'].url}), "
        f"[{literature_catalog()['yeh2024'].url}]({literature_catalog()['yeh2024'].url}))."
    )

    partial_text = (
        "Partial support or broader mechanistic extension is available for the SNr and "
        "SNr-STN candidate clusters. The most relevant partially supported candidates are "
        f"{_candidate_labels(partial.head(8)) if not partial.empty else 'none'}, "
        "because prior work shows that nigral activity participates in locomotor control, "
        "that ventral STN/SNr low-frequency activity tracks gait vulnerability, and that "
        "phase-resolved low-frequency coherence exists in related locomotor networks even "
        "though direct chronic SNr-STN gait-cycle reports are scarce "
        f"([{literature_catalog()['chen2019'].url}]({literature_catalog()['chen2019'].url}), "
        f"[{literature_catalog()['gulberti2023'].url}]({literature_catalog()['gulberti2023'].url}), "
        f"[{literature_catalog()['he2021'].url}]({literature_catalog()['he2021'].url}), "
        f"[{literature_catalog()['thevathasan2012'].url}]({literature_catalog()['thevathasan2012'].url}))."
    )

    novel_text = (
        "Potentially novel findings in the present cycle outputs are concentrated in SNr "
        "aperiodic intervals, SNr gamma periodic intervals, and SNr->STN directional "
        "candidates such as "
        f"{_candidate_labels(novel.head(10)) if not novel.empty else 'none'}. "
        "For these candidates, the reviewed literature provides at most nearby STN "
        "aperiodic or animal STN-centered directional analogs rather than a direct human "
        "PD SNr/SNr-STN gait-cycle match "
        f"([{literature_catalog()['lamos2026'].url}]({literature_catalog()['lamos2026'].url}), "
        f"[{literature_catalog()['guevara2025'].url}]({literature_catalog()['guevara2025'].url}), "
        f"[{literature_catalog()['li2016'].url}]({literature_catalog()['li2016'].url}), "
        f"[{literature_catalog()['wagner2022'].url}]({literature_catalog()['wagner2022'].url}))."
    )

    return "\n\n".join(
        [
            "# Cycle Literature Summary",
            "Search scope: current cycle candidate tables only. Local manuscript files were not used.",
            direct_text,
            partial_text,
            novel_text,
        ]
    )


def write_workbook(frame: pd.DataFrame, path: Path, sheet_name: str) -> None:
    """Write one dataframe to a single-sheet Excel workbook."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name=sheet_name, index=False)


def build_cycle_literature(project_root: Path) -> dict[str, Path]:
    """Generate the cycle literature screening deliverables."""
    output_root = _figure_table_root(project_root)
    output_root.mkdir(parents=True, exist_ok=True)
    screening = build_candidate_frame(project_root)
    phase_map = build_phase_map(screening)
    summary_text = build_summary_markdown(screening)

    screening_path = output_root / "cycle_literature_screening.xlsx"
    phase_map_path = output_root / "cycle_candidate_phase_map.xlsx"
    summary_path = output_root / "cycle_literature_summary.md"

    write_workbook(screening, screening_path, "cycle_literature")
    write_workbook(phase_map, phase_map_path, "cycle_phase_map")
    summary_path.write_text(summary_text + "\n", encoding="utf-8")

    return {
        "screening": screening_path,
        "phase_map": phase_map_path,
        "summary": summary_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="PD project root. Defaults to paper/pd DEFAULT_PROJECT_ROOT.",
    )
    args = parser.parse_args()
    project_root = resolve_project_root(args.project_root)
    outputs = build_cycle_literature(project_root)
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
