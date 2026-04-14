# PD Script Guide

This document summarizes the maintained analysis scripts under `paper/pd`.
It covers the merge, preprocessing, anatomy, example, statistics, and
visualization stages. The local-only `paper/pd/submission` workspace is
intentionally excluded.

## Merge

### `paper/pd/merge/core.py`

- Responsibility: backend for collecting feature inventory from
  `derivatives/lfptensorpipe/sub-*/.../features/**/*.pkl`, normalizing merge
  specs, simplifying table schemas, filtering region and phase combinations,
  and exporting named paper tables under `{project}/summary/table/{name}/...`.
- Inputs: project root, merge spec and region or phase rules from
  `paper/pd/specs.py`, and source feature pickles.
- Outputs: merged `.pkl` tables, sibling `.xlsx` files for scalar tables, and
  a `MergeReport` for interactive review.
- Workflow position: first paper-level consolidation step before preprocessing.

### `paper/pd/merge/features.py`

- Responsibility: IPython entrypoint for previewing the discovered feature
  inventory and running `export_merge_tables`.
- Inputs: `DEFAULT_PROJECT_ROOT`, `DEFAULT_MERGE_SPEC`, and
  `DEFAULT_STRICT_SELECTION`.
- Outputs: inventory preview tables in-session plus the merge exports written
  by `paper/pd/merge/core.py`.
- Workflow position: interactive launcher for the merge stage.

## Preprocessing

### `paper/pd/preproc/aggregate.py`

- Responsibility: nested-aware aggregation helper used to summarize scalar,
  `Series`-valued, and `DataFrame`-valued `Value` cells without flattening the
  original nested structure.
- Inputs: grouped paper-level tables, aggregation mode, grouping columns, and
  alignment rules.
- Outputs: aggregated DataFrames consumed by the preprocessing backend.
- Workflow position: utility layer used by the summary stage.

### `paper/pd/preproc/core.py`

- Responsibility: backend for the four-stage preprocessing pipeline: summary,
  transform, normalize, and cycle shift.
- Inputs: merged tables under `{project}/summary/table/{name}/...`, transform
  mode config, normalization specs, and optional table-name filters.
- Outputs: derived `_summary`, `_trans`, `_normalized`, and `_shift` tables,
  plus a `PreprocReport` describing written outputs and skipped items.
- Workflow position: main preprocessing engine between merge and downstream
  statistics or visualization.

### `paper/pd/preproc/features.py`

- Responsibility: IPython entrypoint that lists eligible preprocessing sources
  and runs `export_preprocessed_tables`.
- Inputs: `DEFAULT_PROJECT_ROOT`, transform defaults, and normalization
  defaults from `paper/pd/specs.py`.
- Outputs: source inventory previews in-session plus the full preprocessing
  export set written by `paper/pd/preproc/core.py`.
- Workflow position: interactive launcher for preprocessing.

### `paper/pd/preproc/cycle_trace_export.py`

- Responsibility: flattens shifted cycle trace tables, fills NaN gaps, applies
  circular LOWESS smoothing, downsamples traces to 100 cycle bins, and writes
  shared CSV inputs for interval-based analyses.
- Inputs: shifted cycle trace pickles named
  `mean-trace_summary_trans_normalized_shift.pkl`.
- Outputs:
  `{project}/summary/table/cycle/preprocessed/cycle_trace_long.csv` and
  `{project}/summary/table/cycle/preprocessed/cycle_trace_parameters.csv`.
- Workflow position: bridge from cycle trace preprocessing into the interval
  statistics pipeline.

## Anatomy

### `paper/pd/anat/anat.py`

- Responsibility: reads representative channel-coordinate pickles, validates
  required columns, recomputes `Region`, and exports a cohort anatomy summary.
- Inputs: `sub-*/sit/localize/channel_representative_coords.pkl`.
- Outputs: `{project}/summary/cohort/channel_coords.csv` and the in-memory
  channel table returned by `build_channel_coords_table`.
- Workflow position: anatomy summary export used alongside merged or
  preprocessed analysis tables.

## Examples

### `paper/pd/examples/med/workflow.py`

- Responsibility: prepares the representative Sub-014 medication burst example
  by selecting one subject, applying a `log10` transform, cropping the
  60 to 120 second window, ordering channels and phases, and optionally
  rendering channel-wise PDFs.
- Inputs: the merged med burst raw table
  `{project}/summary/table/med/burst/na-raw.pkl` or a previously exported
  example pickle.
- Outputs:
  `{project}/summary/eg/med/sub-014_burst_na-raw_log10_60_120.pkl` and one PDF
  per channel in the same output directory.
- Workflow position: reusable example-generation helper for demos and figures.

### `paper/pd/examples/med/export_sub014_burst.py`

- Responsibility: CLI wrapper around `export_subject_burst_example`.
- Inputs: optional `--project-root` and `--subject`.
- Outputs: prints the exported example pickle path after writing it to disk.
- Workflow position: command-line entrypoint for the example export step.

### `paper/pd/examples/med/run_sub014_burst_viz.py`

- Responsibility: CLI wrapper around `run_subject_burst_example_viz`.
- Inputs: optional `--project-root`, `--subject`, and `--source-path`.
- Outputs: prints one generated PDF path per channel.
- Workflow position: command-line entrypoint for rendering the representative
  example figures.

## Statistics

### `paper/pd/stats/emm.R`

- Responsibility: shared helper library for estimated marginal means,
  stratified Tukey contrasts, slice maps, confidence-interval normalization,
  and optional across-strata p-value adjustment.
- Inputs: fitted models and model metadata passed in by `run_scalar.R`.
- Outputs: tidy EMM tables, pairwise-comparison tables, and joint-test tables
  returned to the caller.
- Workflow position: reusable helper layer sourced by scalar mixed-effects
  analyses.

### `paper/pd/stats/run_scalar.R`

- Responsibility: fits `lmer` and `rlmer` models for scalar and spectral
  summaries across the cycle, med, motor, and turn sections, then exports
  diagnostics and EMM or Tukey outputs.
- Inputs: preprocessed `*scalar_summary_trans.xlsx` workbooks under
  `{project}/summary/table/{section}/{metric}/...`.
- Outputs: per-band analysis folders containing `input_band.xlsx`,
  `model_omnibus.csv`, diagnostic PDFs, and `*_emm.csv` or `*_tukey.csv`
  tables.
- Workflow position: main mixed-effects statistics stage for scalar summaries.

### `paper/pd/stats/run_cycle_interval_fit.R`

- Responsibility: reads the shared cycle trace CSV input, fits timepoint-wise
  `lmer` and `rlmer` models for local and connectivity metrics, and exports
  deviation estimates with significance metadata.
- Inputs:
  `{project}/summary/table/cycle/preprocessed/cycle_trace_long.csv`.
- Outputs:
  `{project}/summary/table/cycle/interval/cycle_timepoint_deviation.csv`.
- Workflow position: first interval-analysis step after cycle trace export.

### `paper/pd/stats/run_cycle_interval_postprocess.R`

- Responsibility: converts timepoint-wise interval statistics into contiguous
  candidate windows and final interval summaries for `lmer`, `rlmer`, and
  joint-significance criteria.
- Inputs: `cycle_timepoint_deviation.csv`.
- Outputs:
  `cycle_timepoint_lmer_intervals.csv`,
  `cycle_timepoint_lmer_interval_candidates.csv`,
  `cycle_timepoint_rlmer_intervals.csv`,
  `cycle_timepoint_rlmer_interval_candidates.csv`,
  `cycle_timepoint_joint_intervals.csv`, and
  `cycle_timepoint_joint_interval_candidates.csv` under
  `{project}/summary/table/cycle/interval/`.
- Workflow position: second interval-analysis step that turns pointwise
  significance into reportable cycle windows.

## Visualization

### `paper/pd/viz/defaults.py`

- Responsibility: central plot styling and wrapper layer for scalar, spectral,
  trace, and raw figures built on `lfptensorpipe.viz.visualdf`.
- Inputs: prepared analysis tables, plot role variables, parameter labels, and
  section-specific config.
- Outputs: matplotlib figures saved to disk, typically as PDF files.
- Workflow position: reusable plotting backend used by scripted figures and
  example rendering.

### `paper/pd/viz/run_viz.py`

- Responsibility: defines visualization specs for cycle, med, motor, and turn
  analyses, discovers matching workbooks, and runs plotting tasks sequentially
  or in parallel.
- Inputs: preprocessed `.xlsx` and `.pkl` tables plus selected spec names or
  section names.
- Outputs: exported scalar, spectral, trace, and raw figure sets generated by
  the plotting wrappers in `paper/pd/viz/defaults.py`.
- Workflow position: main batch figure-generation entrypoint after
  preprocessing and statistical export.
