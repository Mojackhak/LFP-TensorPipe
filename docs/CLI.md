# Command-line Workflow

Use `lfptp run` or `lfptensorpipe run` to execute a page configuration or a named preprocessing comparison plan
without opening the GUI. These commands are available in a source installation
in the `lfptp` Conda environment. Running either command without arguments opens
the desktop application.

## Inputs

```bash
conda run -n lfptp lfptp run \
  --subject-path /data/project/derivatives/lfptensorpipe/sub-001 \
  --record walk \
  --trial gait \
  --config /data/configs/alignment.json
```

| Argument | Meaning |
|---|---|
| `--subject-path` | Required. An existing `<project>/derivatives/lfptensorpipe/<subject>` directory. The project and subject are resolved from this layout. |
| `--record` | Required. The name of an imported record. |
| `--config` | Required. Accepts a page JSON, a preprocessing recipe, or a named variant plan. |
| `--trial` | Required for Align, Features and variant plans. Use the exact trial slug, such as `cycle-l`. |
| `--output-root` | Required for a named variant plan. Select the directory for all of its independent run results. |

Shell arguments resolve relative to the working directory. Paths inside JSON files resolve relative to the file that declares them. `lfptp run --help` displays help without opening the GUI or initializing
MATLAB or scientific computation.

## Supported Actions

| Exported schema | Action | Required input |
|---|---|---|
| `lfptensorpipe.preprocess-config` | Execute the four optional preprocessing steps and Finish | Imported or accepted Raw |
| `lfptensorpipe.localize-config` | Apply an atlas or User Labels | Imported channels and a label CSV for User Labels, or the atlas runtime and matching reconstruction |
| `lfptensorpipe.tensor-config` | Build selected Tensor metrics | Current Preprocess Finish |
| `lfptensorpipe.alignment-config` | Align Run, then Finish with all generated epochs or an explicit source-matched selection | Current Tensor metrics |
| `lfptensorpipe.features-config` | Extract Features for the selected trial | Current Align Finish |

Localize reads channel names from the canonical Rawdata FIF as soon as a record
has been imported, including channels marked bad. Existing records with only a
Preprocess Raw FIF use that channel inventory. The other pages require their
upstream processing results.

Each invocation processes one record and, when required, one trial. The source record is imported through the GUI. Preprocess recipes and confirmed epoch selections can be replayed from the command line. A command requires its upstream results to be ready.

For cue-response alignment, the record labels are the cue and response in each
event pair. The generated window anchors describe the target timeline. Explicit
percent groups resolve each full label to one record annotation.

For Tensor, `selected_metrics` selects computation. Align processes all current
Tensor metrics. Features processes all metrics in the accepted Align Finish;
the configuration must supply the required axes for each metric. UI fields such
as `active_metric` select the displayed editor, not the computation scope.

## Configuration and Validation

Export a configuration from the relevant GUI page and pass that JSON to
`--config`. The command validates the fields consumed by the requested action,
including selected channels, pairs, annotations, leads, axes, and numeric
parameters. Unsupported Tensor parameter names, missing required fields, and
invalid or incompatible values are errors. Optional values use their defined
service defaults. SpecParam accepts its exported unlimited peak-count setting.

Atlas runtime paths are read from application configuration. Features uses the explicit execution settings in the page export when supplied. Older page files use local Features defaults. Reproduction therefore requires the same relevant source data,
page parameters, and application settings.

Features configurations can set `time_unit` to `percent` or `s` alongside each metric's `times` list. Seconds use the accepted Clip, Stitch or cue-response epoch start as zero and may extend to the full aligned duration. A standard GUI configuration without a unit uses percentages.

## Trials and Epochs

Align updates the named trial or creates that exact canonical slug after
validating the inputs. An occupied directory that is not a valid trial is an
error. Features requires an existing trial. The GUI, CLI, and Features service
use the same eligibility checks for a saved Align Run. A trial missing a
semantic setting required by its method and metrics needs a new Align Run and
Finish before extraction.

A successful Align Run selects every epoch produced by the method and passes
that selection to Finish. Annotation, duration, and BAD/EDGE rules determine
which epochs are generated. To inspect and choose individual epochs, use the
GUI's Epoch Inspector. Both Run and Finish must succeed for the command to
complete successfully.

Align Finish uses the same record-level localization readiness as the GUI.
When Ready, it joins the last applied source (Lead-DBS atlas or User Labels)
for every requested metric and records the source and result ID in the trial log.
When Not ready, every output omits localization. Both paths retain the selected
signals, including channels with unknown region membership. The atlas Localize JSON action applies the atlas configuration. It validates the external configuration's
lead signature before using or generating atlas results. A signature verified
against the current reconstruction and electrode model can be reused. When
verification is required, missing model contact names or a resolver failure
stops the operation. A change to a mapped representative position affects the
corresponding channel coordinates and their dependent tables.
Reactivating a verified v0.3.0 result from the GUI needs no MATLAB session.

## Outputs and Reruns

Results are saved below the selected record using the same artifact layout and
processing services as the GUI. Only the relevant page or trial configuration
is updated in record state. Reopen or reselect the record in the GUI to load the
command's configuration and result state.

Use one writer for a record at a time. Close active GUI editing workflows before
running a CLI write against that record; open GUI drafts do not synchronize with
another process.

Every explicit `run` invocation executes the requested action. Whitespace and
key order in JSON do not change effective parameters or expand the affected
processing scope.

| Accepted result change | Dependent results affected |
|---|---|
| Tensor metrics | Trials using those metrics and their Features |
| Align Run | That trial's Finish and Features |
| Align Finish | That trial's Features |
| Localize Apply | That record's Align Finish and Features |
| Features | That trial's feature outputs |

Run and Finish use shared trial generations, so their dependencies are scoped
to the trial. Localization is shared by the record's trial finishes. Changes
do not invalidate unrelated projects or records.

## Replaying an accepted analysis

`lfptp export` reads accepted results. It accepts the same `--subject-path` and `--record` as `run`, plus `--page preprocess|epoch-selection|alignment|features` and `--output FILE`. Supply `--trial` for the last three pages. Preprocess starts from the accepted Raw, including its annotations and bad channels. A fresh import can bootstrap Raw for a standalone preprocessing run.

A preprocessing JSON has `"schema": "lfptensorpipe.preprocess-config"` and a `preprocess` object. Give each of `signal_repair`, `filter`, `ecg_artifact_removal` and `annotations` an `action` of `apply` or `skip`. Raw and Finish are implicit. Applied repair takes `settings` and optional source-matched interval decisions. Filter takes `low_freq`, `high_freq`, `notches`, `advance` and a review policy. Use `"review": "detected"` to finalize the newly detected annotations and bad channels, or use exported review decisions. ECG takes `method`, `picks`, `method_kwargs` and `mark_filter_edges`. Annotations takes `rows` or `csv` and `mark_filter_edges`. Relative file references inside configurations are resolved against their containing JSON file. Older accepted reviews that only retain their final annotations and bad channels export a recording-bound snapshot. The snapshot is replayable against the same accepted Raw and preceding preprocessing choices. A changed prefix requires new review decisions, or explicit acceptance of the new detection.

Signal Repair, ECG and Annotations publish the reviewed FIF, settings and run log together. If a saved edit cannot match the candidate, the command reports the mismatch and preserves that step's previous accepted output and dependent results. Replaying equivalent annotation rows from JSON or CSV reuses the accepted Annotations and Finish. Numeric strings and numbers represent the same onset and duration.

Filter replay compares the active notch method and its effective settings. Model Notch exports omit the conventional notch width. A variant that disables Model uses the conventional default width of 2 Hz unless it supplies a width. This also applies to older exports with a null width. Applied ECG and Annotations steps require a JSON boolean for `mark_filter_edges`. Boolean values in active numeric Filter fields are configuration errors and are reported before the recipe starts.

For a recording-bound review, replay checks the saved Raw and preprocessing prefix, then compares the accepted annotations and bad channels. An equivalent historical review reuses the current step. In variant plans, a partial override of Model Notch parameters changes the specified fields for that method and preserves the other settings for that method.

For example, this recipe applies SVD ECG correction to two named channels, retains the behavioral events on Raw, and runs Finish.

```json
{
  "schema": "lfptensorpipe.preprocess-config",
  "preprocess": {
    "signal_repair": {"action": "skip"},
    "filter": {"action": "skip"},
    "ecg_artifact_removal": {"action": "apply", "method": "svd", "picks": ["C1", "C2"]},
    "annotations": {"action": "skip"}
  }
}
```

To export a confirmed selection, run `lfptp export --subject-path /study/derivatives/lfptensorpipe/sub-001 --record gait --trial cycle-l --page epoch-selection --output decisions/include.json`. An alignment page export also embeds the confirmed selection. A plan's explicit `base.epoch_selection` takes precedence over that embedded selection. Each replay writes the decision matches, retained indices, missing and additional candidates to the trial's `epoch_replay.json`.

For Localize, supply `"source": "user_labels"` and `"labels_csv": "labels.csv"` inside the usual Localize page. The CSV is a complete label table with `channel`, optional `assignment_method` and `<region>_in` columns. All-zero rows are assessed outside those regions. All-blank or omitted rows are unassessed. This action requires at least one region and assessed channel.

An Align configuration accepts `epoch_selection` as an exported decision file, an embedded selection or the word `accepted` for the latest accepted Finish of this record and trial. The reference is read before rerunning. Matching uses labels, original event coordinates and source intervals. Ambiguous matches prevent Finish. Exact replay also requires matching candidate sets. Sensitivity variants can report missing and extra candidates and finish the uniquely matched included epochs. Features exports include `execution` with actual output types, reducers, reduction settings and numerical domains when available.

For non-Burst metrics, `occupation` is a proportion from 0 to 1. Burst `occupancy` is a percentage from 0 to 100. Each exported output domain follows the stored PKL. Raw, mean and median use their metric's transform policy. Count, occupation, rate, duration and Burst occupancy use the identity transform. Replay and comparison interpret older exports using these same reducer definitions.

Filter replay applies edits to a private detection preview. A successful review
publishes the accepted data and QC together. A rejected edit preserves the prior
Filter, routing choice and pending GUI preview. Review replay recognizes the
annotation onset and end as serialized in FIF across successive edits. Each
step regenerates its own filter-edge markers after applying the human changes.
New plot edits record the precise times supplied to the FIF writer and compare
them with the reopened file. This also captures timing changes from saving an
annotation again during a later review. For
older six-decimal edits, a removal must identify a single annotation within the
precision of the stored onset and end, with matching description, channels and
extras. An incomplete historical review continues to export its recording-bound
final state after another GUI edit.

## Running named variants

The following plan uses the ordinary page JSON files. The baseline preprocessing recipe applies ECG correction.

```json
{
  "base": {
    "preprocess": "configs/preprocess.json",
    "tensor": "configs/tensor.json",
    "alignment": "configs/align.json",
    "features": "configs/features.json",
    "epoch_selection": "decisions/include.json"
  },
  "variants": [
    {"name": "baseline", "repeats": 2},
    {"name": "skip_ecg", "overrides": {"preprocess": {"ecg_artifact_removal": {"action": "skip"}}}}
  ]
}
```

Run it with `lfptp run --subject-path /study/derivatives/lfptensorpipe/sub-001 --record gait --trial cycle-l --config plan.json --output-root /results/gait_ecg`. The output root holds a fixed input snapshot and separate `baseline/run-01`, `baseline/run-02` and `skip_ecg/run-01` projects. Further invocations allocate new run directories. Each repetition starts from the frozen Raw. Each run records its effective configurations, input and software identity, stage outcomes and accepted output receipts in `run.json`. Ready anatomical labels can be shared as a frozen input. The original project is read only during a plan.

The first listed variant defines the baseline. Independent repetitions compute the numerical stages from Raw. For a changed variant the runner can use accepted preprocessing steps through the first changed choice. A Tensor metric can also be reused when its numerical settings and the Finish inputs it consumes match. Burst additionally checks its baseline annotations. Align uses the trial's input generations and source-event selection. Features retains the applicable prior results when only selected phase outputs change. Reused stages identify the originating run. The Features log records which numerical outputs were computed, retained or relabeled. A failed variant keeps its stage and error in `run.json` and the remaining variants continue. All variant projects contain their own derivative files.

An explicit `alignment.epoch_match` applies to both fresh and reused alignments. Use `exact` to require all reference candidates, or `common` to permit missing or additional candidates. Without an explicit policy, the first run of a preprocessing sensitivity variant uses common matching and independent repetitions use exact matching against the preceding accepted selection. The software manifest records the installed `lfp-tensorpipe` distribution version and, for a source checkout, its Git commit.

A variant materializes only its requested Feature outputs. A retained output must match its input generations, axes and reduction settings. Narrowing the Tensor selection also narrows the inherited Feature definitions to the remaining metrics. Burst reuse checks the variant's baseline annotations before copying the tensor or thresholds. If a variant requires epoch decisions from a failed baseline, its manifest records the missing dependency and independent variants can still run.
The first run of a variant with explicit overrides can reuse the baseline when those overrides have the same effective settings. Additional repetitions still compute independently from Raw.

## Comparing published Features

Both runs must have current extracted Features. If either run uses an obsolete
mean or Burst calculation, comparison stops before writing a report and
identifies the run that needs a new Extract Features result.

Repeat mode requires different result generation IDs for every numerical stage the two runs have in common. This includes Finish, the selected Tensor metrics, Align Run, Align Finish and Features. A missing stage receipt cannot establish independence. The frozen Raw and skipped preprocessing steps are excluded from this comparison.

Run `lfptp compare-features --reference /results/gait_ecg/baseline/run-01 --candidate /results/gait_ecg/baseline/run-02 --mode repeat --atol 1e-9 --rtol 1e-7 --output /results/repeat`. For a preprocessing comparison use `--mode sensitivity` and the variant run as the candidate. `paired.csv` reports values and candidate minus reference in the recorded domain. `unmatched.csv` lists unmatched or ambiguous keys. `coverage.csv` reports per-event coverage and common valid support. `summary.json` records the configurations, retention, missingness and numerical agreement. Repeat mode requires equivalent inputs and settings, matching decisions and missingness, and values within tolerance. Sensitivity mode reports measured differences. Choose a study-specific threshold for any scientific claim.

Values are joined by subject, record, trial, original event, channel or ordered pair, metric, output type, reducer, band and phase. Spectra and traces also use frequency and time coordinates. Different numerical definitions are listed in `unmatched.csv`. The signed difference is the candidate value minus the reference value. Repeat mode checks `abs(difference) <= atol + rtol * abs(reference)` with both tolerances defaulting to zero. It requires at least one finite pair. Tolerance failures return exit code 1 and preserve the report.

Coverage counts finite aligned time–frequency cells in each configured band and phase, plus the times with at least one finite frequency. Duration sums the left-edge support of those selected time points within the phase, in seconds or normalized epoch percent. Common duration requires matching time grids and finite data at a common frequency. Different grids retain common-coordinate counts. Native Burst and other native-source reducers may have different integration support from the aligned display. Choose a fresh report directory or export filename for each comparison.

Phase comparisons use the accepted epoch duration to convert between percent and seconds. Pairing and coverage use the extraction tolerance for floating-point window endpoints, independently of the value tolerances `--atol` and `--rtol`. For a legacy phase table without `time_unit`, coverage follows the accepted reduction settings, including absolute seconds on physical-time epochs. Burst phase windows without a unit use percent.

The Tensor setting `mask_edge_effects` is part of each output definition. Changing it sends the affected values to `unmatched.csv` as `different_definition`. The coverage report still describes the observed support in both runs.

Native Burst scalar tables always identify observations by `Band`, `Phase` and `Value`. Comparison reads those columns even when the same run assigns custom output column names to other metrics. A sensitivity comparison with zero finite pairs saves its reports, explains the failure on standard error and exits with code 1. Interrupting an export or comparison returns code 130.

## Failures and Interruption

| Exit code | Meaning |
|---|---|
| `0` | All required actions completed |
| `1` | Required upstream result unavailable, execution failure, or partial failure |
| `2` | Invalid arguments, target, or configuration |
| `130` | User interruption after cleanup of owned resources |

Completion messages identify the page, target, processing scope, and output
location. Warnings and errors are written to standard error.

If Align Run succeeds but Finish fails, the Run output remains available while
Features stays blocked. If one Tensor metric fails, successfully published
metrics remain available. The command reports partial failure and does not
retry automatically.

Ctrl+C requests interruption. Tensor cleans up its worker processes and pending
writes; additional Ctrl+C signals do not interrupt that cleanup. Localize closes
its application-owned MATLAB runtime on exit.

## Python PSI Execution

The Python PSI grid accepts `outer_n_jobs > 1` for processing time blocks in
separate workers. Raw arrays are serialized to workers without joblib's automatic
array memmapping, so additional workers can increase memory use. Worker count
does not change PSI values, masks, or result freshness. The standard GUI/CLI
orchestration uses one outer PSI worker; this Python argument is not a page or
CLI configuration field.
