# Command-line Workflow

Use `lfptp run` or `lfptensorpipe run` to execute one exported page configuration
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
| `--config` | Required. A JSON file saved by a page's `Export Configs...` action. Its `schema` selects the action. |
| `--trial` | Required for Align Epochs and Extract Features; rejected for Localize and Build Tensor. Use the canonical trial slug, such as `cycle-l`, rather than a directory path. |

Relative paths are resolved from the working directory; `~` expands to the home
directory. `lfptp run --help` displays help without opening the GUI or initializing
MATLAB or scientific computation.

## Supported Actions

| Exported schema | Action | Required input |
|---|---|---|
| `lfptensorpipe.localize-config` | Apply Localize | Imported channels, matching reconstruction and transforms, configured runtime paths |
| `lfptensorpipe.tensor-config` | Build selected Tensor metrics | Current Preprocess Finish |
| `lfptensorpipe.alignment-config` | Align Run, then Finish with every generated epoch | Current Tensor metrics |
| `lfptensorpipe.features-config` | Extract Features for the selected trial | Current Align Finish |

Each invocation processes one record and, when required, one trial. Record
import, Preprocess Signal, batch traversal, and manual epoch selection are GUI
workflows. A command requires its upstream results to be ready.

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

The page JSON does not replace global application settings. Localize runtime
paths and Features output/reducer settings are read from application
configuration. Reproduction therefore requires the same relevant source data,
page parameters, and application settings.

## Trials and Epochs

Align updates the named trial or creates that exact canonical slug after
validating the inputs. An occupied directory that is not a valid trial is an
error. Features requires an existing trial.

A successful Align Run selects every epoch produced by the method and passes
that selection to Finish. Annotation, duration, and BAD/EDGE rules determine
which epochs are generated. To inspect and choose individual epochs, use the
GUI's Epoch Inspector. Both Run and Finish must succeed for the command to
complete successfully.

Align Finish merges location information when the record's Localize result is
current. Otherwise, it produces tables without location information. The
completion message reports location-merge readiness and any service warnings.

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
