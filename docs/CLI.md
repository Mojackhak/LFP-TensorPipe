# Command-line workflow

The Python package exposes `lfptp run` and `lfptensorpipe run` for running one
page configuration without opening the GUI. Use the installed package in the
`lfptp` Conda environment. Running either command without arguments opens the
desktop application. Standalone console packaging of desktop installers is
outside this interface.

## Inputs

```bash
conda run -n lfptp lfptp run \
  --subject-path /data/project/derivatives/lfptensorpipe/sub-001 \
  --record walk \
  --trial gait \
  --config /data/configs/alignment.json
```

- `--subject-path` is required and must identify an existing
  `<project>/derivatives/lfptensorpipe/<subject>` directory. Project and subject
  are resolved from that layout. Relative paths are relative to the working
  directory; `~` is expanded.
- `--record` is required and names an existing imported record.
- `--config` is required and identifies one JSON file exported by a page's
  **Export Configs** action. Its `schema` selects the action. The existing
  JSON structures and supported schema versions are unchanged.
- `--trial` is required for Align Epochs and Extract Features, and is rejected
  for Localize and Build Tensor. It is an exact canonical trial slug, not a
  filesystem path.
- `lfptp run --help` displays help without initializing the GUI, application
  settings, MATLAB, or scientific computation.

## Actions

| Exported schema | Action | Required existing input |
|---|---|---|
| `lfptensorpipe.localize-config` | Apply Localize | Imported channels, matching reconstruction, configured runtime paths |
| `lfptensorpipe.tensor-config` | Build selected Tensor metrics | Current Preprocess Finish |
| `lfptensorpipe.alignment-config` | Align Run, then Finish with every generated epoch | Current Tensor metrics |
| `lfptensorpipe.features-config` | Extract Features for the selected trial | Current Align Finish |

Preprocess Signal, record import, batch traversal, and manual epoch selection
are not CLI actions. Each invocation operates on one record and, where needed,
one trial. It does not run missing upstream pages automatically.

For Tensor, `selected_metrics` selects computation. For Align, all current
Tensor metrics are processed. Features processes all metrics in the current
accepted Finish; every metric must have its required axis configuration in the
JSON. UI fields such as `active_metric` do not select computation. Unused metric
drafts do not cause unrelated resource checks or computation.

CLI configuration input is strict for fields used by the requested computation:
missing required fields, invalid values, and incompatible selected channels,
pairs, annotations, leads, or axes are errors. Existing equivalent normalization
and supported schema reading are reused. CLI input does not silently remove
requested inputs or replace invalid computation values with application defaults.
Optional values retain the existing service defaults. Unsupported Tensor parameter
names are rejected for selected metrics. The existing exported unlimited
SpecParam peak count remains supported.
The GUI retains its existing interactive import-and-review behavior.

Settings outside the exported page contract continue to use the existing app
configuration, including Localize runtime paths and Features output/reducer
settings. Supplying a page JSON does not install it as global defaults. Results
are reproducible against the same relevant inputs and application settings;
the page JSON alone is not a portable runtime environment.

## Trials and epochs

Align updates the named trial if it exists, or creates that exact trial slug
after validating its inputs. An occupied directory that is not a valid trial
is an error; it is not renamed or replaced. Features requires an existing trial.

After a successful Align Run, all epoch indices returned by that run are saved
as the current selection and passed to Finish. Existing annotation, duration,
and BAD/EDGE method rules still determine which epochs the run produces. Previous
manual epoch picks are replaced by the complete set from this run. Both Run and
Finish must succeed for the command to succeed.

Localize is independent. Finish preserves the existing automatic location-merge
rules: it merges location information when the Localize result is current, and
otherwise produces tables without location information. The completion message
reports location-merge readiness and any existing service warnings.

## Persistence, reruns, and failures

CLI writes standard results below the selected record, using the same services,
logs, output transactions, and generation checks as the GUI. It updates only
the relevant page/trial in the existing record state. Reopening or reselecting
the record in the GUI restores the CLI configuration and result state.

Use one writer for a record at a time; do not run GUI or CLI writes against that
record concurrently. There is no live synchronization between open GUI drafts
and another process.

Every explicit `run` invocation reruns the requested action, using the existing
service behavior to update its standard outputs. There is no automatic skip,
resume, global cache, or configuration-file watcher. JSON whitespace and key
ordering do not affect effective parameters or widen invalidation.

| Published result change | Existing downstream invalidation |
|---|---|
| Tensor metrics | Trials using those metrics and their Features |
| Align Run | That trial's Finish and Features |
| Align Finish | That trial's Features |
| Localize Apply | That record's Align Finish and Features |
| Features | That trial's feature outputs |

Trial-level invalidation remains because accepted Run/Finish results use shared
trial generations. Localize affects all trial finishes in the record because
they share the record's localization inputs. No project-wide invalidation is
introduced.

A successful Align Run remains available if Finish fails; the command reports
failure and Features cannot consume an incomplete Finish. Tensor metrics already
successfully published remain available if another selected metric fails. There
is no cross-page transaction or automatic retry.

| Exit code | Meaning |
|---|---|
| `0` | All required actions completed |
| `1` | Required upstream result unavailable, execution failure, or partial failure |
| `2` | Invalid arguments, target, or configuration |
| `130` | User interruption after cleanup of owned resources |

Completion messages report the page, target, processing scope, and result
location. Warnings and known errors go to standard error. Unexpected failures
remain visible. Tensor interruption reuses its worker/process-tree cancellation
and transaction recovery; Localize closes its owned MATLAB runtime on exit.
During Tensor shutdown, additional Ctrl+C signals do not interrupt cleanup.
