# LFP-TensorPipe App

![LFP-TensorPipe app overview](docs/assets/app-control-reference/controlref-basic-main-window-overview.png)

LFP-TensorPipe is a desktop GUI for DBS LFP workflows that works alongside
Lead-DBS projects. It supports record import, synchronization, reset-reference
montage setup, localization, signal preprocessing, tensor building, epoch
alignment, and feature extraction.

## What This App Covers

- import records into a Lead-DBS-compatible project
- reuse existing Lead-DBS outputs for `Localize`
- preprocess signals step by step
- build tensor outputs for multiple metrics
- align epochs for paradigms like gait, ERP paradigms
- extract plot-ready features from aligned outputs

## Documentation Map

- Installation guide: [docs/INSTALL.md](docs/INSTALL.md)
- Demo tutorial: [docs/APP_TUTORIAL.md](docs/APP_TUTORIAL.md)
- Control reference: [docs/APP_CONTROL_REFERENCE.md](docs/APP_CONTROL_REFERENCE.md)

Start with the installation guide. The public repository provides source
snapshots and release downloads; desktop packaging is maintained only in the
private source repository. Then follow the demo tutorial for the validated
single-record workflow. Use the control reference when you need a page-by-page
or dialog-by-dialog explanation.

## Installation

LFP-TensorPipe supports two installation methods:

| Method | Best For | Summary |
|---|---|---|
| `PyInstaller desktop app` | Readers who want the packaged GUI | Install the PyInstaller-built desktop app, then configure `Settings -> Configs` on first launch |
| `Developer setup` | Readers who want a source checkout or a local development environment | Create the `lfptp` Conda environment, install the package in editable mode, and launch the app from that environment |

### PyInstaller Desktop App

1. Download the PyInstaller desktop app package for your platform.
2. Install or unpack it as instructed for that release.
3. Launch the app.
4. Open `Settings -> Configs` and enter valid machine-local paths if you plan to
   use `Localize`.

Desktop app rebuilds are private-repository-only. The public repository is not
a desktop packaging source. Private-repo maintainers should use
`docs/PYINSTALLER_PACKAGING.md` and `docs/RELEASE_RUNBOOK.md` from the
`lfptensorpipe` source repository.

### Developer Setup

```bash
conda env create -f envs/lfptp_py311_base.yml
conda activate lfptp
python -m pip install -e ".[dev]"
```

After launch, use `Settings -> Configs` for `Lead-DBS Directory` and
`MATLAB Installation Path`. The GUI no longer expects a manual MATLAB Engine
package path.

The supported install naming is:

- distribution: `lfp-tensorpipe`
- import: `lfptensorpipe`
- product/UI: `LFP-TensorPipe`

## Practice Setup

Use the following setup while reading
[docs/APP_TUTORIAL.md](docs/APP_TUTORIAL.md):

- `Project`: `<demo-project-root>`
- `Lead-DBS Directory`: `<lead-dbs-root>`
- `MATLAB Installation Path`: `<matlab-root>`
- `Subject`: `sub-001`
- `Record`: `gait`
- `Trial`: `cycle_l`

Replace `<demo-project-root>`, `<lead-dbs-root>`, and `<matlab-root>` with the
matching local paths on your machine.

`Project` uses a Lead-DBS-compatible layout, so you can point it directly to an
existing Lead-DBS toolbox project path. In the tutorial, `<demo-project-root>`
means the root of your local demo project copy.

## Workflow At a Glance

1. Select `Project` and `Subject`.
2. Import `sub-001 / gait` with `Sync` and `Reset Reference`.
3. Configure `Localize` if representative coordinates and brain region mapping are required.
4. Run `Preprocess Signal`.
5. Run `Build Tensor`.
6. Run `Align Epochs`.
7. Run `Extract Features`.
8. Plot one result from `Available Features`.

Each downstream stage is gated by the upstream stage: `Build Tensor` by
`Preprocess Signal`, `Align Epochs` by `Build Tensor`, and `Extract Features`
by the selected trial in `Align Epochs`. `Localize` is separate.

## Key UI Concepts

- Stage lights are log-driven and use `gray`, `yellow`, and `green`.
- Inline panel indicators are draft-aware and compare the visible settings
  against the latest successful run or apply.
- `Import Configs...` and `Export Configs...` move record-scoped or trial-scoped
  JSON payloads, not whole projects.
- In `Import Record`, `Confirm Import` stays disabled until required `Sync` and
  `Reset Reference` settings are saved.
- In `Preprocess`, `Filter -> Plot` is where you review and edit `bad` spans,
  while `Annotations` is where you manage named event annotations.
- `Localize` atlas choices are discovered from the configured Lead-DBS
  installation for the current subject space.
- When `Localize` is green, `Align Epochs -> Finish` attempts to merge the
  Localize representative-coordinate columns into the finished raw tables.

Continue with [docs/APP_TUTORIAL.md](docs/APP_TUTORIAL.md) for the complete
single-record walkthrough, or open
[docs/APP_CONTROL_REFERENCE.md](docs/APP_CONTROL_REFERENCE.md) for detailed
control descriptions.
