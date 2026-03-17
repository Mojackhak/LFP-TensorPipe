# TECH_STACK — LFP-TensorPipe

## Document Metadata
- Document Version: v1.4
- Owner: Mojackhak
- Last Updated: 2026-03-16
- Status: Approved for Implementation
- Open Questions: None
- Change Log:
  - v1.4 (2026-03-16): Raised the NumPy baseline to `>=2.0,<3` for the developer environment and package metadata, standardizing on `np.trapezoid` in the tabular integration path.
  - v1.3 (2026-03-15): Restricted Extract Features `.xlsx` artifacts to `scalar` outputs while keeping normalization `.xlsx` exports unchanged.
  - v1.2 (2026-03-15): Renamed the Localize app config file from `lead.yml` to `localization.yml` and updated the managed user-storage inventory accordingly.
  - v1.1 (2026-03-15): Moved app-owned YAML persistence to platform user-storage roots, expanded the managed config inventory to include `alignment_preview.yml`, `features_plot.yml`, and `lead.yml`, and relocated `recent_projects.yml` to `state/`.
  - v1.0 (2026-03-15): Renamed the Extract Features defaults config file from `derive.yml` to `features.yml`.
  - v0.1 (2026-02-10): Initial technical stack definition.
  - v0.2 (2026-02-10): Added runtime startup command contract.
  - v0.3 (2026-02-10): Added icon asset generation toolchain notes.
  - v0.4 (2026-02-11): Synced with merged Extract Features + Visualization stage model.
  - v0.5 (2026-02-11): Synced frozen display/internal naming map and Extract Features normalization artifact conventions.
  - v0.6 (2026-02-12): Added deterministic shell appearance/runtime constraints (Fusion + Arial + compact geometry contract) and PNG-first icon source baseline.
  - v0.7 (2026-02-12): Updated default shell geometry to `800x600` while retaining minimum geometry `640x450`.
  - v0.8 (2026-02-26): Added mandatory validation-command rule to run all checks inside conda env `lfptp`.
  - v0.9 (2026-03-11): Added `configs/record.yml` to the project-level configuration inventory for Record Import defaults.

## 1. Runtime and Environment
## 1.1 Python Runtime
- Python: 3.11 (managed via conda environment)

## 1.2 Environment Source of Truth
- Primary environment file: `envs/lfptp_py311_base.yml`
- Version policy:
  - Do not hard-pin every dependency in docs.
  - Resolve versions from the conda environment file during setup.
  - Accept patch-level variability across machines.

## 2. GUI Stack
- Framework: PySide6
- UI model: QWidget-based desktop app, page routing via stage selection
- Plot rendering: Matplotlib Qt backend (`%matplotlib qt` behavior in desktop runtime)
- Runtime shell appearance contract:
  - Qt style engine: `Fusion`
  - Global application font family: `Arial`
  - Default window geometry: `800x600`
  - Minimum window geometry: `640x450`
  - Left column width formula: `clamp(round(window_width * 0.33), 200, 600)`
  - Right stage-area minimum width: `400`

## 3. Scientific and Signal Processing Stack
Core scientific packages (from conda env):
- `numpy>=2.0,<3`
- `scipy`
- `pandas`
- `matplotlib`
- `statsmodels`
- `joblib`
- `tqdm`
- `cloudpickle`
- `h5py`

Neurosignal packages:
- `mne>=1.10`
- `mne-connectivity`
- `autoreject`
- `scikit-learn` (required by `autoreject`)
- `specparam==2.0.0rc6` (pip)
- `tensorpac` (pip)

## 4. Domain Integration
- MATLAB Engine for Python (required for localization/contact viewer workflows)
- Lead-DBS integration through configured user path settings in `paths.yml`

## 5. Data IO and Serialization
- Primary signal format: FIF (`raw.fif`)
- Additional imports: CSV/EDF/BDF/BrainVision and other `mne.io`-supported formats
- Tensor/table serialization:
  - `.pkl` for structured tensors and tabular objects
  - `.xlsx` for Extract Features `scalar` tables and normalization tables
- Logs: `lfptensorpipe_log.json` (UTF-8)
- Configs: YAML in user storage (`configs/*.yml`)
- UI state: YAML in user storage (`state/recent_projects.yml`)

## 6. Code Quality and Testing Toolchain
- Formatter: `black`
- Linter: `ruff`
- Test runner: `pytest`
- Coverage target: 100% line coverage including GUI button flows and core logic paths
- Mandatory execution rule: run all validation/check commands with `conda run -n lfptp ...` (do not use system Python for checks)

## 7. Project-Level Configuration Files
- macOS root: `~/Library/Application Support/LFP-TensorPipe/`
- Windows root: `%LOCALAPPDATA%\\LFP-TensorPipe\\`
- `configs/paths.yml`
- `configs/preproc.yml`
- `configs/tensor.yml`
- `configs/alignment.yml`
- `configs/alignment_preview.yml`
- `configs/features.yml`
- `configs/features_plot.yml`
- `configs/localization.yml`
- `configs/record.yml`
- `state/recent_projects.yml`

## 7.1 Frozen Internal Keys
- `preproc`
- `tensor`
- `alignment`
- `features`

## 8. Build and Execution Model
- Run from source code repository
- Offline local execution only
- No remote API, no cloud jobs, no database service
- Startup command contract:
  1. Install and activate the required conda environment.
  2. Launch the app via terminal command: `lfptensorpipe`.

## 9. Cross-Platform Scope
- Supported OS: macOS and Windows
- No Linux requirement in v0.1

## 10. Icon Build Toolchain (Current Environment)
- Available native tools:
  - `sips` (image resize/conversion)
  - `iconutil` (macOS `.icns` assembly)
- Planned icon build tasks:
  - generate multi-size PNG files from `png/app_icon.png` (SVG fallback allowed)
  - generate `.ico` and `.icns` artifacts
  - wire generated artifacts into Qt startup icon loading.

## 11. Extract Features Artifact Conventions
- Output roots:
  - `features/derivatives`
  - `features/derivatives_transformed`
  - `features/normalization`
  - `features/normalization_transformed`
- Naming:
  - `subparam-{derived_type}.pkl`
  - `subparam-{derived_type}.xlsx` for Extract Features `scalar` outputs and normalization outputs
- Normalization emits both `.pkl` and `.xlsx`.
