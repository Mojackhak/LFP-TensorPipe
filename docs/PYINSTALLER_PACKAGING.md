# PYINSTALLER_PACKAGING - LFP-TensorPipe

## Document Metadata
- Document Version: v0.8
- Owner: Mojackhak
- Last Updated: 2026-03-18
- Status: Active
- Open Questions: None

## 1. Goal

This document defines the app-first desktop packaging workflow built with
PyInstaller.

Current target outputs:

- macOS app bundle: `LFP-TensorPipe-<artifact-version>.app`
- macOS installer media: `LFP-TensorPipe-<artifact-version>.dmg`
- Windows `onedir` desktop bundle:
  `LFP-TensorPipe-<artifact-version>/LFP-TensorPipe.exe`
- Windows release archive:
  `LFP-TensorPipe-<artifact-version>-windows-x86_64.zip`

Artifact naming and metadata rules:

- `<artifact-version>` is derived from the same Git tag lineage used by
  `hatch-vcs`
- release builds from `vX.Y.Z` publish `X.Y.Z`
- non-tagged builds follow the same next-dev lineage as `hatch-vcs` and
  publish a sanitized snapshot form such as
  `X.Y.(Z+1)-devN-g<sha>[-dYYYYMMDD]`
- the internal product name remains `LFP-TensorPipe`
- macOS app metadata must write `CFBundleShortVersionString` and
  `CFBundleVersion`
- Windows executable metadata must write numeric file/product versions plus a
  human-readable product-version string

## 2. Scope

The current implementation slice is `P0-P4`:

- `P0`: choose an app-first packaging target and user-visible install model
- `P1`: make the runtime entrypoints compatible with a frozen app
- `P2`: add PyInstaller spec/build helpers and packaged resource collection
- `P3`: produce a macOS `.app` and wrap it in a `.dmg`
- `P4`: produce a Windows `onedir` app directory and wrap it in a `.zip`

## 3. Why PyInstaller

PyInstaller is the current packaging direction because the repository now
optimizes for a native desktop-app experience:

- Finder-launchable `.app` on macOS
- app bundle that can live in `/Applications`
- `.dmg` distribution flow that matches normal macOS installs
- direct path toward an optional later signing and notarization workflow

The packaging model uses `--windowed --onedir`, following the PyInstaller
one-folder recommendation for easier debugging and more predictable bundled
resource paths. This also aligns with the PyInstaller documentation for macOS
app bundles and frozen multiprocessing support. See the official docs:

- [Using PyInstaller](https://pyinstaller.org/en/stable/usage.html)
- [Using Spec Files](https://pyinstaller.org/en/stable/spec-files.html)
- [Run-time Information](https://pyinstaller.org/en/stable/runtime-information.html)
- [Common Issues and Pitfalls](https://pyinstaller.org/en/stable/common-issues-and-pitfalls.html)

## 4. Build Model

The current PyInstaller flow is:

1. prepare or reuse icon assets
2. run PyInstaller from the private source repository using a checked-in spec
3. produce the native app output for the target platform
4. wrap that output into the platform-specific release artifact

This packaging path builds directly from the validated source tree rather than
from an editable install or an environment-image installer model.

Because the app is frozen directly from source, the spec must also map
repository-owned runtime assets into their in-bundle package locations:

- `src/configs/*.yml` -> `lfptensorpipe/resources/config_defaults/`
- full `mne` submodule collection -> bundled implementation modules required by
  MNE lazy-loader imports in the frozen app
- `mne/**/*.pyi` -> bundled stub files required by `lazy_loader.attach_stub()`
  at runtime; collecting implementation modules alone is not sufficient
- `mne/html_templates/**/*.{jinja,css,js}` -> bundled MNE HTML template assets
  required by frozen `mne` imports
- `mne_qt_browser/icons/**/*` -> bundled icon theme assets required by
  `mne_qt_browser` when `raw.plot()` resolves icons relative to `__file__`
- `autoreject` -> bundled explicitly because `lfptensorpipe.preproc.filter`
  imports it lazily at runtime during `Filter -> Apply`; analysis-time imports
  alone do not keep it in the frozen app
- `scikit-learn` -> must remain bundled because `autoreject` imports it at
  runtime; excluding `sklearn` breaks `Filter -> Apply` in the packaged app
- `mne_connectivity` -> must remain bundled because tensor connectivity metrics
  import it lazily at runtime; excluding it breaks packaged Coherence / PLV /
  ciPLV / PLI / wPLI / PSI / TRGC execution
- `xarray` -> must remain bundled because `mne_connectivity.base` imports it at
  module-import time; excluding `xarray` causes the frozen runtime to surface a
  misleading "mne_connectivity is required" error during tensor connectivity runs
- `mne_connectivity` distribution metadata -> must remain bundled because
  `mne_connectivity._version` calls `importlib.metadata.version(__package__)`
  at import time; excluding the package metadata causes the frozen runtime to
  fail with `No package metadata was found for mne_connectivity`
- Matplotlib PDF backend support -> must remain bundled because SpecParam report
  export writes PDF files during packaged tensor runs; collecting only `Agg`
  breaks packaged `Periodic/APeriodic` report generation

## 5. Runtime Requirements

The frozen app must support these runtime rules:

- the GUI, tensor worker, runtime-plan worker, and Localize viewer worker all
  dispatch from one bundled executable
- `multiprocessing.freeze_support()` runs before any frozen-worker path
- worker launches do not depend on `python -m ...` when the app is frozen
- packaged resources are accessed through stable package-relative paths
- the desktop app must bootstrap MATLAB Engine from a user-selected MATLAB
  installation path without asking the user for a package source directory

## 6. Bundled and External Components

Bundled into the app package:

- Python runtime
- `lfptensorpipe` application code
- packaged icons
- default YAML templates
- MATLAB helper `.m` files owned by this repository
- the default YAML templates remapped from `src/configs/`
- the full `mne` Python module graph required by lazy-loaded runtime imports
- the `mne` `.pyi` stub files required by the MNE lazy-loader runtime
- the `mne.html_templates` HTML template assets used by lazy-loaded `mne`
  internals
- the `mne_qt_browser` icon-theme assets required by the packaged Qt browser

Not bundled into the app package:

- MATLAB
- Lead-DBS
- signing identities or notarization assets

The MATLAB Engine bootstrap is shared between desktop and developer modes. The
desktop app stores its managed MATLAB bridge cache under the app-storage root
instead of mutating the `.app` bundle.

## 7. Build Prerequisites

The local machine must provide:

- `conda`
- the `lfptp` Conda environment
- `pyinstaller` installed into `lfptp`
- macOS `hdiutil` for `.dmg` creation when building the macOS artifact

Minimal local setup in the repository root:

```bash
conda activate lfptp
python -m pip install pyinstaller
python -m PyInstaller --version
```

## 8. Repository Layout

Tracked packaging assets live under:

- `packaging/pyinstaller/`
- `tools/release/build_pyinstaller.py`
- `tools/release/build_pyinstaller_macos.sh`

The Windows helper now lives under:

- `tools/release/build_pyinstaller_windows.ps1`

Generated artifacts should stay under ignored build/output directories such as:

- `build/pyinstaller/`
- `dist/desktop/`

Additional macOS release helpers now live under:

- `tools/release/sign_macos_app.py`
- `tools/release/sign_macos_app.sh`
- `tools/release/notarize_macos_app.py`
- `tools/release/notarize_macos_app.sh`

The default macOS release path does not require these helpers. They remain
available only as an optional advanced workflow for a future signed release.

The packaged macOS regression checklist is maintained in:

- `docs/MACOS_APP_REGRESSION.md`
- `docs/APP_REVIEW_PLAN.md`
- `docs/NUMERICAL_VALIDATION_PLAN.md`

## 9. First Real Build Commands

PyInstaller must build native bundles on the target platform. The current
implemented platforms are macOS and Windows.

### 9.1 macOS

Minimal first build from the private repository root:

```bash
conda activate lfptp
python -m pip install pyinstaller
./tools/release/build_pyinstaller_macos.sh
```

Equivalent direct Python entry point:

```bash
python tools/release/build_pyinstaller.py --target-platform macos
```

Optional helper modes:

```bash
python tools/release/build_pyinstaller.py --target-platform macos --skip-dmg
python tools/release/build_pyinstaller.py --target-platform macos --dmg-only
```

Internal packaged-app smoke command for frozen raw plotting:

```bash
dist/desktop/macos/LFP-TensorPipe-<artifact-version>.app/Contents/MacOS/LFP-TensorPipe \
  --smoke-raw-plot-fif \
  /Users/mojackhu/Research/pd1/derivatives/lfptensorpipe/sub-001/gait/preproc/raw/raw.fif \
  --smoke-raw-plot-close-ms 1500
```

Internal packaged-app smoke command for demo record parsing:

```bash
dist/desktop/macos/LFP-TensorPipe-<artifact-version>.app/Contents/MacOS/LFP-TensorPipe \
  --smoke-demo-records-root \
  /Users/mojackhu/Github/lfptensorpipe/demo/records
```

Packaged-app smoke command for GUI demo-record import:

```bash
dist/desktop/macos/LFP-TensorPipe-<artifact-version>.app/Contents/MacOS/LFP-TensorPipe \
  --smoke-demo-record-imports-root \
  /Users/mojackhu/Github/lfptensorpipe/demo/records
```

Packaged-app smoke command for demo config import:

```bash
dist/desktop/macos/LFP-TensorPipe-<artifact-version>.app/Contents/MacOS/LFP-TensorPipe \
  --smoke-demo-configs-root \
  /Users/mojackhu/Github/lfptensorpipe/demo/configs \
  --smoke-project-root /Users/mojackhu/Research/pd1 \
  --smoke-subject sub-001 \
  --smoke-record gait
```

Packaged-app smoke command for Preprocess-page UI coverage:

```bash
dist/desktop/macos/LFP-TensorPipe-<artifact-version>.app/Contents/MacOS/LFP-TensorPipe \
  --smoke-preproc-ui \
  --smoke-project-root /Users/mojackhu/Research/pd1 \
  --smoke-subject sub-001 \
  --smoke-record gait
```

Current macOS status:

- packaged demo-record parser smoke: pass
- packaged demo-record import smoke: pass
- packaged demo-config import smoke: pass
- packaged Preprocess-page smoke: pass

Implementation note:

- packaged step-plot validation should prefer a subprocess call back into the
  same bundled executable when the smoke runner needs to close an `mne_qt_browser`
  window automatically; this avoids false negatives caused by in-process browser
  teardown during automated regression runs

Expected outputs:

```text
dist/desktop/macos/LFP-TensorPipe-<artifact-version>.app
dist/desktop/macos/LFP-TensorPipe-<artifact-version>.dmg
```

Default macOS release artifact:

- unsigned `LFP-TensorPipe-<artifact-version>.dmg`

### 9.2 Windows

Minimal first build from the private repository root on Windows:

```powershell
conda activate lfptp
python -m pip install pyinstaller
.\tools\release\build_pyinstaller_windows.ps1
```

Equivalent direct Python entry point:

```powershell
python tools/release/build_pyinstaller.py --target-platform windows
```

Optional helper mode:

```powershell
python tools/release/build_pyinstaller.py --target-platform windows --dry-run
```

Windows rebuild note:

- the final `dist/desktop/windows/LFP-TensorPipe-<artifact-version>` directory
  is replaced in place on each rebuild for the same resolved version
- any running
  `dist/desktop/windows/LFP-TensorPipe-<artifact-version>/LFP-TensorPipe.exe`
  instance can lock that destination and must be treated as a fatal rebuild
  blocker
- the PowerShell wrapper must propagate the underlying Python helper exit code
  unchanged so shell automation can detect failed Windows builds

Expected outputs:

```text
dist/desktop/windows/LFP-TensorPipe-<artifact-version>/LFP-TensorPipe.exe
dist/desktop/windows/LFP-TensorPipe-<artifact-version>-windows-x86_64.zip
```

Default Windows release artifact:

- `dist/desktop/windows/LFP-TensorPipe-<artifact-version>-windows-x86_64.zip`

Implementation note:

- if the icon master source `app_icon.png` / `app_icon.svg` is not present in
  the repository checkout, the build helper reuses the committed generated icon
  assets instead of failing the package build

For the detailed implementation and validation record, see
[WINDOWS_PYINSTALLER_PLAN.md](WINDOWS_PYINSTALLER_PLAN.md).

## 10. First-Build Notes

Before the first real build:

- confirm the private repository worktree is clean
- confirm icon assets exist; the build helper regenerates them only when the
  icon master source is present, otherwise it reuses committed generated icons
- confirm `pyinstaller` resolves from the active `lfptp` environment
- confirm the native build runs on the target platform

The build helper is expected to:

1. clean previous PyInstaller build output for the target platform
2. run the checked-in PyInstaller spec
3. create the native app output for that platform
4. create the platform-specific release artifact:
   - macOS: `.dmg`
   - Windows: `.zip`

The default build remains unsigned and not notarized. This is acceptable for
informal distribution and internal beta use, but users should expect Gatekeeper
warnings on first launch.

## 11. Acceptance Checklist

Use the following checklist after producing a real app package.

### 11.1 macOS Acceptance Checklist

- Confirm `LFP-TensorPipe-<artifact-version>.app` launches from Finder without
  showing a terminal window.
- Confirm the app can be moved to `/Applications` and launched from there.
- Confirm the app opens, closes, and opens again without crashing.
- Confirm the user config directory is created under
  `~/Library/Application Support/LFP-TensorPipe`.
- Confirm icons appear correctly in Finder, the dock, and the window title bar.
- Confirm Dataset, Preprocess Signal, Build Tensor, Align Epochs, and Extract
  Features all open without import/runtime errors.
- Confirm the tensor worker and runtime-plan worker launch successfully from the
  frozen app.
- Confirm `Settings -> Configs` now accepts `MATLAB Installation Path` instead
  of a raw MATLAB Engine package directory.
- Confirm the first Localize warmup can create the managed MATLAB bridge cache
  under app storage without mutating the `.app` bundle.
- Confirm Localize failures report a clear MATLAB-installation or Lead-DBS
  problem instead of asking the user to run `pip install`.
- Confirm the `.dmg` mounts successfully and exposes the expected app bundle.
- Confirm the release notes explain that the macOS build is unsigned and
  requires a manual first-launch approval path when Gatekeeper blocks it.

### 11.2 Windows Acceptance Checklist

- Confirm `LFP-TensorPipe.exe` launches from the unpacked `onedir` output.
- Confirm the app can launch from a path that contains spaces.
- Confirm the user config directory is created under
  `%LOCALAPPDATA%\\LFP-TensorPipe`.
- Confirm the tensor worker and runtime-plan worker launch successfully from
  the frozen app.
- Confirm `Settings -> Configs` accepts `MATLAB Installation Path`.
- Confirm the first Localize warmup can create the managed MATLAB bridge cache
  on Windows.
- Confirm the zipped `onedir` artifact can be unpacked and launched on a clean
  Windows machine.

### 11.3 Shared Release Acceptance Checklist

- Confirm the packaged app version matches the release tag.
- Confirm the bundled app was built from the tagged private-repository commit.
- Confirm macOS `Info.plist` and Windows executable version metadata match the
  same resolved artifact version.
- Confirm release notes state the current desktop-package scope and Localize
  limitations clearly.
- Confirm the public release uploads the desktop artifact (`.dmg` on macOS,
  `.zip` on Windows) rather than raw PyInstaller scratch output.
- Confirm the public release labels the macOS artifact as an unsigned or
  informal build.

## 12. Bundle Size Diagnostics and Reduction

The first real macOS PyInstaller build was large enough that size had to be
treated as a release concern:

- initial app bundle: about `1.8G`
- initial `.dmg`: about `707M`

After tightening the spec and removing unneeded interactive/3D stacks from the
desktop build, the current macOS outputs are approximately:

- current app bundle: about `910M`
- current `.dmg`: about `420M`

When diagnosing bundle size on macOS, do not add the reported sizes of
`Contents/Frameworks` and `Contents/Resources` together and treat that sum as
the release size. The release gate should use:

- the bundle root size, for example
  `du -sh dist/desktop/macos/LFP-TensorPipe-<artifact-version>.app`

## 13. MATLAB Bridge Bootstrap

The desktop package now assumes a shared MATLAB bootstrap model:

1. the user saves `MATLAB Installation Path`
2. the app derives the MATLAB Engine source layout from that installation
3. the app creates a managed MATLAB bridge cache under app storage
4. the app imports `matlab.engine` from that managed cache

For the full architecture, see
[MATLAB_BRIDGE_RUNTIME.md](MATLAB_BRIDGE_RUNTIME.md).

## 14. Bundle Size Diagnostics and Reduction

- the distribution image size, for example
  `du -sh dist/desktop/macos/LFP-TensorPipe-<artifact-version>.dmg`

Recommended inspection commands:

```bash
du -sh dist/desktop/macos/LFP-TensorPipe-<artifact-version>.app
du -sh dist/desktop/macos/LFP-TensorPipe-<artifact-version>.dmg
du -sh dist/desktop/macos/LFP-TensorPipe-<artifact-version>.app/Contents/Frameworks/* | sort -h | tail -n 40
du -sh dist/desktop/macos/LFP-TensorPipe-<artifact-version>.app/Contents/Resources/* | sort -h | tail -n 40
```

The initial bundle shows that the largest contributors are:

- `vtkmodules`
- `llvmlite`
- `scipy`
- `sklearn`
- `PySide6`

The size-reduction policy for the PyInstaller app is:

1. remove broad hidden-import collection before excluding packages manually
2. keep the desktop app focused on the frozen GUI scope, not the full developer
   environment
3. prefer targeted `excludes` in the spec over deleting bundled files after the
   build
4. rerun the smoke test after every spec reduction

The current implementation target is to eliminate packaging of modules that are
not needed for the app-first desktop slice, starting with whole-package hidden
imports and unused interactive tooling.

The latest reduction step removed the unused
`src/lfptensorpipe/anat/voxelize.py` module from the repository and excludes
`scikit-learn` from both the package dependency graph and the PyInstaller
desktop bundle.

The next reduction pass targets hook-driven extras that are not part of the
desktop GUI contract:

- force Matplotlib backend collection to `Agg`
- exclude Pillow `ImageTk` support
- prune unused `libtcl` and `libtk` payloads from the macOS bundle while
  keeping `pyqtgraph` and `mne_qt_browser`
- exclude pandas-adjacent optional storage and stats stacks that are not
  imported by the desktop app (`statsmodels`, `tables`, `xarray`, `zarr`,
  `netCDF4`, `numcodecs`, `seaborn`) while keeping `pandas`, `openpyxl`, and
  `h5py`
- exclude the unused `QtDataVisualization` compatibility path from `qtpy` while
  keeping `QtSvg` and `QtTest` for `pyqtgraph`/`mne_qt_browser`
- enable macOS binary stripping during PyInstaller collection and keep the
  smoke test as the acceptance gate

The packaging regression to avoid is trying to keep `mne` “light” by bundling
only stubs or a few targeted submodules. The frozen app needs the full `mne`
submodule graph for lazy imports such as `mne.html_templates._templates` and
`mne.io.array`; otherwise record parsing fails before any CSV or
vendor-specific validation runs.

## 15. Optional macOS Signing and Notarization Preparation

The app-first packaging flow can support Developer ID signing and Apple
notarization after the `.app` and `.dmg` are reproducible, but this is not the
default release path for the current macOS build.

Official references:

- [Signing Mac Software with Developer ID](https://developer.apple.com/developer-id/)
- [Notarizing macOS software before distribution](https://developer.apple.com/documentation/xcode/notarizing_macos_software_before_distribution)
- [Customizing the notarization workflow](https://developer.apple.com/documentation/security/customizing-the-notarization-workflow)

### 13.1 Required local prerequisites

The signing machine must provide:

- an installed `Developer ID Application` certificate in the login keychain
- Xcode command line tools with `codesign`, `xcrun notarytool`, and
  `xcrun stapler`
- an Apple Developer team with notarization access
- a stored notarytool keychain profile

Recommended local checks:

```bash
security find-identity -v -p codesigning
xcrun notarytool help
xcrun stapler help
codesign --help
```

### 13.2 Credential bootstrap

Recommended one-time setup for notarization credentials:

```bash
xcrun notarytool store-credentials "lfptensorpipe-notary" \
  --apple-id "YOUR_APPLE_ID" \
  --team-id "YOUR_TEAM_ID"
```

App Store Connect API keys are also supported by `notarytool`, but the current
runbook assumes a reusable keychain profile because it keeps the release
commands shorter and avoids embedding secrets into shell history.

### 13.3 Environment variables expected by release helpers

Release helpers should read these variables:

- `LFPTP_CODESIGN_IDENTITY`
- `LFPTP_NOTARY_PROFILE`
- `LFPTP_CODESIGN_TEAM_ID` (optional verification hint)

Example exports:

```bash
export LFPTP_CODESIGN_IDENTITY="Developer ID Application: YOUR NAME (TEAMID)"
export LFPTP_NOTARY_PROFILE="lfptensorpipe-notary"
export LFPTP_CODESIGN_TEAM_ID="TEAMID"
```

Tracked helper entry points:

```bash
./tools/release/sign_macos_app.sh
./tools/release/notarize_macos_app.sh
```

### 13.4 Release signing flow

Recommended order for a release build:

1. build the unsigned `.app`
2. sign the `.app` with hardened runtime enabled
3. verify the signed `.app`
4. create the distributable `.dmg` from the signed `.app`
5. submit the `.dmg` to the Apple notary service with `notarytool`
6. staple the ticket to both the `.app` and the `.dmg`
7. run post-staple verification with `spctl`

Representative commands:

```bash
codesign --force --deep --options runtime --timestamp \
  --sign "$LFPTP_CODESIGN_IDENTITY" \
  dist/desktop/macos/LFP-TensorPipe-<artifact-version>.app

codesign --verify --deep --strict --verbose=2 \
  dist/desktop/macos/LFP-TensorPipe-<artifact-version>.app

xcrun notarytool submit dist/desktop/macos/LFP-TensorPipe-<artifact-version>.dmg \
  --keychain-profile "$LFPTP_NOTARY_PROFILE" \
  --wait

xcrun stapler staple dist/desktop/macos/LFP-TensorPipe-<artifact-version>.app
xcrun stapler staple dist/desktop/macos/LFP-TensorPipe-<artifact-version>.dmg
spctl --assess --type exec --verbose=4 \
  dist/desktop/macos/LFP-TensorPipe-<artifact-version>.app
```

### 13.5 Scope of the current preparation work

The current repository work should prepare the signing and notarization path by
adding documented prerequisites and release-helper entry points. It does not
assume that a Developer ID certificate or Apple notarization credentials are
already available on every development machine.
