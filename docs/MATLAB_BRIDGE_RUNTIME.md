# MATLAB_BRIDGE_RUNTIME - LFP-TensorPipe

## Document Metadata
- Document Version: v0.1
- Owner: Mojackhak
- Last Updated: 2026-03-16
- Status: Active
- Open Questions: Windows bootstrap validation remains pending.

## 1. Goal

This document defines the shared MATLAB bootstrap model used by both:

- the PyInstaller desktop app
- the Conda-based developer setup

The design goal is to keep the user-facing configuration simple:

- users select a `MATLAB Installation Path`
- users select a `Lead-DBS Directory`
- the app performs the remaining MATLAB Engine bootstrap automatically

Users should not need to manually install `matlab.engine` or browse to a
`setup.py` directory.

## 2. User-Facing Contract

The app now treats the following paths as the only machine-local Localize
settings:

- `leaddbs_dir`
- `matlab_root`

The legacy `matlab_engine_path` setting is migrated automatically when
possible. The GUI no longer exposes that legacy field.

## 3. Managed Runtime Layout

The MATLAB bridge runtime is stored in the platform user-storage root managed
by `AppConfigStore`.

macOS example:

```text
~/Library/Application Support/LFP-TensorPipe/
  cache/
    matlab_bridge/
      R2024b/
        matlab/
          __init__.py
          engine/
            _arch.txt
            ...
```

The managed cache is app-owned data, not part of the bundled `.app`, and not
part of the repository checkout.

## 4. Bootstrap Source

The bootstrap uses files from the user-selected MATLAB installation:

- `{matlab_root}/extern/engines/python/dist/matlab/`
- `{matlab_root}/extern/engines/python/build/lib/matlab/engine/_arch.txt` when
  available
- `{matlab_root}/bin/<arch>/`
- `{matlab_root}/extern/bin/<arch>/`

If `_arch.txt` is not available from MATLAB, the app writes an equivalent file
into the managed cache using the selected MATLAB installation paths.

## 5. Runtime Model

The runtime model is intentionally shared between desktop and developer modes:

1. load `paths.yml`
2. resolve `matlab_root`
3. ensure the managed bridge cache exists under app storage
4. prepend the managed cache to `sys.path`
5. import `matlab.engine`
6. start MATLAB and add Lead-DBS plus repository-owned MATLAB helper functions

This keeps the MATLAB execution path aligned across:

- `LFP-TensorPipe-<artifact-version>.app`
- `conda activate lfptp && lfptp`

## 6. Desktop Packaging Implications

The PyInstaller app does not bundle MATLAB or a preinstalled MATLAB Engine
wheel. Instead, the app performs the MATLAB bootstrap on first use after the
user selects a valid MATLAB installation directory.

This keeps the desktop package independent from one specific MATLAB release
while still allowing Localize to run with the user's local MATLAB installation.

## 7. Validation Rules

The bootstrap validates:

- the MATLAB installation directory exists
- the MATLAB installation contains the expected `extern/engines/python` layout
- the MATLAB executable exists for the current platform
- the Lead-DBS directory exists

When bootstrap fails, the app surfaces a user-facing message that refers to the
MATLAB installation directory rather than to a Python package source path.

## 8. Migration Rules

Older configs may still contain `matlab_engine_path`.

Migration policy:

1. if `matlab_root` is present, use it
2. otherwise, if `matlab_engine_path` is present, attempt to infer
   `matlab_root` from the MATLAB engine source tree
3. if inference is not possible, preserve the legacy raw value as
   `matlab_root` so the user does not lose the configured location
4. persist the migrated `matlab_root` immediately when the config is loaded
5. remove `matlab_engine_path` from the saved config after migration
6. stop exposing `matlab_engine_path` in the GUI

## 9. Non-Goals

This design does not attempt to:

- install a separate user-managed Conda environment
- install a separate app-managed Python environment for MATLAB
- mutate the `.app` bundle at runtime
- require the user to run `pip install` manually
- support MATLAB-free Localize execution

## 10. Expected Outcome

After the first successful bootstrap:

- desktop users can run Localize from the PyInstaller app after selecting their
  local MATLAB installation
- developer users follow the same Localize bootstrap path instead of a separate
  manual `matlab.engine` installation flow
