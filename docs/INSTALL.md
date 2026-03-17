# Installation Guide

LFP-TensorPipe supports two installation methods:

- `PyInstaller desktop app`: use the PyInstaller-built desktop GUI
- `Developer setup`: run the app from a source checkout in the `lfptp` Conda
  environment

Choose the installation path that matches your setup.

## 1. Choose an Installation Method

| Method | Best For | What You Install |
|---|---|---|
| `PyInstaller desktop app` | Readers who want the packaged GUI | The desktop app together with its bundled Python runtime |
| `Developer setup` | Readers who want a source checkout or local development environment | The repository, a Conda environment, and the package installed into that environment |

## 2. PyInstaller Desktop App

Use this method when you want the GUI without setting up a source checkout.

### 2.1 macOS

1. Download the PyInstaller-built macOS package for your release.
2. Open the `.dmg`.
3. Drag `LFP-TensorPipe-<artifact-version>.app` into `/Applications`.
4. Launch the app from Finder or Launchpad.
5. If Gatekeeper blocks the first launch, right-click the app, choose `Open`,
   and confirm.

### 2.2 Windows

1. Download the PyInstaller-built Windows package for your release.
2. Extract the package to a writable directory.
3. Open the extracted app folder.
4. Launch `LFP-TensorPipe.exe`.

### 2.3 What the Package Includes

The PyInstaller package includes:

- the LFP-TensorPipe GUI
- the Python runtime needed by the app
- packaged icons and default app resources

The PyInstaller package does not include:

- MATLAB
- Lead-DBS

## 3. Developer Setup

Use this method when you want to run the app from the repository.

### 3.1 Create the Environment

Run from the repository root:

```bash
conda env create -f envs/lfptp_py311_base.yml
conda activate lfptp
```

To refresh an existing environment:

```bash
conda env update -f envs/lfptp_py311_base.yml --prune
conda activate lfptp
```

### 3.2 Install the Package

```bash
python -m pip install -e ".[dev]"
```

### 3.3 Launch the App

From the `lfptp` environment:

```bash
lfptensorpipe
```

or:

```bash
lfptp
```

## 4. First Launch

![Settings dialog](assets/app-tutorial/dialog-settings-configs.png)

After either installation method:

1. Launch the app.
2. Open `Settings -> Configs`.
3. Enter valid machine-local paths for:
   - `Lead-DBS Directory`
   - `MATLAB Installation Path`
4. Save the dialog.

Use the following values for the example workflow in
[APP_TUTORIAL.md](APP_TUTORIAL.md):

- `Lead-DBS Directory`: `<lead-dbs-root>`
- `MATLAB Installation Path`: `<matlab-root>`

Replace `<lead-dbs-root>` and `<matlab-root>` with the matching local paths on
your machine.

## 5. Localize Requirements

`Localize` depends on:

- a valid `Lead-DBS Directory`
- a valid `MATLAB Installation Path`
- existing Lead-DBS results for the selected subject inside the chosen project

## 6. Next Step

Continue with the full walkthrough in [APP_TUTORIAL.md](APP_TUTORIAL.md).
