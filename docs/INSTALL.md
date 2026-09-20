# Installation Guide

LFP-TensorPipe runs as a desktop application or from a Python source checkout.
The desktop application includes its Python runtime. A source installation also
provides the `lfptp run` command-line workflow.

## 1. Choose an Installation Method

| Method | Use it for | Requirements |
|---|---|---|
| Desktop application | Running the GUI without a Python environment | An application package for your operating system and CPU architecture |
| Source installation | Using the GUI, command-line workflow, or Python package | The source repository and the `lfptp` Conda environment |

MATLAB and Lead-DBS are external dependencies for **Localize** and **Contact
Viewer**. Record import, preprocessing, Tensor computation, alignment, and
feature extraction do not require localization. Results can be analyzed without
location columns.

## 2. Desktop Application

### 2.1 macOS

The Apple Silicon package runs on Macs with an M-series processor.

1. Open the supplied `.dmg` file.
2. Drag `LFP-TensorPipe.app` onto the `Applications` shortcut in the installation
   window. Finder Copy/Paste into `Applications` also installs the application.
3. Eject the disk image.
4. Open `LFP-TensorPipe` from `Applications`.

The application contains its Python runtime, scientific dependencies, icons,
and default settings. It does not require a Conda installation. The local
macOS package is ad-hoc signed and is not notarized; installation remains subject
to your macOS software-security policy.

### 2.2 Windows

For a Windows desktop package:

1. Extract the supplied archive to a writable directory.
2. Keep the application folder and its bundled files together.
3. Launch `LFP-TensorPipe.exe` from that folder.

### 2.3 External Resources

Install MATLAB and Lead-DBS separately when using Localize or Contact Viewer.
Project data, subject reconstructions, normalization transforms, and atlases
are supplied separately from the application.

## 3. Source Installation

Run these commands from the repository root.

### 3.1 Create the Environment

```bash
conda env create -f envs/lfptp_py311_base.yml
conda activate lfptp
```

To synchronize an installed environment with the environment file:

```bash
conda env update -f envs/lfptp_py311_base.yml --prune
conda activate lfptp
```

### 3.2 Install the Package

```bash
python -m pip install -e ".[dev]"
```

### 3.3 Launch the Application

From the `lfptp` environment, run either command:

```bash
lfptensorpipe
```

```bash
lfptp
```

To execute an exported page configuration without opening the GUI, use
`lfptp run`. See [Command-line workflow](CLI.md) for the inputs, supported
pages, outputs, and error handling.

## 4. First Launch

1. Launch the application.
2. Use `Project +` to select a writable project folder.
3. Select or create a subject, then use `Record +` to import a recording.

For localization, open the application configuration dialog:

- macOS: `LFP-TensorPipe -> Preferences...` in the system menu bar.
- Windows: `Settings -> Configs` in the application menu.

![Configs dialog](assets/app-control-reference/controlref-advance-configs-dialog.png)

Set `Lead-DBS Directory` and `MATLAB Installation Path` to the corresponding
installation folders, then click `Save`. The Localize panel reports the MATLAB
connection state. `MATLAB: Ready` confirms the runtime connection; a localization
run also requires the selected subject's anatomical inputs.

## 5. Localize Requirements

Localize uses:

- the configured MATLAB and Lead-DBS installations;
- a subject reconstruction whose contacts match the recording channels;
- the subject's anatomical image and valid normalization transforms;
- an atlas available in the subject's target space and a selection of regions.

Use `Match -> Configure...` to map channels to contacts and `Atlas -> Configure...`
to select the atlas and regions. All recording channels must be mapped before
Apply. See the [Localize tutorial](APP_TUTORIAL.md#4-localize) for the data layout
and workflow.

## 6. Continue

- [Demo tutorial](APP_TUTORIAL.md): import a recording, preprocess it, build
  tensors, align epochs, and visualize extracted features.
- [Control reference](APP_CONTROL_REFERENCE.md): parameters, controls,
  availability rules, and saved results for each page.
- [Command-line workflow](CLI.md): execute exported page configurations.
