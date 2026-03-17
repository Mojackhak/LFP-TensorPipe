# RELEASE_STRATEGY - LFP-TensorPipe

## Document Metadata
- Document Version: v0.6
- Owner: Mojackhak
- Last Updated: 2026-03-18
- Status: Active
- Open Questions: None

## 1. Purpose

This document defines the release model for LFP-TensorPipe across two GitHub
repositories:

- Private source repository: `lfptensorpipe`
- Public mirror repository: `LFP-TensorPipe`

The private repository is the only development source of truth. The public
repository is a release mirror created from an exported snapshot.

## 2. Repository Roles

### 2.1 Private source repository: `lfptensorpipe`

The private repository owns:

- day-to-day development
- full test and validation history
- internal release runbooks
- private mirror-management helpers
- signing and notarization material
- release-only build outputs before publication

No public release should be assembled by editing the public repository first.

### 2.2 Public mirror repository: `LFP-TensorPipe`

The public repository receives:

- approved source snapshots exported from the private repository
- the full `src/lfptensorpipe/` application and package tree
- packaged config defaults from `src/configs/`
- repository demo fixtures from `demo/`
- public-facing installation and usage documentation, including
  `docs/APP_TUTORIAL.md`
- tutorial screenshot assets under `docs/assets/app-tutorial/`
- selected developer-facing runtime documentation
- public-safe desktop packaging documentation
- public-safe desktop build helper scripts and PyInstaller specs
- one lightweight public CI workflow, one Python release workflow, and safe
  helper tools
- exclusion of release-helper tests that depend on private-only packaging
  modules
- release tags that match the private source tag
- developer artifacts such as `wheel` and `sdist`
- desktop artifacts such as macOS and Windows installers

The public repository must not become a second development trunk.

Public mirror CI is intentionally minimal and limited to editable-install plus
import verification on Ubuntu. Public pytest smoke gates are intentionally
omitted. Full-suite pytest validation remains a private-repository
responsibility, and Python release artifacts are built by a separate public tag
workflow rather than by the default CI gate.

## 3. Distribution Lines

### 3.1 Developer distribution

Developer releases are intended for users who need source access, editable
installs, tests, or local debugging.

Primary install modes:

- internal developers: private repository clone + `conda` environment +
  editable install
- external developers: public mirror clone or published `wheel` / `sdist`

Recommended environment baseline:

- Python `3.11`
- Conda environment name: `lfptp`

### 3.2 Desktop distribution

Desktop releases are intended for users who only need the GUI application.

Desktop artifacts are built in the private repository and then published
through the public mirror release page. The build pipeline may change over
time, but the current strategy is:

- build native installers in the private repository on each target platform
- derive desktop artifact versioning from the same Git tag lineage used by
  `hatch-vcs`
- keep the internal product name stable as `LFP-TensorPipe`, while publishing
  versioned artifact filenames and app metadata
- validate the installers in clean environments
- publish only the final installers to the public mirror

Current runtime boundary:

- MATLAB is not bundled
- MATLAB Engine for Python is not bundled
- Lead-DBS is not bundled

Users who need Localize workflows must still configure those dependencies on
their own machines.

## 4. Versioning and Tagging

- The private repository tag is created first.
- The public mirror tag must match the private tag exactly.
- Build artifacts published in the public mirror must come from the matching
  private tag.
- Version derivation remains controlled by `hatch-vcs` in the private
  repository.

## 5. Release Artifact Matrix

| Release line | Built in | Published in | Typical artifacts |
|---|---|---|---|
| Developer | Private repo | Public mirror release | `wheel`, `sdist`, public docs |
| Desktop | Private repo | Public mirror release | macOS installer, Windows installer, release notes |

## 6. Public Export Policy

Public publication must use an export step from the private repository. Do not
hand-copy files into the public mirror.

The export step must be:

- manifest-driven
- repeatable
- reviewable
- narrow by default

The export step should include only approved files and should keep shared root
docs identical across the private and public repositories when they are part of
the approved snapshot, especially `README.md` and `docs/INSTALL.md`.

Public-only workflow remapping is acceptable when it keeps the public
automation lighter than the private automation, as long as shared root docs are
not rewritten.

## 7. Never Export

The public mirror must not receive:

- internal runbooks
- private delivery tracking docs
- private decision logs
- local machine paths, secrets, or signing assets
- unpublished experimental files
- build caches and local artifacts
- private mirror-management helpers whose only job is to copy from the private
  repository into the public clone

## 8. Release Flow Summary

1. Develop and validate in the private repository.
2. Build developer and desktop artifacts in the private repository.
3. Export an approved public snapshot from the private repository.
4. Copy the snapshot into the public mirror repository.
5. Create the matching public tag and GitHub release.
6. Upload developer and desktop artifacts to the public release.
