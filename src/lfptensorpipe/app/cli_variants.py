"""Execute named sensitivity variants in independent, ordinary record layouts."""

from copy import deepcopy
from importlib import metadata
from pathlib import Path
import platform
import re
import shutil
import subprocess

from lfptensorpipe.cli import read_page, resolve_references
from lfptensorpipe.io.atomic_file import write_json_atomically
from .cli import ConfigError, RunFailed, run_page
from .cli_epochs import selection_reference
from .cli_export import export_page
from .cli_preprocess import STEPS, recipe_steps, run_preprocess, step_entry
from .path_resolver import PathResolver, RecordContext
from .preproc.lineage import preproc_step_lineage_is_current
from .preproc.paths import preproc_step_log_path, preproc_step_raw_path
from .shared.generation_lineage import accepted_result_generation_id
from .shared.runlog_store import read_run_log, read_ui_state

PAGES = ("preprocess", "tensor", "alignment", "features")


def merged(base, changes):
    result = deepcopy(base)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merged(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def config_changes(base, variant, path=""):
    differences = {}
    for key in sorted(base.keys() | variant.keys()):
        label = f"{path}.{key}" if path else key
        a, b = base.get(key), variant.get(key)
        if a == b:
            continue
        if isinstance(a, dict) and isinstance(b, dict):
            differences.update(config_changes(a, b, label))
        else:
            differences[label] = {"baseline": a, "variant": b}
    return differences


def _pages(plan, parent, context, trial):
    base = plan.get("base")
    if (
        not isinstance(base, dict)
        or set(base) - set(PAGES) - {"epoch_selection"}
        or set(PAGES) - set(base)
    ):
        raise ConfigError(
            "The plan base needs preprocess, tensor, alignment and features configuration files."
        )
    pages, versions = {}, {}
    for key in PAGES:
        if not isinstance(base[key], str):
            raise ConfigError(f"base.{key} must be a page JSON path.")
        page, node, version = read_page((parent / base[key]).resolve())
        if page != key:
            raise ConfigError(f"Expected a {key} configuration, got {page}.")
        pages[key] = node
        versions[key] = version
    if "epoch_selection" in base:
        pages["alignment"]["epoch_selection"] = base["epoch_selection"]
        resolve_references("alignment", pages["alignment"], parent)
    if "epoch_selection" in pages["alignment"]:
        pages["alignment"]["epoch_selection"] = selection_reference(
            pages["alignment"]["epoch_selection"], context, trial
        )
    return pages, versions


def variants(plan, parent, context, trial):
    pages, versions = _pages(plan, parent, context, trial)
    items = plan.get("variants")
    if (
        not isinstance(items, list)
        or not items
        or not isinstance(items[0], dict)
        or items[0].get("overrides")
    ):
        raise ConfigError(
            "The first variant must use the baseline settings without overrides."
        )
    seen = set()
    result = []
    for item in items:
        name = item.get("name") if isinstance(item, dict) else None
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name)
            or name.lower() in seen | {"inputs"}
        ):
            raise ConfigError(f"Invalid or duplicate variant name: {name!r}")
        seen.add(name.lower())
        repeats = item.get("repeats", 1)
        overrides = deepcopy(item.get("overrides", {}))
        if (
            type(repeats) is not int
            or repeats < 1
            or not isinstance(overrides, dict)
            or set(overrides) - set(PAGES)
        ):
            raise ConfigError(f"Invalid repeats or overrides for {name}.")
        for page, changes in overrides.items():
            if not isinstance(changes, dict):
                raise ConfigError(f"{name}.{page} overrides must be an object.")
            resolve_references(page, changes, parent)
        configs = merged(pages, overrides)
        annotation_change = overrides.get("preprocess", {}).get("annotations", {})
        if isinstance(annotation_change, dict):
            for selected, alternate in (("csv", "rows"), ("rows", "csv")):
                if selected in annotation_change and alternate not in annotation_change:
                    configs["preprocess"]["annotations"].pop(alternate, None)
        error = None
        try:
            configs["preprocess"] = recipe_steps(configs["preprocess"])
            if "epoch_selection" in configs["alignment"]:
                configs["alignment"]["epoch_selection"] = selection_reference(
                    configs["alignment"]["epoch_selection"], context, trial
                )
        except (ValueError, OSError, RunFailed) as exc:
            error = str(exc)
        result.append((name, repeats, configs, error))
    return result, versions


def _next_directory(parent, prefix):
    parent.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        path = parent / f"{prefix}-{index:02d}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            index += 1


def _copy_raw(source, target):
    from .preproc.steps.raw import _copy_raw_fif_set

    dst = preproc_step_raw_path(target, "raw")
    _copy_raw_fif_set(preproc_step_raw_path(source, "raw", create=False), dst)
    shutil.copy2(
        preproc_step_log_path(source, "raw", create=False),
        preproc_step_log_path(target, "raw"),
    )


def _copy_localize(source, target):
    from .localize.paths import (
        TABLE_STEMS,
        USER_LABELS,
        applied_source,
        localize_indicator_state,
        localize_log_path,
        source_directory,
        source_results,
    )

    ctx = source.context
    args = ctx.project_root, ctx.subject, ctx.record
    if localize_indicator_state(*args) != "green":
        return None
    log = localize_log_path(*args)
    payload = read_run_log(log)
    active = applied_source(payload)
    src = source_directory(log.parent, active)
    dest = source_directory(target.lfp_root / "localize", active)
    dest.mkdir(parents=True, exist_ok=True)
    for stem in TABLE_STEMS:
        for ext in (".pkl", ".csv"):
            shutil.copy2(src / (stem + ext), dest / (stem + ext))
    if active == USER_LABELS:
        from .localize.user_labels import UserLabels, write_labels_csv

        accepted = source_results(payload)[active]["label_signature"]
        labels = UserLabels(
            tuple(accepted["regions"]),
            {ch: set(names) for ch, names in accepted["membership"].items()},
            accepted["assignment_method"],
        )
        write_labels_csv(dest / "input.csv", labels, tuple(accepted["channels"]))
    shutil.copy2(log, target.lfp_root / "localize" / log.name)
    current = read_run_log(log)
    if (
        applied_source(current) != active
        or source_results(current).get(active) != source_results(payload)[active]
    ):
        raise RunFailed("The applied localization changed during the snapshot.")
    return {
        "source": active,
        "result_id": accepted_result_generation_id(
            {"completed": True, "params": source_results(payload)[active]}
        ),
    }


def software_identity():
    import sys

    packages = {}
    for name in (
        "lfp-tensorpipe",
        "numpy",
        "scipy",
        "mne",
        "pandas",
        "mne-connectivity",
        "specparam",
        "fooof",
        "scikit-learn",
    ):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    result = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
    }
    root = Path(__file__).resolve().parents[3]
    if (root / ".git").exists() and shutil.which("git"):
        result["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        result["working_tree_modified"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--", "src/lfptensorpipe"],
                cwd=root,
                text=True,
            ).strip()
        )
    return result


def _reuse_prefix(source, target, base, selected):
    reused = []
    for step in STEPS:
        if base[step] != selected[step]:
            break
        if base[step]["action"] == "apply" and not preproc_step_lineage_is_current(
            source, step
        ):
            break
        folder = source.preproc_step_dir(step)
        if folder.is_dir():
            shutil.copytree(folder, target.preproc_step_dir(step))
        reused.append(step)
    if len(reused) == len(STEPS) and preproc_step_lineage_is_current(source, "finish"):
        shutil.copytree(
            source.preproc_step_dir("finish"), target.preproc_step_dir("finish")
        )
    return reused


def _reuse_tensor_metrics(source, target, snapshot):
    from .tensor.lineage import (
        capture_tensor_input_generation,
        tensor_metric_lineage_is_current,
    )
    from .tensor.paths import tensor_output_metric_keys

    finish = capture_tensor_input_generation(target)
    if finish is None or finish != capture_tensor_input_generation(source):
        return set()
    prior = read_ui_state(source.record_ui_state_path())["tensor"]
    if prior["mask_edge_effects"] != snapshot["mask_edge_effects"]:
        return set()
    keep = set()
    for metric in snapshot["selected_metrics"]:
        if (
            metric not in prior["selected_metrics"]
            or prior["metric_params"][metric] != snapshot["metric_params"][metric]
            or not tensor_metric_lineage_is_current(source, metric)
        ):
            continue
        for key in tensor_output_metric_keys([metric]):
            shutil.copytree(
                source.tensor_metric_dir(key), target.tensor_metric_dir(key)
            )
        # Burst also reads the baseline annotations specific to that metric.
        if tensor_metric_lineage_is_current(target, metric):
            keep.add(metric)
    return keep


def _adopt_alignment(source, target, trial, align, baseline, metrics):
    """Reuse a checked warp and apply a new Include selection when it changed."""
    from .alignment import service as svc
    from .alignment.generation import (
        latest_alignment_step_entry,
        capture_alignment_run_input_generations,
    )
    from .cli import _checked_result, _select_trial_state, _trial_config
    from .cli_epochs import accepted_selection, epoch_match_mode, replay_picks
    from .shared.generation_lineage import input_generation_receipts_match
    from .tensor.paths import tensor_output_metric_keys

    if any(align[k] != baseline[k] for k in ("method", "method_params")):
        return ()
    prior = latest_alignment_step_entry(
        source, trial_slug=trial, step="run_align_epochs"
    )
    outputs = tensor_output_metric_keys(metrics)
    if not prior or set(prior[1]["params"]["metrics"]) != set(outputs):
        return ()
    expected = capture_alignment_run_input_generations(
        target,
        metrics=outputs,
        method=align["method"],
        method_params=align["method_params"],
    )
    if not expected or not input_generation_receipts_match(prior[1], expected=expected):
        return ()
    old = accepted_selection(source.context, trial)
    reference = align.get("epoch_selection")
    mode = epoch_match_mode(align, reference)
    shutil.copytree(
        source.alignment_paradigm_dir(trial), target.alignment_paradigm_dir(trial)
    )
    ctx = target.context
    config = _trial_config(ctx, trial, allow_missing=False)
    if reference is None:
        picks = list(range(len(old["decisions"])))
    else:
        picks = replay_picks(ctx, trial, reference, mode)
    _select_trial_state(ctx, config, picks, submitted=True)
    old_picks = [i for i, r in enumerate(old["decisions"]) if r["include"]]
    if picks == old_picks and (
        reference is None or reference["decisions"] == old["decisions"]
    ):
        return ("run_align_epochs", "build_raw_table")
    if not svc.persist_alignment_epoch_picks(
        ctx, paradigm_slug=trial, picked_epoch_indices=picks
    ):
        raise RunFailed("Could not save the variant's epoch selection.")
    _checked_result(
        svc.finish_alignment_epochs(
            ctx, paradigm_slug=trial, picked_epoch_indices=picks
        )
    )
    return ("run_align_epochs",)


def _eligible_tensor(node, resolver):
    from .tensor.params import TENSOR_CHANNEL_SELECTOR_KEYS
    from .tensor.selectors import load_tensor_channel_inventory

    inventory = load_tensor_channel_inventory(
        preproc_step_raw_path(resolver, "finish", create=False)
    )
    bad = set(inventory.bad_channels)
    node = deepcopy(node)
    losses = {}
    if not isinstance(node.get("selected_metrics"), list) or not isinstance(
        node.get("metric_params"), dict
    ):
        raise ConfigError(
            "The Tensor page requires selected_metrics and metric_params."
        )
    for metric in list(node["selected_metrics"]):
        field = (
            "selected_channels"
            if metric in TENSOR_CHANNEL_SELECTOR_KEYS
            else "selected_pairs"
        )
        params = node["metric_params"].get(metric)
        choices = params.get(field) if isinstance(params, dict) else None
        if not isinstance(choices, list):
            continue

        def lost(choice):
            endpoints = [choice] if field == "selected_channels" else choice
            return (
                isinstance(endpoints, list)
                and all(
                    isinstance(ch, str) and ch in inventory.all_channels
                    for ch in endpoints
                )
                and any(ch in bad for ch in endpoints)
            )

        removed = [choice for choice in choices if lost(choice)]
        if removed:
            losses[metric] = {
                "reason": "Marked bad in the variant Preprocess Finish.",
                "excluded": removed,
            }
            params[field] = [choice for choice in choices if choice not in removed]
            if not params[field]:
                node["selected_metrics"].remove(metric)
    return node, losses


def _record_run(
    run_dir,
    frozen,
    trial,
    configs,
    versions,
    inputs,
    software,
    base,
    reference,
    *,
    reuse,
    sensitivity,
    differences,
    configuration_error,
    store,
):
    from .alignment.generation import latest_alignment_step_entry
    from .alignment.paths import alignment_paradigm_log_path
    from .features.indicator import features_derivatives_log_path
    from .tensor.paths import tensor_metric_log_path

    context = RecordContext(run_dir, frozen.context.subject, frozen.context.record)
    resolver = PathResolver(context)
    manifest = {
        "status": "running",
        "subject": context.subject,
        "record": context.record,
        "trial": trial,
        "input": inputs,
        "software": software,
        "settings": deepcopy(configs),
        "differences": differences,
        "page_versions": versions,
        "stages": {},
    }
    path = run_dir / "run.json"

    def save():
        write_json_atomically(path, manifest)

    def receipt(step, status, log=None, entry=None):
        info = {"status": status}
        if status == "reused" and reference is not None:
            info["source_run"] = str(reference.context.project_root)
        if log and log.is_file():
            info.update(
                log=str(log.relative_to(run_dir)),
                generation=accepted_result_generation_id(entry or read_run_log(log)),
            )
        manifest["stages"][step] = info
        save()

    def preproc_receipt(step, status):
        nonlocal stage
        if status == "running":
            stage = step
        receipt(
            step,
            status,
            (
                preproc_step_log_path(resolver, step, create=False)
                if status != "skipped"
                else None
            ),
        )

    save()
    stage = "configuration"
    try:
        if configuration_error:
            raise ConfigError(configuration_error)
        stage = "raw"
        _copy_raw(frozen, resolver)
        _copy_localize(frozen, resolver)
        preproc_receipt("raw", "frozen_input")
        reused = (
            _reuse_prefix(
                reference, resolver, base["preprocess"], configs["preprocess"]
            )
            if reuse
            else []
        )
        for step in reused:
            preproc_receipt(
                step,
                (
                    "skipped"
                    if configs["preprocess"][step]["action"] == "skip"
                    else "reused"
                ),
            )
        stage = "preprocess"
        run_preprocess(
            context, configs["preprocess"], reused=reused, report=preproc_receipt
        )
        # Export effective (including defaulted) decisions as actually executed.
        manifest["effective"] = {
            "preprocess": export_page(context, "preprocess")["preprocess"]
        }
        stage = "tensor"
        tensor_node = configs[stage]
        if sensitivity:
            tensor_node, losses = _eligible_tensor(tensor_node, resolver)
            manifest["excluded_selections"] = losses
            save()
            if not tensor_node["selected_metrics"]:
                raise RunFailed(
                    "No selected Tensor metrics retain an eligible channel or pair."
                )
        from .cli import prepare_tensor_config, execute_tensor_metrics

        actual = prepare_tensor_config(context, tensor_node, versions[stage])
        retained_metrics = (
            _reuse_tensor_metrics(reference, resolver, actual) if reuse else set()
        )
        execute_tensor_metrics(
            context,
            actual,
            [key for key in actual["selected_metrics"] if key not in retained_metrics],
        )
        metrics = actual["selected_metrics"]
        manifest["effective"][stage] = {
            k: actual[k]
            for k in ("selected_metrics", "cpu_percent", "mask_edge_effects")
        }
        manifest["effective"][stage]["metric_params"] = {
            m: actual["metric_params"][m] for m in metrics
        }
        for metric in metrics:
            receipt(
                "tensor/" + metric,
                "reused" if metric in retained_metrics else "computed",
                tensor_metric_log_path(resolver, metric),
            )
        stage = "alignment"
        align = dict(configs[stage])
        retained_alignment = (
            _adopt_alignment(
                reference, resolver, trial, align, base["alignment"], metrics
            )
            if reuse and len(retained_metrics) == len(metrics)
            else ()
        )
        if not retained_alignment:
            run_page(
                context,
                page=stage,
                node=align,
                version=versions[stage],
                trial=trial,
                config_store=store,
            )
        for step in ("run_align_epochs", "build_raw_table"):
            entry = latest_alignment_step_entry(resolver, trial_slug=trial, step=step)[
                1
            ]
            receipt(
                step,
                "reused" if step in retained_alignment else "computed",
                alignment_paradigm_log_path(resolver, trial),
                entry,
            )
        manifest["effective"][stage] = export_page(context, "alignment", trial)[stage]
        stage = "features"
        prior_features = None
        if "build_raw_table" in retained_alignment:
            shutil.copytree(
                reference.features_root / trial, resolver.features_root / trial
            )
            prior_features = accepted_result_generation_id(
                read_run_log(features_derivatives_log_path(reference, trial_slug=trial))
            )
        features_node = configs[stage]
        if manifest.get("excluded_selections") and "execution" in features_node:
            from .alignment.generation import accepted_alignment_metrics

            remaining = accepted_alignment_metrics(
                resolver, trial_slug=trial, stage="finish"
            )
            features_node = deepcopy(features_node)
            definitions = features_node["execution"]["metrics"]
            features_node["execution"]["metrics"] = {
                m: definitions[m] for m in remaining if m in definitions
            }
        keep_features = False
        if prior_features:
            from .cli import _patch_state, prepare_features_config
            from .features.execution import (
                execution_snapshot,
                normalized_execution_metadata,
            )

            seed = export_page(context, stage, trial)[stage]
            keys = list(seed["axes_by_metric"])
            requested = prepare_features_config(
                context, features_node, keys, trial=trial
            )
            numerical = normalized_execution_metadata(features_node.get("execution"))
            if numerical is None:
                numerical = execution_snapshot(context, trial, keys, store)
            keep_features = (
                requested["axes_by_metric"] == seed["axes_by_metric"]
                and numerical == seed["execution"]
            )
            if keep_features:
                _patch_state(
                    context, ("features", "trial_params_by_slug", trial), requested
                )
        if not keep_features:
            run_page(
                context,
                page=stage,
                node=features_node,
                version=versions[stage],
                trial=trial,
                config_store=store,
            )
        log = features_derivatives_log_path(resolver, trial_slug=trial)
        accepted = read_run_log(log)
        receipt(
            stage,
            (
                "reused"
                if prior_features
                and prior_features == accepted_result_generation_id(accepted)
                else "computed"
            ),
            log,
            accepted,
        )
        if keep_features:
            manifest["stages"][stage]["output_work"] = {
                "retained": [
                    p
                    for paths in accepted["params"]["outputs_by_metric"].values()
                    for p in paths
                ]
            }
        else:
            manifest["stages"][stage]["output_work"] = accepted["params"]["output_work"]
        manifest["effective"][stage] = export_page(context, stage, trial)[stage]
        manifest["status"] = "completed"
    except (OSError, ValueError, RuntimeError) as exc:
        manifest["stages"][stage] = {"status": "failed", "error": str(exc)}
        manifest.update(status="failed", failed_stage=stage, error=str(exc))
    except KeyboardInterrupt:
        manifest.update(status="interrupted", failed_stage=stage)
        raise
    finally:
        save()
    return resolver, manifest


def run_plan(context, trial, plan, parent, root):
    """Freeze this invocation's inputs, then run each repetition and retain failures."""
    selections, versions = variants(plan, parent, context, trial)
    resolver = PathResolver(context)
    if root == context.project_root or any(
        root.is_relative_to(context.project_root / key)
        for key in ("rawdata", "derivatives")
    ):
        raise ConfigError(
            "Choose an output root outside the source data and derivatives."
        )
    for name in ["inputs", *(name for name, _, _, _ in selections)]:
        if (root / name).resolve().parent != root:
            raise ConfigError(f"Output directory leaves the experiment root: {name}")
    if not preproc_step_lineage_is_current(resolver, "raw"):
        raise RunFailed("The experiment requires a current accepted Raw.")
    location = _next_directory(root / "inputs", "snapshot")
    frozen = PathResolver(RecordContext(location, context.subject, context.record))
    original = accepted_result_generation_id(step_entry(resolver, "raw"))
    _copy_raw(resolver, frozen)
    localization = _copy_localize(resolver, frozen)
    if original != accepted_result_generation_id(step_entry(resolver, "raw")):
        raise RunFailed("The source Raw changed during the snapshot.")
    inputs = {
        "project": str(context.project_root),
        "subject": context.subject,
        "record": context.record,
        "raw_generation": original,
        "localization": localization,
        "snapshot": str(location),
    }
    import mne
    from .preproc.review_edits import current_state
    from .config_store import AppConfigStore

    with mne.io.read_raw_fif(
        preproc_step_raw_path(frozen, "raw"), preload=False, verbose="ERROR"
    ) as raw:
        inputs["raw"] = {
            "sample_rate_hz": float(raw.info["sfreq"]),
            "samples": int(raw.n_times),
            "channels": raw.ch_names,
            **current_state(raw),
        }
    # Read the local numerical defaults once for legacy Features page files.
    store = AppConfigStore(location / "settings")
    if any("execution" not in nodes["features"] for _, _, nodes, _ in selections):
        store.write_yaml("features.yml", AppConfigStore().read_yaml("features.yml", {}))
    write_json_atomically(location / "input.json", inputs)
    software = software_identity()
    base = selections[0][2]
    reference = None
    reference_features = None
    decisions = None
    paths = []
    failures = []
    for index, (name, count, configs, error) in enumerate(selections):
        previous_selection = None
        for repetition in range(count):
            settings = deepcopy(configs)
            changed = configs["preprocess"] != base["preprocess"]
            align = settings["alignment"]
            if previous_selection and "epoch_match" not in align:
                align["epoch_selection"] = previous_selection
            elif decisions and "epoch_selection" not in align:
                align["epoch_selection"] = decisions
            if (
                reference_features
                and "execution" not in settings["features"]
                and settings["tensor"] == base["tensor"]
            ):
                settings["features"]["execution"] = reference_features
            if "epoch_selection" in align and "epoch_match" not in align:
                align["epoch_match"] = (
                    "common"
                    if index and changed and not previous_selection
                    else "exact"
                )
            directory = _next_directory(root / name, "run")
            use_prefix = bool(
                index and not repetition and configs != base and reference
            )
            child, manifest = _record_run(
                directory,
                frozen,
                trial,
                settings,
                versions,
                inputs,
                software,
                base,
                reference,
                reuse=use_prefix,
                sensitivity=bool(index and changed),
                differences=config_changes(base, configs),
                configuration_error=error,
                store=store,
            )
            paths.append(directory)
            if manifest["status"] != "completed":
                failures.append(
                    f"{directory}: {manifest['failed_stage']}: {manifest['error']}"
                )
            else:
                previous_selection = manifest["effective"]["alignment"][
                    "epoch_selection"
                ]
                if not index and reference is None:
                    reference = child
                    decisions = previous_selection
                    reference_features = manifest["effective"]["features"]["execution"]
    if failures:
        raise RunFailed(
            "Some runs failed. Inspect their run.json files:\n" + "\n".join(failures)
        )
    return paths
