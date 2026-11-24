"""Portable numerical settings for a Features page execution."""

from pathlib import Path

from lfptensorpipe.app.alignment.paths import alignment_trial_raw_table_path
from lfptensorpipe.app.path_resolver import PathResolver
from lfptensorpipe.io.pkl_io import load_pkl
from lfptensorpipe.tabular.grid import GridResultColumns, ReducerKind
from lfptensorpipe.utils.transforms import transform_policy_metadata
from .derive_defaults import DEFAULT_COLLAPSE_BASE_CFG, _load_collapse_base_cfg
from .phase_reuse import collapse_settings

KINDS = ("raw", "spectral", "trace", "scalar")


def acquisition_power_unit(context, channels):
    import mne
    from mne.io.constants import FIFF
    from lfptensorpipe.app.preproc.paths import preproc_step_raw_path

    path = preproc_step_raw_path(PathResolver(context), "finish", create=False)
    if path.is_file():
        info = mne.io.read_info(path, verbose="ERROR")
        voltage = {
            ch["ch_name"] for ch in info["chs"] if ch["unit"] == FIFF.FIFF_UNIT_V
        }
        if channels and set(channels) <= voltage:
            return "V²"
    return "squared input unit"


def measurement_units(metric, reducers, *, raw_output, power_unit=None):
    from .burst_native import BURST_UNITS
    from lfptensorpipe.app.tensor.params import TENSOR_UNDIRECTED_SELECTOR_KEYS

    if metric == "raw_power":
        native = power_unit
    elif metric in TENSOR_UNDIRECTED_SELECTOR_KEYS | {"periodic", "trgc", "psi"}:
        native = "dimensionless"
    elif metric == "burst":
        native = "threshold multiple"
    else:
        native = "component-specific"
    units = {"na": native} if raw_output else {}
    overrides = (
        BURST_UNITS
        if metric == "burst"
        else {
            "count": "segments",
            "occupation": "%",
            "rate": "per reduced-axis unit",
            "duration": "reduced-axis units",
        }
    )
    units.update({r: overrides.get(r, native) for r in reducers})
    return units


def value_definition(context, trial, metric, kinds, reducers):
    from .extract_runner import _feature_storage_policy
    from lfptensorpipe.utils.transforms import get_transform_policy

    table = load_pkl(
        alignment_trial_raw_table_path(
            PathResolver(context), trial_slug=trial, metric_key=metric
        )
    )
    policy = transform_policy_metadata(_feature_storage_policy(table.attrs))
    power_unit = (
        acquisition_power_unit(context, table["Channel"].drop_duplicates().tolist())
        if metric == "raw_power"
        else None
    )
    domains = {}
    for kind in kinds:
        for reducer in (["na"] if kind == "raw" else reducers):
            domains[f"{reducer}-{kind}"] = (
                transform_policy_metadata(get_transform_policy("none"))
                if metric == "burst" and kind == "scalar" and reducer != "mean"
                else policy
            )
    return {
        "value_policy": policy,
        "units": measurement_units(
            metric, reducers, raw_output="raw" in kinds, power_unit=power_unit
        ),
        "output_domains": domains,
    }


def execution_snapshot(context, trial, metrics, store, *, accepted=None):
    from . import service as svc
    from .burst_native import normalize_burst_reducers
    from .extract_runner import _resolve_metric_request

    resolver = PathResolver(context)
    if accepted is None:
        method = svc._extract_alignment_method_from_log(resolver, trial_slug=trial)
        outputs_cfg = svc._load_derive_param_cfg(store)
        reducers_cfg = svc._load_reducer_cfg(store)
        rules = svc._load_reducer_rule_by_method(store)
        settings = collapse_settings(_load_collapse_base_cfg(store))
    else:
        if "collapse_settings" not in accepted:
            raise ValueError(
                "The accepted Features log lacks its reduction settings. Extract again to export them."
            )
        settings = accepted["collapse_settings"]
    definitions = {}
    for metric in metrics:
        if accepted is None:
            outputs, reducers = _resolve_metric_request(
                metric_key=metric,
                derive_param_cfg=outputs_cfg,
                reducer_cfg=reducers_cfg,
                reducer_rule_by_method=rules,
                alignment_method=method,
                override_outputs=None,
                reducer_override="",
            )
            kinds = [k for k in KINDS if outputs[k]]
        else:
            stems = [
                Path(p).stem.rsplit("-", 1)
                for p in accepted["outputs_by_metric"][metric]
            ]
            kinds = [k for k in KINDS if any(t == k for _, t in stems)]
            reducers = list(dict.fromkeys(r for r, k in stems if k != "raw"))
        if metric == "burst":
            reducers = normalize_burst_reducers(reducers) if reducers else []
        definitions[metric] = {
            "outputs": kinds,
            "reducers": reducers,
            **value_definition(context, trial, metric, kinds, reducers),
        }
    return {"metrics": definitions, "collapse": settings}


def execution_configs(execution, metrics):
    """Translate a frozen definition into the inputs consumed by native reducers."""
    from .burst_native import BURST_REDUCERS

    if (
        not isinstance(execution, dict)
        or not isinstance(execution.get("metrics"), dict)
        or set(execution["metrics"]) != set(metrics)
    ):
        raise ValueError("Features execution must define exactly the selected metrics.")
    options = execution.get("collapse")
    if not isinstance(options, dict) or (set(options) - {"out_cols"}) != set(
        DEFAULT_COLLAPSE_BASE_CFG
    ):
        raise ValueError("Features execution requires the complete collapse settings.")
    collapse = dict(options)
    collapse.setdefault(
        "out_cols", {"band": "Band", "phase": "Phase", "value": "Value"}
    )
    if not isinstance(collapse["out_cols"], dict) or set(collapse["out_cols"]) != {
        "band",
        "phase",
        "value",
    }:
        raise ValueError("Invalid Features output column names.")
    if any(not isinstance(s, str) or not s for s in collapse["out_cols"].values()):
        raise ValueError("Features output column names must be nonempty strings.")
    if (
        not isinstance(collapse["value_col"], str)
        or not collapse["value_col"]
        or collapse["on_missing"] not in ("raise", "skip")
    ):
        raise ValueError("Invalid Features value column or missing-value policy.")
    collapse["out_cols"] = GridResultColumns(**collapse["out_cols"])
    if any(
        collapse[k] not in ("absolute", "percent")
        for k in ("time_interval_mode", "freq_interval_mode")
    ) or any(
        type(collapse[k]) is not bool
        for k in ("inclusive", "drop_value", "keep_full_dim_cols")
    ):
        raise ValueError("Invalid Features interval or Boolean reduction settings.")
    outputs, reducers = {}, {}
    for key, definition in execution["metrics"].items():
        if not isinstance(definition, dict):
            raise ValueError(f"Invalid execution definition for {key}.")
        kinds, names = definition.get("outputs"), definition.get("reducers")
        if not isinstance(kinds, list) or any(
            not isinstance(k, str) or k not in KINDS for k in kinds
        ):
            raise ValueError(f"Invalid outputs for {key}.")
        allowed = BURST_REDUCERS if key == "burst" else {r.value for r in ReducerKind}
        if (
            not isinstance(names, list)
            or any(not isinstance(r, str) or r not in allowed for r in names)
            or (set(kinds) - {"raw"} and not names)
            or (key == "burst" and set(kinds) - {"raw", "scalar"})
        ):
            raise ValueError(f"Invalid reducers or output types for {key}.")
        if not isinstance(definition.get("value_policy"), dict) or not isinstance(
            definition.get("units"), dict
        ):
            raise ValueError(f"Invalid numerical domain or units for {key}.")
        outputs[key] = {k: k in kinds for k in KINDS}
        reducers[key] = {"default": names}
    return outputs, reducers, collapse
