"""Portable epoch-review rules and their Boolean expression."""

from __future__ import annotations

import json
import math
import re
from string import ascii_uppercase

SIGNAL = "preprocessed_signal"
LETTERS = ascii_uppercase


def rule_defaults(metric=SIGNAL, method="MAD"):
    rule = {
        "metric": metric,
        "channels": "All",
        "match": "Any",
        "time": "All",
        "frequency": "All",
        "method": method,
    }
    if method == "Valid coverage":
        rule["coverage"] = 80.0
    elif metric == SIGNAL:
        rule.update(threshold=6.0, minimum_epochs=3, baseline=0.0, duration=None)
    else:
        rule.update(threshold=4.0, minimum_epochs=3, area=5.0)
    return rule


def factory_scheme():
    return {
        "rules": [rule_defaults(), rule_defaults(SIGNAL, "Valid coverage")],
        "expression": "A OR B",
    }


def tokens(expression):
    return re.findall(r"[A-Za-z]+|[^\s]", expression)


def expression_program(expression, count):
    """Compile a restricted expression to postfix after checking its grammar."""
    if not isinstance(expression, str):
        raise ValueError("Enter a final expression.")
    operators, program = [], []
    need_operand = True
    for token in tokens(expression):
        if need_operand and token == "(":
            operators.append(token)
        elif need_operand and len(token) == 1 and token in LETTERS[:count]:
            program.append(token)
            need_operand = False
        elif not need_operand and token in {"AND", "OR"}:
            while (
                operators
                and operators[-1] != "("
                and (token == "OR" or operators[-1] == "AND")
            ):
                program.append(operators.pop())
            operators.append(token)
            need_operand = True
        elif not need_operand and token == ")":
            while operators and operators[-1] != "(":
                program.append(operators.pop())
            if not operators:
                raise ValueError("Unmatched closing parenthesis.")
            operators.pop()
        else:
            raise ValueError(f"Unexpected token in final expression: {token}")
    if need_operand or "(" in operators:
        raise ValueError("The final expression is incomplete.")
    return program + operators[::-1]


def logic(operator, values):
    decisive = operator == "OR"
    if decisive in values:
        return decisive
    return None if None in values else not decisive


def evaluate(program, scores):
    stack = []
    for token in program:
        if token in {"AND", "OR"}:
            stack.append(logic(token, [stack.pop(), stack.pop()]))
        else:
            stack.append(scores[token])
    return stack[0]


def _number(x, name, minimum=None, maximum=None):
    if type(x) not in (float, int) or not math.isfinite(x):
        raise ValueError(f"{name} must be a finite number.")
    if (minimum is not None and x < minimum) or (maximum is not None and x > maximum):
        raise ValueError(f"{name} is outside its allowed range.")
    return float(x)


def _bounds(value, name):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} needs a lower and upper bound.")
    lo, hi = [_number(x, name) for x in value]
    if lo >= hi:
        raise ValueError(f"{name} bounds must increase.")
    return [lo, hi]


def normalize_scheme(config):
    """Validate a user file, default, or editor draft at its input boundary."""
    if not isinstance(config, dict) or not isinstance(config.get("rules"), list):
        raise ValueError("Expected an Epoch Inspector QC configuration.")
    if not 1 <= len(config["rules"]) <= 26:
        raise ValueError("Configure between 1 and 26 rules.")
    expression = config.get("expression")
    expression_program(expression, len(config["rules"]))
    rules = []
    for item in config["rules"]:
        if not isinstance(item, dict):
            raise ValueError("Each rule must be an object.")
        metric, method = item.get("metric"), item.get("method")
        if (
            not isinstance(metric, str)
            or not metric
            or method not in ("MAD", "Valid coverage")
        ):
            raise ValueError("Unknown QC metric or method.")
        rule = rule_defaults(metric, method)
        if set(item) - set(rule) - {"axis"} or set(rule) - set(item):
            raise ValueError("Missing or unsupported QC rule fields.")
        channels = item["channels"]
        if channels != "All":
            if (
                not isinstance(channels, list)
                or not channels
                or not all(isinstance(x, str) and x for x in channels)
            ):
                raise ValueError("Select All or at least one channel.")
            rule["channels"] = sorted(set(channels))
        if item["match"] not in ("All", "Any"):
            raise ValueError("Channel match must be All or Any.")
        rule["match"] = item["match"]
        if item["time"] != "All":
            rule["time"] = _bounds(item["time"], "Time window")
        freq = item["frequency"]
        if freq != "All":
            if metric == SIGNAL or not isinstance(freq, dict):
                raise ValueError("Invalid frequency selection for this metric.")
            if set(freq) == {"label"} and isinstance(freq["label"], str):
                rule["frequency"] = {"label": freq["label"]}
            elif set(freq) == {"label", "segments_hz"} and isinstance(
                freq["label"], str
            ):
                segments = freq["segments_hz"]
                if not isinstance(segments, list) or not segments:
                    raise ValueError("A named band needs its frequency segments.")
                rule["frequency"] = {
                    "label": freq["label"],
                    "segments_hz": [
                        _bounds(pair, "Frequency segment") for pair in segments
                    ],
                }
            elif set(freq) in ({"hz"}, {"hz", "label"}) and isinstance(
                freq.get("label", ""), str
            ):
                rule["frequency"] = {**freq, "hz": _bounds(freq["hz"], "Frequency")}
            else:
                raise ValueError("Specify a frequency band or bin.")
        if method == "Valid coverage":
            rule["coverage"] = _number(item["coverage"], "Minimum coverage", 0, 100)
        else:
            rule["threshold"] = _number(item["threshold"], "MAD threshold", 0)
            if type(item["minimum_epochs"]) is not int or item["minimum_epochs"] < 1:
                raise ValueError("Minimum valid epochs must be a positive integer.")
            rule["minimum_epochs"] = item["minimum_epochs"]
            if metric == SIGNAL:
                rule["baseline"] = _number(item["baseline"], "Baseline window", 0)
                if item["duration"] is not None:
                    rule["duration"] = _number(
                        item["duration"], "Minimum excursion duration", 0
                    )
            else:
                rule["area"] = _number(item["area"], "Minimum cluster area", 0, 100)
        if rule["time"] != "All" or rule.get("baseline") or rule.get("duration"):
            if item.get("axis") not in ("epoch_seconds", "cue_seconds", "percent"):
                raise ValueError(
                    "Numeric time settings require their unit and zero anchor."
                )
            rule["axis"] = item["axis"]
        rules.append(rule)
    return {"rules": rules, "expression": " ".join(tokens(expression))}


def epoch_start_rule(rule, pre_cue_s):
    """Translate positions in a historical cue-relative rule, retaining lengths."""
    if pre_cue_s is None or rule.get("axis") != "cue_seconds":
        return rule
    converted = {**rule, "axis": "epoch_seconds"}
    if rule["time"] != "All":
        converted["time"] = [t + pre_cue_s for t in rule["time"]]
    return converted


def epoch_start_scheme(config, pre_cue_s):
    if not config or pre_cue_s is None:
        return config
    return {
        **config,
        "rules": [epoch_start_rule(r, pre_cue_s) for r in config["rules"]],
    }


def effective_program(config):
    """Resolve letter references to rule values, independent of unused rules."""
    stack = []
    for token in expression_program(config["expression"], len(config["rules"])):
        if token not in {"AND", "OR"}:
            rule = dict(config["rules"][LETTERS.index(token)])
            freq = rule["frequency"]
            if isinstance(freq, dict) and "hz" in freq:
                rule["frequency"] = {"hz": freq["hz"]}
            if isinstance(rule["channels"], list) and len(rule["channels"]) == 1:
                rule["match"] = "Any"
            if rule.get("duration") == 0:
                rule["duration"] = None
            stack.append(("rule", json.dumps(rule, sort_keys=True)))
            continue
        right, left = stack.pop(), stack.pop()
        terms = set()
        for node in (left, right):
            terms.update(node[1] if node[0] == token else (node,))
        stack.append(
            next(iter(terms)) if len(terms) == 1 else (token, tuple(sorted(terms)))
        )
    return stack[0]
