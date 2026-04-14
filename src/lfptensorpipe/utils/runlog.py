from __future__ import annotations

import io
import os
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from functools import wraps
from typing import Any, Callable
import json
from pathlib import Path


def with_run_log(
    *,
    capture_stdout: bool = True,
    capture_stderr: bool = True,
    include_traceback: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., dict[str, Any]]]:
    """Decorator that wraps a function and returns a structured run log.

    The wrapped function's return value is stored under log["result"].
    Any printed output is captured under log["stdout"] / log["stderr"].

    This is parallel-safe and avoids interleaving prints from multiple workers.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., dict[str, Any]]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
            t0 = time.perf_counter()
            pid = os.getpid()

            stdout_buf = io.StringIO() if capture_stdout else None
            stderr_buf = io.StringIO() if capture_stderr else None

            log: dict[str, Any] = {
                "ok": True,
                "pid": pid,
                "func": func.__name__,
                "args_repr": repr(args)[:5000],
                "kwargs_repr": repr(kwargs)[:5000],
                "result": None,
                "stdout": "",
                "stderr": "",
                "error": None,
                "duration_sec": None,
            }

            try:
                if capture_stdout and capture_stderr:
                    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                        log["result"] = func(*args, **kwargs)
                elif capture_stdout:
                    with redirect_stdout(stdout_buf):
                        log["result"] = func(*args, **kwargs)
                elif capture_stderr:
                    with redirect_stderr(stderr_buf):
                        log["result"] = func(*args, **kwargs)
                else:
                    log["result"] = func(*args, **kwargs)

            except Exception as e:
                log["ok"] = False
                log["error"] = {
                    "type": type(e).__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc(limit=50) if include_traceback else None,
                }

            finally:
                if stdout_buf is not None:
                    log["stdout"] = stdout_buf.getvalue()
                if stderr_buf is not None:
                    log["stderr"] = stderr_buf.getvalue()
                log["duration_sec"] = float(time.perf_counter() - t0)

            return log

        return wrapper
    return decorator


def save_logs(logs: list[dict[str, Any]], out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _sanitize(obj: Any) -> Any:
        try:
            json.dumps(obj)
            return obj
        except TypeError:
            return repr(obj)

    with out_path.open("w", encoding="utf-8") as f:
        for log in logs:
            safe = {k: _sanitize(v) for k, v in log.items()}
            f.write(json.dumps(safe, ensure_ascii=False) + "\n")
    return out_path
