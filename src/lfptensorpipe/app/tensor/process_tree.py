"""Cross-platform ownership and termination for Build Tensor descendants."""

from __future__ import annotations

from dataclasses import dataclass
import os
import signal
import subprocess
import time
from typing import Any


def build_tensor_popen_kwargs() -> dict[str, Any]:
    """Return platform launch flags for the top-level Build Tensor worker."""
    if os.name == "posix":
        return {"start_new_session": True}
    return {}


def _posix_group_has_live_members(process_group_id: int) -> bool:
    try:
        completed = subprocess.run(
            ["ps", "-o", "stat=", "-g", str(int(process_group_id))],
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        completed = None
    if completed is not None and completed.returncode in {0, 1}:
        for line in completed.stdout.splitlines():
            if not line.strip().upper().startswith("Z"):
                return True
        return False

    try:
        os.killpg(int(process_group_id), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class _WindowsJob:
    """Kill-on-close Windows Job Object owned by the GUI process."""

    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9
    _JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION_CLASS = 1

    def __init__(self, process: subprocess.Popen[Any]) -> None:
        import ctypes
        from ctypes import wintypes

        class _BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class _ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BasicLimitInformation),
                ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        class _BasicAccountingInformation(ctypes.Structure):
            _fields_ = [
                ("TotalUserTime", ctypes.c_longlong),
                ("TotalKernelTime", ctypes.c_longlong),
                ("ThisPeriodTotalUserTime", ctypes.c_longlong),
                ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
                ("TotalPageFaultCount", wintypes.DWORD),
                ("TotalProcesses", wintypes.DWORD),
                ("ActiveProcesses", wintypes.DWORD),
                ("TotalTerminatedProcesses", wintypes.DWORD),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.QueryInformationJobObject.restype = wintypes.BOOL
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.restype = wintypes.BOOL

        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        self._ctypes = ctypes
        self._kernel32 = kernel32
        self._handle = handle
        self._accounting_type = _BasicAccountingInformation
        try:
            limits = _ExtendedLimitInformation()
            limits.BasicLimitInformation.LimitFlags = (
                self._JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            )
            if not kernel32.SetInformationJobObject(
                handle,
                self._JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
                ctypes.byref(limits),
                ctypes.sizeof(limits),
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            process_handle = wintypes.HANDLE(int(process._handle))  # type: ignore[attr-defined]
            if not kernel32.AssignProcessToJobObject(handle, process_handle):
                raise ctypes.WinError(ctypes.get_last_error())
        except Exception:
            kernel32.CloseHandle(handle)
            self._handle = None
            raise

    def terminate(self) -> None:
        if self._handle is None:
            return
        if not self._kernel32.TerminateJobObject(self._handle, 1):
            raise self._ctypes.WinError(self._ctypes.get_last_error())

    def is_quiescent(self) -> bool:
        if self._handle is None:
            return True
        accounting = self._accounting_type()
        if not self._kernel32.QueryInformationJobObject(
            self._handle,
            self._JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION_CLASS,
            self._ctypes.byref(accounting),
            self._ctypes.sizeof(accounting),
            None,
        ):
            raise self._ctypes.WinError(self._ctypes.get_last_error())
        return int(accounting.ActiveProcesses) == 0

    def close(self) -> None:
        if self._handle is None:
            return
        handle = self._handle
        self._handle = None
        if not self._kernel32.CloseHandle(handle):
            raise self._ctypes.WinError(self._ctypes.get_last_error())


@dataclass
class BuildTensorProcessTree:
    """Own one Build Tensor leader and every process it creates."""

    leader_pid: int
    process_group_id: int | None = None
    windows_job: _WindowsJob | None = None

    @classmethod
    def attach(cls, process: subprocess.Popen[Any]) -> BuildTensorProcessTree:
        if os.name == "posix":
            return cls(
                leader_pid=int(process.pid),
                process_group_id=int(process.pid),
            )
        if os.name == "nt":
            return cls(
                leader_pid=int(process.pid),
                windows_job=_WindowsJob(process),
            )
        raise RuntimeError(f"Unsupported Build Tensor process platform: {os.name}")

    def terminate(self) -> None:
        if self.windows_job is not None:
            self.windows_job.terminate()
            return
        if self.process_group_id is None:
            return
        try:
            os.killpg(self.process_group_id, signal.SIGTERM)
        except ProcessLookupError:
            return

    def force_kill(self) -> None:
        if self.windows_job is not None:
            self.windows_job.terminate()
            return
        if self.process_group_id is None:
            return
        try:
            os.killpg(self.process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            return

    def is_quiescent(self) -> bool:
        if self.windows_job is not None:
            return self.windows_job.is_quiescent()
        if self.process_group_id is None:
            return True
        return not _posix_group_has_live_members(self.process_group_id)

    def wait_for_quiescence(
        self,
        timeout_s: float,
        *,
        poll_interval_s: float = 0.05,
    ) -> bool:
        deadline = time.monotonic() + max(float(timeout_s), 0.0)
        while True:
            if self.is_quiescent():
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False
            time.sleep(min(float(poll_interval_s), remaining))

    def close(self) -> None:
        if self.windows_job is not None:
            self.windows_job.close()


__all__ = [
    "BuildTensorProcessTree",
    "build_tensor_popen_kwargs",
]
