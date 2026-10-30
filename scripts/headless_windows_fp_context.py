"""Bounded, diagnostic PSS capture of FP controls from the pinned Windows client.

No manual SuspendThread/ResumeThread, context writes, remote memory reads, VA
clone, or privilege changes. The parent bounds its separate worker to five
seconds; Windows owns the transient thread-context snapshot.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

EXPECTED_SHA256 = "3fc7d8cda542beda131c44306b123f4004d07d7e22f512b46b762afc29f6edda"
CAPTURE_FLAGS = 0x180  # PSS_CAPTURE_THREADS | PSS_CAPTURE_THREAD_CONTEXT
CONTEXT_FLAGS = 0x100009  # AMD64 | CONTROL | FLOATING_POINT
MAX_THREADS = 256
THREAD_TERMINATED = 0x0001  # PSS_THREAD_FLAGS_TERMINATED in pinned ProcessSnapshot.h
DWORD, WORD, QWORD = ct.c_uint32, ct.c_uint16, ct.c_uint64
HANDLE = ct.c_void_p
ROUNDING = ("nearest-even", "toward-negative", "toward-positive", "toward-zero")


class FileTime(ct.Structure):
    _fields_ = [("low", DWORD), ("high", DWORD)]

    def integer(self) -> int:
        return int(self.low) | int(self.high) << 32


class Context(ct.Structure):
    # WinNT.h AMD64 CONTEXT. Opaque fields are deliberately never exported.
    _fields_ = [
        ("homes", QWORD * 6),
        ("flags", DWORD),
        ("mxcsr", DWORD),
        ("segments", WORD * 6),
        ("eflags", DWORD),
        ("debug", QWORD * 6),
        ("integer", QWORD * 16),
        ("rip", QWORD),
        ("fxsave", ct.c_ubyte * 512),
        ("vectors", ct.c_ubyte * 416),
        ("vector_control", QWORD),
        ("branch_control", QWORD * 5),
    ]


class ThreadEntry(ct.Structure):
    _fields_ = [
        ("exit_status", DWORD),
        ("teb", HANDLE),
        ("pid", DWORD),
        ("tid", DWORD),
        ("affinity", ct.c_size_t),
        ("priority", ct.c_int32),
        ("base_priority", ct.c_int32),
        ("syscall_arg", HANDLE),
        ("syscall_number", WORD),
        ("created", FileTime),
        ("exited", FileTime),
        ("kernel", FileTime),
        ("user", FileTime),
        ("start", HANDLE),
        ("captured", FileTime),
        ("flags", DWORD),
        ("suspend_count", WORD),
        ("context_size", WORD),
        ("context", HANDLE),
    ]


class ThreadInfo(ct.Structure):
    _fields_ = [("count", DWORD), ("context_length", DWORD)]


class ToolhelpThread(ct.Structure):
    _fields_ = [(name, DWORD) for name in ("size", "usage", "tid", "pid")]
    _fields_ += [("base_priority", ct.c_int32), ("delta_priority", ct.c_int32), ("flags", DWORD)]


class ToolhelpModule(ct.Structure):
    _fields_ = [
        ("size", DWORD),
        ("module_id", DWORD),
        ("pid", DWORD),
        ("global_count", DWORD),
        ("process_count", DWORD),
        ("base", HANDLE),
        ("image_size", DWORD),
        ("module", HANDLE),
        ("name", ct.c_wchar * 256),
        ("path", ct.c_wchar * 260),
    ]


class SnapshotError(RuntimeError):
    def __init__(self, message: str, cleanup: dict[str, int | None]) -> None:
        super().__init__(message)
        self.cleanup = cleanup


def validate_layout() -> dict[str, Any]:
    expected = {"flags": 48, "mxcsr": 52, "rip": 248, "fxsave": 256}
    if ct.sizeof(HANDLE) != 8 or ct.sizeof(Context) != 1232:
        raise ValueError("AMD64 CONTEXT size required")
    if any(getattr(Context, field).offset != offset for field, offset in expected.items()):
        raise ValueError("AMD64 CONTEXT offsets disagree with WinNT.h")
    if ct.sizeof(ThreadEntry) != 120 or ThreadEntry.context.offset != 112:
        raise ValueError("x64 PSS_THREAD_ENTRY layout required")
    if ct.sizeof(ToolhelpThread) != 28:
        raise ValueError("THREADENTRY32 layout required")
    return {
        "context_bytes": 1232,
        "context_alignment": 16,
        "offsets": expected,
        "fxsave_control": 256,
        "fxsave_status": 258,
        "fxsave_mxcsr": 280,
        "pss_thread_bytes": 120,
        "pss_context_pointer_offset": 112,
    }


def decode_context(raw: bytes, modules: list[dict[str, Any]]) -> dict[str, Any]:
    if len(raw) != 1232:
        raise ValueError("complete AMD64 CONTEXT required")
    flags, mxcsr = struct.unpack_from("<II", raw, 48)
    if flags & CONTEXT_FLAGS != CONTEXT_FLAGS:
        raise ValueError("snapshot lacks requested CONTROL/FLOATING_POINT context")
    cw, sw = struct.unpack_from("<HH", raw, 256)
    saved_mxcsr, mask = struct.unpack_from("<II", raw, 280)
    if mxcsr != saved_mxcsr or mxcsr & ~0xFFFF:
        raise ValueError("saved and top-level MXCSR differ or contain reserved upper bits")
    rip = struct.unpack_from("<Q", raw, 248)[0]
    matches = [module for module in modules if module["base"] <= rip < module["end"]]
    if len(matches) > 1:
        raise ValueError("ambiguous instruction-pointer module mapping")
    location = (
        {"module": matches[0]["name"], "rva": hex(rip - matches[0]["base"])}
        if matches
        else {"module": None, "rva": None, "reason": "outside enumerated image modules"}
    )
    return {
        "mxcsr_raw": mxcsr,
        "mxcsr_mask_raw": mask,
        "mxcsr_rounding": ROUNDING[(mxcsr >> 13) & 3],
        "mxcsr_flush_to_zero": bool(mxcsr & 0x8000),
        "mxcsr_denormals_are_zero": bool(mxcsr & 0x40),
        "mxcsr_exception_masks": (mxcsr >> 7) & 63,
        "mxcsr_exception_status": mxcsr & 63,
        "x87_control_raw": cw,
        "x87_status_raw": sw,
        "x87_rounding": ROUNDING[(cw >> 10) & 3],
        "x87_precision_bits": {0: 24, 1: "reserved", 2: 53, 3: 64}[(cw >> 8) & 3],
        "instruction_location": location,
    }


class WindowsAPI:
    # Functions are assigned below with exact ctypes argtypes/restype. Their
    # Python types remain dynamic at this native ABI boundary.
    def __getattr__(self, name: str) -> Any:
        return getattr(self.dll, name)

    def __init__(self) -> None:
        if os.name != "nt" or ct.sizeof(ct.c_wchar) != 2:
            raise RuntimeError("native 64-bit Windows Python required")
        self.dll = ct.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "OpenProcess": ([DWORD, ct.c_int, DWORD], HANDLE),
            "CloseHandle": ([HANDLE], ct.c_int),
            "GetCurrentProcess": ([], HANDLE),
            "GetProcessId": ([HANDLE], DWORD),
            "GetExitCodeProcess": ([HANDLE, ct.POINTER(DWORD)], ct.c_int),
            "GetProcessTimes": ([HANDLE] + [ct.POINTER(FileTime)] * 4, ct.c_int),
            "QueryFullProcessImageNameW": (
                [HANDLE, DWORD, ct.c_wchar_p, ct.POINTER(DWORD)],
                ct.c_int,
            ),
            "IsWow64Process2": ([HANDLE, ct.POINTER(WORD), ct.POINTER(WORD)], ct.c_int),
            "CreateToolhelp32Snapshot": ([DWORD, DWORD], HANDLE),
            "Thread32First": ([HANDLE, ct.POINTER(ToolhelpThread)], ct.c_int),
            "Thread32Next": ([HANDLE, ct.POINTER(ToolhelpThread)], ct.c_int),
            "Module32FirstW": ([HANDLE, ct.POINTER(ToolhelpModule)], ct.c_int),
            "Module32NextW": ([HANDLE, ct.POINTER(ToolhelpModule)], ct.c_int),
            "PssCaptureSnapshot": ([HANDLE, DWORD, DWORD, ct.POINTER(HANDLE)], DWORD),
            "PssQuerySnapshot": ([HANDLE, DWORD, HANDLE, DWORD], DWORD),
            "PssWalkMarkerCreate": ([HANDLE, ct.POINTER(HANDLE)], DWORD),
            "PssWalkMarkerFree": ([HANDLE], DWORD),
            "PssWalkSnapshot": ([HANDLE, DWORD, HANDLE, HANDLE, DWORD], DWORD),
            "PssFreeSnapshot": ([HANDLE, HANDLE], DWORD),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.dll, name)
            function.argtypes, function.restype = args, result
            setattr(self, name, function)
        if ct.sizeof(ToolhelpModule) != 1080 or ToolhelpModule.base.offset != 24:
            raise ValueError("x64 MODULEENTRY32W layout required")

    @staticmethod
    def check(result: int, name: str) -> None:
        if not result:
            raise OSError(ct.get_last_error(), name)

    @staticmethod
    def success(result: int, name: str) -> None:
        if result:
            raise OSError(result, name)

    def identity(self, process: int) -> dict[str, Any]:
        created, exited, kernel, user = (FileTime() for _ in range(4))
        self.check(
            self.GetProcessTimes(
                process, ct.byref(created), ct.byref(exited), ct.byref(kernel), ct.byref(user)
            ),
            "GetProcessTimes",
        )
        status = DWORD()
        self.check(self.GetExitCodeProcess(process, ct.byref(status)), "GetExitCodeProcess")
        if exited.integer() or status.value != 259:
            raise ValueError("pinned process exited")
        name, size = ct.create_unicode_buffer(32768), DWORD(32768)
        self.check(
            self.QueryFullProcessImageNameW(process, 0, name, ct.byref(size)),
            "QueryFullProcessImageNameW",
        )
        return {
            "pid": int(self.GetProcessId(process)),
            "created_filetime": created.integer(),
            "path": name.value,
        }

    def inventory(self, pid: int, *, modules: bool = False) -> list[Any]:
        snapshot = self.CreateToolhelp32Snapshot(0x18 if modules else 4, pid if modules else 0)
        if snapshot in (None, ct.c_void_p(-1).value):
            raise OSError(ct.get_last_error(), "CreateToolhelp32Snapshot")
        try:
            entry = ToolhelpModule() if modules else ToolhelpThread()
            entry.size = ct.sizeof(entry)
            first, next_item = (
                (self.Module32FirstW, self.Module32NextW)
                if modules
                else (self.Thread32First, self.Thread32Next)
            )
            rows = []
            present = first(snapshot, ct.byref(entry))
            while present:
                if entry.pid == pid:
                    rows.append(
                        {
                            "base": entry.base,
                            "end": entry.base + entry.image_size,
                            "name": entry.name,
                        }
                        if modules
                        else int(entry.tid)
                    )
                if len(rows) > (1024 if modules else MAX_THREADS):
                    raise ValueError("inventory exceeds diagnostic bound")
                present = next_item(snapshot, ct.byref(entry))
            if ct.get_last_error() != 18:  # ERROR_NO_MORE_FILES
                raise OSError(ct.get_last_error(), "Toolhelp iteration")
            return rows
        finally:
            self.CloseHandle(snapshot)


def capture_snapshot(  # noqa: PLR0913 - explicit handles, identity and mapping
    api: WindowsAPI, process: int, identity: dict[str, Any], modules: list[dict[str, Any]]
) -> dict[str, Any]:
    """Release the local snapshot even if allocation, walk or decoding fails."""
    snapshot, marker = HANDLE(), HANDLE()
    receipt: dict[str, int | None] = {"snapshot_free_code": None, "marker_free_code": None}
    rows: list[dict[str, Any]] = []
    terminated_threads: list[dict[str, Any]] = []
    seen_threads: set[int] = set()
    started = time.monotonic()
    try:
        api.success(
            api.PssCaptureSnapshot(process, CAPTURE_FLAGS, CONTEXT_FLAGS, ct.byref(snapshot)),
            "PssCaptureSnapshot",
        )
        info = ThreadInfo()
        api.success(
            api.PssQuerySnapshot(snapshot, 5, ct.byref(info), ct.sizeof(info)),
            "PssQuerySnapshot(threads)",
        )
        if not 0 < info.count <= MAX_THREADS or info.context_length < ct.sizeof(Context):
            raise ValueError("snapshot thread count/context length outside accepted bound")
        api.success(api.PssWalkMarkerCreate(None, ct.byref(marker)), "PssWalkMarkerCreate")
        while True:
            if time.monotonic() - started > 4:
                raise TimeoutError("snapshot walk budget exhausted")
            entry = ThreadEntry()
            status = api.PssWalkSnapshot(snapshot, 3, marker, ct.byref(entry), ct.sizeof(entry))
            if status == 259:  # ERROR_NO_MORE_ITEMS
                break
            api.success(status, "PssWalkSnapshot(threads)")
            if (
                entry.pid != identity["pid"]
                or entry.created.integer() < identity["created_filetime"]
            ):
                raise ValueError("snapshot thread process identity mismatch")
            if entry.tid in seen_threads or len(seen_threads) >= MAX_THREADS:
                raise ValueError("snapshot thread count is excessive or duplicated")
            seen_threads.add(entry.tid)
            if entry.flags & ~THREAD_TERMINATED:
                raise ValueError("unknown snapshot thread flags")
            if entry.flags & THREAD_TERMINATED:
                # A terminated thread has no executing FP context. Preserve its
                # descriptor explicitly; do not read a null pointer or invent FP.
                terminated_threads.append(
                    {
                        "tid": int(entry.tid),
                        "thread_flags": int(entry.flags),
                        "exit_status": int(entry.exit_status),
                        "context_bytes": int(entry.context_size),
                        "context_present": bool(entry.context),
                        "fp_context_decoded": False,
                    }
                )
                continue
            if entry.context_size < ct.sizeof(Context) or not entry.context or entry.context % 16:
                metadata = {
                    "tid": entry.tid,
                    "exit_status": entry.exit_status,
                    "thread_flags": entry.flags,
                    "context_bytes": entry.context_size,
                    "context_present": bool(entry.context),
                    "context_alignment_remainder": entry.context % 16 if entry.context else None,
                }
                raise ValueError(f"missing, truncated or unaligned AMD64 context: {metadata}")
            # This pointer belongs to the local snapshot, not target process memory.
            selected = decode_context(ct.string_at(entry.context, ct.sizeof(Context)), modules)
            rows.append(
                {
                    "tid": int(entry.tid),
                    "created_filetime": entry.created.integer(),
                    "capture_filetime": entry.captured.integer(),
                    "snapshot_suspend_count": entry.suspend_count,
                    **selected,
                }
            )
            if len(rows) > MAX_THREADS:
                raise ValueError("snapshot grew beyond accepted thread bound")
        if len(seen_threads) != info.count:
            raise ValueError("snapshot enumeration is incomplete or duplicated")
        if not rows:
            raise ValueError("snapshot contains no verified active thread contexts")
    except Exception as exc:
        # Keep this dictionary by reference: the finally block fills it before
        # the exception reaches the worker's JSON reporting boundary.
        raise SnapshotError(f"{type(exc).__name__}: {exc}", receipt) from exc
    finally:
        try:
            if marker.value:
                receipt["marker_free_code"] = int(api.PssWalkMarkerFree(marker))
        finally:
            if snapshot.value:
                receipt["snapshot_free_code"] = int(
                    api.PssFreeSnapshot(api.GetCurrentProcess(), snapshot)
                )
    if any(value != 0 for value in receipt.values()):
        raise RuntimeError(f"PSS cleanup failed: {receipt}")
    return {
        "rows": rows,
        "terminated_threads": terminated_threads,
        "enumerated_thread_count": len(seen_threads),
        "cleanup": receipt,
        "capture_walk_seconds": time.monotonic() - started,
    }


def worker(args: argparse.Namespace) -> dict[str, Any]:
    layout = validate_layout()
    api = WindowsAPI()
    # Query only. No VM_WRITE, VM_OPERATION, CREATE_THREAD, SUSPEND_RESUME or all-access.
    process = api.OpenProcess(0x400, False, args.pid)
    api.check(process, "OpenProcess(PROCESS_QUERY_INFORMATION)")
    try:
        identity = api.identity(process)
        if os.path.normcase(os.path.realpath(identity["path"])) != os.path.normcase(
            os.path.realpath(args.executable)
        ):
            raise ValueError("process path differs from pinned Windows client")
        with Path(identity["path"]).open("rb") as executable:
            digest = hashlib.file_digest(executable, "sha256").hexdigest()
        if digest != EXPECTED_SHA256:
            raise ValueError("process executable SHA256 differs from pinned Windows client")
        machine, native_machine = WORD(), WORD()
        api.check(
            api.IsWow64Process2(process, ct.byref(machine), ct.byref(native_machine)),
            "IsWow64Process2",
        )
        if machine.value != 0 or native_machine.value != 0x8664:
            raise ValueError("native AMD64 Windows client required")
        if identity != api.identity(process):
            raise ValueError("process identity changed during admission")
        result = {
            "schema": "TM-WINDOWS-FP-PSS-DIAGNOSTIC-1",
            "identity": identity,
            "executable_sha256": digest,
            "layout": layout,
            "capture_flags": CAPTURE_FLAGS,
            "context_flags": CONTEXT_FLAGS,
            "manual_thread_suspends": 0,
            "va_clone": False,
            "physics_thread_identified": False,
            "claim": "thread FP controls at rest; simulation thread remains unidentified",
        }
        if args.inspect:
            return {**result, "mode": "identity-only", "snapshot_taken": False}
        if args.expected_creation_filetime != identity["created_filetime"]:
            raise ValueError("explicit expected process creation FILETIME required and must match")
        threads, modules = api.inventory(args.pid), api.inventory(args.pid, modules=True)
        if not threads or identity != api.identity(process):
            raise ValueError("process exited/changed or no bounded thread inventory")
        result.update(capture_snapshot(api, process, identity, modules))
        if identity != api.identity(process):
            raise ValueError("process identity changed after capture")
        return {**result, "mode": "fp-controls", "snapshot_taken": True}
    finally:
        api.CloseHandle(process)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument(
        "--executable", type=Path, required=True, help="absolute client executable path"
    )
    parser.add_argument("--expected-creation-filetime", type=int)
    parser.add_argument("--inspect", action="store_true", help="identity/hash only; no PSS capture")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.pid <= 0 or args.pid == os.getpid():
        parser.error("explicit different positive PID required")
    if not args.executable.is_absolute():
        parser.error("--executable must be an absolute path")
    if args.worker:
        try:
            result = {"ok": True, **worker(args)}
        except Exception as exc:
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            if isinstance(exc, SnapshotError):
                result["cleanup"] = exc.cleanup
        print(json.dumps(result, sort_keys=True))
        return 0 if result["ok"] else 1
    if not args.output:
        parser.error("--output must name a new evidence file")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--pid",
        str(args.pid),
        "--executable",
        str(args.executable),
    ]
    if args.inspect:
        command.append("--inspect")
    elif args.expected_creation_filetime is not None:
        command.extend(("--expected-creation-filetime", str(args.expected_creation_filetime)))
    else:
        parser.error("capture requires --expected-creation-filetime from identity-only inspection")
    with args.output.open("x", encoding="utf-8") as output:
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            result = json.loads(completed.stdout)
        except subprocess.TimeoutExpired:
            result = {
                "ok": False,
                "error": "PSS worker exceeded five-second budget and was terminated",
            }
        result["parent_elapsed_seconds"] = time.monotonic() - started
        result["reader_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps({"ok": result["ok"], "output": str(args.output.resolve())}))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
