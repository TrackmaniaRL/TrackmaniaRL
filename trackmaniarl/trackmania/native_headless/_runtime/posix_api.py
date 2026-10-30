"""Type the Linux-only capabilities used by the guarded native runtime.

The aliases retain the actual standard-library modules. They introduce no
fallback constants, emulation, or import-time access to Linux-only attributes.
Offline imports therefore remain portable; executing the native runtime still
requires its existing Linux and capability guards.
"""

from __future__ import annotations

import os
import select
import signal
import socket
from typing import Protocol, cast


class _LinuxOS(Protocol):
    O_CLOEXEC: int
    O_DIRECTORY: int
    O_NOFOLLOW: int
    O_NONBLOCK: int
    O_PATH: int
    WNOHANG: int

    def pidfd_open(self, pid: int, flags: int = 0) -> int: ...
    def pread(self, fd: int, length: int, offset: int) -> bytes: ...
    def getuid(self) -> int: ...
    def getgid(self) -> int: ...


class _LinuxSignal(Protocol):
    SIGKILL: int

    def pidfd_send_signal(self, pidfd: int, signalnum: int) -> None: ...


class _LinuxSocket(Protocol):
    AF_UNIX: int
    SO_PEERCRED: int


class _Poll(Protocol):
    def register(self, fd: int, eventmask: int) -> None: ...
    def poll(self, timeout: int | None = None) -> list[tuple[int, int]]: ...


class _LinuxSelect(Protocol):
    POLLIN: int

    def poll(self) -> _Poll: ...


linux_os = cast(_LinuxOS, os)
linux_signal = cast(_LinuxSignal, signal)
linux_socket = cast(_LinuxSocket, socket)
linux_select = cast(_LinuxSelect, select)
