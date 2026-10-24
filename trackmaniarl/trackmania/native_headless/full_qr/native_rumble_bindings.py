"""Read independently bound live material/model inputs from an owned held actor."""

from __future__ import annotations

import os
from pathlib import Path
from struct import unpack
from typing import Any

from trackmaniarl.trackmania.native_headless._runtime.posix_api import linux_os, linux_select
from trackmaniarl.trackmania.native_headless.full_qr.final_capture import FinalCollectorBoundary
from trackmaniarl.trackmania.native_headless.full_qr.final_rumble import (
    FinalRumbleContext,
    decode_final_rumble_input,
)
from trackmaniarl.trackmania.native_headless.full_qr.physical_rumble import PhysicalRumbleInput
from trackmaniarl.trackmania.native_headless.full_qr.prepared_impact_capture import (
    PreparedImpactContext,
)
from trackmaniarl.trackmania.native_headless.full_qr.rumble_material import select_rumble_material
from trackmaniarl.trackmania.native_headless.full_qr.vehicle_capture import VehicleCapture


class HeldRumbleBindings:
    """Own read-only descriptors; never call engine code or write engine memory."""

    def __init__(self, session: Any, *, material14_uses_default: bool) -> None:
        if type(material14_uses_default) is not bool:
            raise ValueError("actual paired client material14 override setting required")
        self.material14_uses_default = material14_uses_default
        self.pid, self.meta = session.pid, dict(session.meta)
        self.pidfd: int | None = linux_os.pidfd_open(self.pid)
        self.mem = None
        self._poll = linux_select.poll()
        self._poll.register(self.pidfd, linux_select.POLLIN)
        try:
            self.birth = self._birth()
            self.mem = os.open(f"/proc/{self.pid}/mem", os.O_RDONLY | linux_os.O_CLOEXEC)
            self.core = self.meta["vehicle"]
            self.descriptor = self._q(self.core)
            self.owner = self._q(self.core + 8)
            self._class(self.descriptor, 0x090EA000)
            if self._u(self.core + 0x380) != 4:
                raise ValueError("four actual current car wheel models required")
            self.wheels = tuple(self._q(self.core + 0x388 + i * 8) for i in range(4))
            self.models = tuple(self._read(pointer, 32) for pointer in self.wheels)
            self.contact_context = PreparedImpactContext(
                pid=self.pid,
                tid=self.pid,
                module_base=self.meta["module_slide"],
                vehicle=self.core,
                participant=self.meta["participant"],
                sim=self.meta["sim"],
                controller=self.meta["controller"],
                normalization_owner=self.owner,
                normalization_header=self._read(self.owner, 32),
                normalization_thresholds=self._read(self.owner + 0x1B0, 16),
                wheel_models=self.wheels,
                wheel_headers=tuple(model[:16] for model in self.models),
            )
            self._guard()
        except BaseException:
            self.close()
            raise

    def _birth(self) -> str:
        return Path(f"/proc/{self.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]

    def _read(self, address: int, size: int) -> bytes:
        if self.mem is None or not 4096 <= address < address + size < 1 << 47:
            raise ValueError("open bounded actual native read required")
        if not 0 < size <= 4096 or self._poll.poll(0):
            raise ValueError("live owned native process and bounded source read required")
        raw = linux_os.pread(self.mem, size, address)
        if len(raw) != size:
            raise ValueError("complete actual native source read required")
        return raw

    def _q(self, address: int) -> int:
        _decoded_scalar: int = unpack("<Q", self._read(address, 8))[0]
        return _decoded_scalar

    def _u(self, address: int) -> int:
        _decoded_scalar: int = unpack("<I", self._read(address, 4))[0]
        return _decoded_scalar

    def _class(self, pointer: int, expected: int) -> None:
        # Source435: these native model objects expose GetId in slot4.
        getter = self._q(self._q(pointer) + 4 * 8)
        code = self._read(getter, 11)
        if code != b"\x55\x48\x89\xe5\xb8" + expected.to_bytes(4, "little") + b"\x5d\xc3":
            raise ValueError("actual model source class getter changed")

    def _guard(self) -> None:
        if self._birth() != self.birth or self._poll.poll(0):
            raise ValueError("owned native process identity changed")
        meta = self.meta
        if (
            self._q(meta["participant"] + 0x1118) != self.core
            or self._q(meta["controller"] + 0x1A70) != meta["sim"]
            or self._q(self.core) != self.descriptor
            or self._q(self.core + 8) != self.owner
            or self._u(self.core + 0x380) != 4
            or tuple(self._q(self.core + 0x388 + i * 8) for i in range(4)) != self.wheels
            or tuple(self._read(pointer, 32) for pointer in self.wheels) != self.models
            or self._read(self.owner, 32) != self.contact_context.normalization_header
            or self._read(self.owner + 0x1B0, 16) != self.contact_context.normalization_thresholds
        ):
            raise ValueError("current native car/model source binding changed")

    def source(self, capture: VehicleCapture) -> PhysicalRumbleInput:
        self._guard()
        clock = self._u(self.meta["sim"] + 0x48)
        if capture.native_t != clock:
            raise ValueError("actual current held physical capture required")
        material = select_rumble_material(capture.visual_state, self.models).material
        palette = {}
        if material != 80:
            selected = 0 if material == 14 and self.material14_uses_default else material
            pointer = self._q(self.descriptor + 0x6B8 + selected * 8)
            self._class(pointer, 0x090F1000)
            palette[material] = self._read(pointer, 0x90)
            if self._q(self.descriptor + 0x6B8 + selected * 8) != pointer:
                raise ValueError("selected actual material binding changed")
        enabled = self._u(self.core + 0x1388) != 0
        result = PhysicalRumbleInput(capture.visual_state, self.models, palette, enabled)
        self._guard()
        if self._u(self.meta["sim"] + 0x48) != clock:
            raise ValueError("actor advanced during held physical source read")
        return result

    def final_source(
        self, capture: VehicleCapture, witness: FinalCollectorBoundary, raw: bytes
    ) -> PhysicalRumbleInput:
        """Join an actual final snapshot to this episode's original binding.

        The live-reader guard above remains unchanged. A dead child never
        authorizes reuse of its last palette or an assumed surface flag.
        """
        return decode_final_rumble_input(
            raw,
            capture,
            witness,
            context=FinalRumbleContext(
                self.contact_context,
                self.descriptor,
                self.models,
                self.material14_uses_default,
            ),
        )

    def close(self) -> None:
        if self.mem is not None:
            os.close(self.mem)
            self.mem = None
        if self.pidfd is not None:
            os.close(self.pidfd)
            self.pidfd = None
