"""Join actual delivered contacts to held reader boundaries, without motor labels.

Receipts count the observed pinned delivery hook. They do not by themselves
qualify every possible engine contact path, a motor waveform or negative
collision coverage. The owner must hold the matching actor throughout reads.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from struct import unpack_from

from trackmaniarl.trackmania.native_headless._runtime.posix_api import linux_os
from trackmaniarl.trackmania.native_headless.full_qr.prepared_contact_capture import (
    PreparedContactCapture,
)
from trackmaniarl.trackmania.native_headless.full_qr.prepared_impact_capture import (
    PreparedImpactContext,
    decode_prepared_impact,
)

RECEIPT_SIZE = 96
CONTACT_SIZE = 512
MAX_CONTACTS = 100_000


@dataclass(frozen=True, slots=True)
class ContactDeliveryReceipt:
    sequence: int
    native_t: int
    contacts: int
    last_callback: int
    last_event: int


def decode_contact_receipt(raw: bytes, context: PreparedImpactContext) -> ContactDeliveryReceipt:
    if type(raw) is not bytes or len(raw) != RECEIPT_SIZE or raw[:8] != b"TMCR0001":
        raise ValueError("exact TMCR0001 contact delivery receipt required")
    if unpack_from("<4I", raw, 8) != (RECEIPT_SIZE, 1, context.pid, context.tid):
        raise ValueError("contact receipt header/owner mismatch")
    sequence, clock, reserved = unpack_from("<QII", raw, 24)
    if (
        not 1 <= sequence <= 16200
        or clock != 2190 + (sequence - 1) * 10
        or clock > 162200
        or reserved
    ):
        raise ValueError("contact receipt sequence/clock mismatch")
    contacts, callback, event = unpack_from("<QII", raw, 40)
    if (
        contacts > MAX_CONTACTS
        or unpack_from("<Q", raw, 88)[0] != contacts
        or (contacts == 0 and (callback != 0 or event != 0))
        or (contacts > 0 and not 2190 <= callback <= clock)
        or (contacts > 0 and (callback % 10 or not 2190 <= event <= 162200))
    ):
        raise ValueError("contact receipt delivered counter/time mismatch")
    if unpack_from("<4Q", raw, 56) != (
        context.participant,
        context.vehicle,
        context.sim,
        context.module_base,
    ):
        raise ValueError("contact receipt current actor binding mismatch")
    return ContactDeliveryReceipt(sequence, clock, contacts, callback, event)


@dataclass(frozen=True, slots=True)
class DeliveredContactBoundary:
    receipt: ContactDeliveryReceipt
    contacts: tuple[PreparedContactCapture, ...]


def _snapshot(path: Path, record_size: int, maximum: int) -> tuple[int, os.stat_result]:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    info = os.fstat(descriptor)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or not 0 <= info.st_size <= maximum * record_size
        or info.st_size % record_size
    ):
        os.close(descriptor)
        raise ValueError("incomplete or unbounded contact delivery stream")
    return descriptor, info


class HeldContactDeliveryReader:
    """Incremental two-stream join, transactional on a complete held snapshot.

    Every intermediate10ms receipt is consumed, including zero new deliveries.
    No watermark is sent to the reward collector by this component.
    """

    def __init__(self, receipts: Path, contacts: Path, context: PreparedImpactContext) -> None:
        self.receipts_path, self.contacts_path, self.context = receipts, contacts, context
        self._receipt_sequence = 0
        self._contacts = 0
        self._last_contact: PreparedContactCapture | None = None
        self._files: tuple[tuple[int, int], tuple[int, int]] | None = None

    def take(self, native_t: int) -> tuple[DeliveredContactBoundary, ...]:
        if type(native_t) is not int or not 2190 <= native_t <= 162200 or native_t % 10:
            raise ValueError("actual held contact boundary inside the native domain required")
        receipt_fd, before = _snapshot(self.receipts_path, RECEIPT_SIZE, 16200)
        contact_fd = None
        try:
            contact_fd, contacts_before = _snapshot(self.contacts_path, CONTACT_SIZE, MAX_CONTACTS)
            files = (
                (before.st_dev, before.st_ino),
                (contacts_before.st_dev, contacts_before.st_ino),
            )
            if self._files is not None and self._files != files:
                raise ValueError("contact delivery stream identity changed")
            target_sequence = (native_t - 2190) // 10 + 1
            if (
                before.st_size != target_sequence * RECEIPT_SIZE
                or target_sequence <= self._receipt_sequence
                or contacts_before.st_size < self._contacts * CONTACT_SIZE
            ):
                raise ValueError("contact streams do not match a fresh held reader boundary")
            sequence, count, last = self._receipt_sequence, self._contacts, self._last_contact
            boundaries = []
            for index in range(sequence, target_sequence):
                raw = linux_os.pread(receipt_fd, RECEIPT_SIZE, index * RECEIPT_SIZE)
                receipt = decode_contact_receipt(raw, self.context)
                if receipt.sequence != index + 1 or not count <= receipt.contacts:
                    raise ValueError("contact delivered prefix regressed")
                if receipt.contacts * CONTACT_SIZE > contacts_before.st_size:
                    raise ValueError("contact receipt promises records absent from its stream")
                delivered = []
                for contact_index in range(count, receipt.contacts):
                    raw_contact = linux_os.pread(
                        contact_fd, CONTACT_SIZE, contact_index * CONTACT_SIZE
                    )
                    contact = decode_prepared_impact(raw_contact, self.context)
                    if (
                        contact.sequence != contact_index + 1
                        or contact.native_time > receipt.native_t
                        or (last is not None and contact.native_time < last.native_time)
                    ):
                        raise ValueError("contact delivery sequence/callback order mismatch")
                    delivered.append(contact)
                    last = contact
                expected = (0, 0) if last is None else (last.native_time, last.event_time)
                if (receipt.last_callback, receipt.last_event) != expected:
                    raise ValueError("contact receipt does not describe its exact delivered prefix")
                count = receipt.contacts
                boundaries.append(DeliveredContactBoundary(receipt, tuple(delivered)))
            if contacts_before.st_size != count * CONTACT_SIZE:
                raise ValueError("uncredited contact records appeared beyond held reader receipt")
            for descriptor, original in ((receipt_fd, before), (contact_fd, contacts_before)):
                after = os.fstat(descriptor)
                if (after.st_size, after.st_mtime_ns) != (original.st_size, original.st_mtime_ns):
                    raise ValueError("contact stream changed while actor should be held")
        finally:
            os.close(receipt_fd)
            if contact_fd is not None:
                os.close(contact_fd)
        # Failed reads leave the cursor intact; owners still treat source faults
        # as fatal for the environment rather than accepting partial transitions.
        self._receipt_sequence, self._contacts = target_sequence, count
        self._last_contact, self._files = last, files
        return tuple(boundaries)
