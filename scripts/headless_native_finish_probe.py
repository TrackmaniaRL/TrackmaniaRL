"""Bounded diagnostic replay of original native records, preserving natural EOF.

No policy/action IDs are manufactured for the original waveform. V7 op6 writes
the literal original SGB bytes back unchanged before their first load. A neutral
administrative interval is explicit and may prevent the human tape finishing.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
import time
from pathlib import Path

from headless_immediate_probe import ImmediateSession

from trackmaniarl.trackmania.native_headless._runtime import env, validator
from trackmaniarl.trackmania.native_headless._runtime.host import u32, u64
from trackmaniarl.trackmania.native_headless._runtime.immediate import (
    ImmediateSession as ProductionImmediateSession,
)
from trackmaniarl.trackmania.native_headless._runtime.outcome import MAP_UID
from trackmaniarl.trackmania.native_headless._runtime.ownership import process_identity
from trackmaniarl.trackmania.native_headless._runtime.shapes import (
    ParseRecordsSpec,
    QualifyReaderSpec,
    RequestSpec,
    ResponseSpec,
)

MAX_INTERVALS = 4001
MAX_BYTES = 64 * 1024 * 1024


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def tree_bytes(paths: tuple[Path, ...]) -> int:
    return sum(file.stat().st_size for path in paths for file in path.rglob("*") if file.is_file())


class OriginalSession(ImmediateSession):
    def advance_original(self, n: int, immutable: bytes) -> dict:
        reader, anchor = self.reader, u32(self.boundary, 112)
        env.qualify_reader(
            QualifyReaderSpec(raw=reader, pid=self.pid, meta=self.meta, anchor=anchor)
        )
        p = u32(reader, 112)
        if not 1 <= n <= 5 or p + n > self.meta["input_count"]:
            raise ValueError("bounded original record range required")
        original = self.read_mem(u64(reader, 88) - 4 + p * 32, n * 32)
        if original != immutable[p * 32 : (p + n) * 32]:
            raise ValueError("child future records differ from immutable original parent tape")
        channels = [original[index * 32 + 4 : index * 32 + 16] for index in range(n)]
        if not all(math.isfinite(value) for raw in channels for value in struct.unpack("<3f", raw)):
            raise ValueError("nonfinite original waveform rejected")
        sequence, epoch = self.seq + 1, self.epoch
        row = {
            "p": p,
            "requested_intervals": n,
            "before_native_t": anchor,
            "original_record_hex": original.hex(),
            "original_sgb_hex": [raw.hex() for raw in channels],
            "input_source": "immutable parent records before first load; op6 identity write",
            "policy_action_id": None,
            "rx_start_offset": self.rx.tell(),
            "eof": False,
            "eof_stage": None,
        }
        self.last_attempt = row
        self.request(RequestSpec(op=6, n=n, p=p, k=p, befores=channels, afters=channels))
        committed, receipt = self.response(ResponseSpec(kind=6, seq=sequence, n=n, epoch=epoch))
        expected = struct.pack("<4I", p, p, n, 2) + b"".join(raw + raw for raw in channels)
        expected += bytes(144 - n * 24)
        if committed != reader or receipt != expected:
            raise ValueError("original waveform exact identity-write receipt mismatch")
        row["receipt_hex"] = receipt.hex()
        try:
            self.boundary, _ = self.response(
                ResponseSpec(kind=1, seq=sequence, n=n, epoch=epoch + 1)
            )
        except EOFError:
            row.update(eof=True, eof_stage="boundary")
        else:
            if u32(self.boundary, 112) != anchor + n * 10:
                raise ValueError("complete original waveform boundary interval count mismatch")
            try:
                self.arm()
            except EOFError:
                row.update(eof=True, eof_stage="reader")
        path = self.local_posts / f"post-{self.pid}.bin"
        with path.open("rb") as stream:
            stream.seek(self.finish_post_offset)
            raw = stream.read()
            self.finish_post_offset = stream.tell()
        if len(raw) % 256:
            raise ValueError("partial native POST/audit record preserved")
        records = [raw[offset : offset + 256] for offset in range(0, len(raw), 256)]
        posts = [
            record for record in records if record[:8] == b"TMPOST03" and u32(record, 148) == 75
        ]
        audits = [
            record
            for record in records
            if record[:8] == b"TMAUD003" and u32(record, 200) == sequence
        ]
        if len(posts) > n or (not row["eof"] and len(posts) != n) or len(audits) != n:
            raise ValueError(
                "original replay requires complete audits and actual consecutive POST prefix"
            )
        for index, post in enumerate(posts):
            env.qualify_post(post, self.pid, self.meta)
            gas, brake, steer = struct.unpack_from("<3f", post, 88)
            if (
                u32(post, 112) != anchor + (index + 1) * 10
                or struct.pack("<3f", steer, gas, brake) != channels[index]
            ):
                raise ValueError(
                    "actual applied original POST channels or literal sample clock differ"
                )
        for index, audit in enumerate(audits):
            identity = tuple(
                u32(audit, offset) for offset in (176, 180, 184, 188, 204, 208, 248, 252)
            )
            record = original[index * 32 : (index + 1) * 32]
            if identity != (p, p + index, n, anchor, 1, p, n, index):
                raise ValueError("original identity-write audit authorization mismatch")
            if (
                audit[40:72] != record
                or audit[72:104] != record
                or audit[104:136] != reader[152:184]
                or audit[136:168] != reader[152:184]
                or audit[168:172] != channels[index][:4]
                or audit[172:176] != channels[index][:4]
            ):
                raise ValueError("identity write changed original record or current shadow bytes")
        if posts:
            self.boundary = posts[-1]
        row.update(
            actual_intervals=len(posts),
            after_native_t=u32(self.boundary, 112),
            applied_post_hex=[post.hex() for post in posts],
            audit_hex=[audit.hex() for audit in audits],
            rx_end_offset=self.rx.tell(),
        )
        return row


def probe_production_guard(session: OriginalSession) -> dict:
    """Exercise the exact production preflight with the real held native reader."""
    p, count = u32(session.reader, 112), session.meta["input_count"]
    if p + 5 != count:
        raise ValueError("real final-five-record reader required for live guard proof")
    original_read, original_request = session.read_mem, session.request
    calls = {"native_reads_during_guard": 0, "requests_during_guard": 0}

    def tracked_read(address: int, size: int) -> bytes:
        calls["native_reads_during_guard"] += 1
        return original_read(address, size)

    def tracked_request(spec: RequestSpec) -> None:
        calls["requests_during_guard"] += 1
        original_request(spec)

    def snapshot() -> dict:
        owner = process_identity(session.pid)
        if owner is None or owner["state"] in ("Z", "X") or session.proc.poll() is not None:
            raise ValueError("real owned validator and retained driver must remain live")
        return {
            "seq": session.seq,
            "epoch": session.epoch,
            "rx_bytes": session.rx.tell(),
            "request_count": len(session.requests),
            "response_count": len(session.responses),
            "post_bytes": (session.local_posts / f"post-{session.pid}.bin").stat().st_size,
            "reader_hex": session.reader.hex(),
            "boundary_hex": session.boundary.hex(),
            "input_bytes_sha256": digest(original_read(session.meta["input_base"] - 4, count * 32)),
            "process_identity": {key: owner[key] for key in ("pid", "birth", "uid", "pgid", "sid")},
            "driver_alive": True,
        }

    before = snapshot()
    session.read_mem, session.request = tracked_read, tracked_request
    rejection = None
    try:
        ProductionImmediateSession.advance(session, 39, 5)
    except ValueError as error:
        if "unsupported-final-reader-boundary" not in str(error):
            raise
        rejection = str(error)
    finally:
        session.read_mem, session.request = original_read, original_request
    after = snapshot()
    if rejection is None or before != after or any(calls.values()):
        raise ValueError(
            "production final-reader preflight had native side effects or no rejection"
        )
    return {
        "rejected": True,
        "error": rejection,
        "before": before,
        "after": after,
        **calls,
        "production_method_source_sha256": digest(
            Path(sys.modules[ProductionImmediateSession.__module__].__file__).read_bytes()
        ),
        "scope": "production advance with actual experimental session; no provisioning claim",
        "terminal_claim": False,
    }


def qualify_terminal(session: OriginalSession, reaped: dict) -> dict:
    if not (
        reaped["kind"] == "reaped"
        and reaped["pid"] == session.pid
        and reaped["process_gone"] is True
        and reaped["parent_unchanged"] is True
        and reaped["truncated_by_driver"] is False
        and reaped["exit_status"] is None
    ):
        raise ValueError("natural driver reap proof required")
    prefix = (session.out / "stdout-prefix.bin").read_bytes()
    child = (session.ep / "stdout-slice.bin").read_bytes()
    envelope = child if child.lstrip().startswith(b"[") else prefix + child
    record = validator.parse_records(
        ParseRecordsSpec(
            text=envelope.decode("utf-8"),
            expected_files={"g.Ghost.Gbx"},
            expected_map_uid=MAP_UID,
            mode="parent",
            allow_inherited_array_close=False,
        )
    )[0]
    collector = (session.local_posts / f"result-{session.pid}.bin").read_bytes()
    if not collector or len(collector) % 768 or len(collector) > 4096 * 768:
        raise ValueError("complete bounded original TMRESULT collector required")
    results = [collector[offset : offset + 768] for offset in range(0, len(collector), 768)]
    for hit, raw in enumerate(results, 1):
        if raw[:8] != b"TMRESULT" or (
            u32(raw, 8),
            u32(raw, 12),
            u32(raw, 16),
            u32(raw, 20),
            u64(raw, 24),
            u32(raw, 148),
            u32(raw, 240),
            u32(raw, 248),
        ) != (768, 1, session.pid, session.pid, hit, 24575, 13, 1):
            raise ValueError("original complete own TMRESULT profile mismatch")
        if u64(raw, 40) != session.meta["controller"] or u64(raw, 48) != session.meta["sim"]:
            raise ValueError("original TMRESULT owner chain mismatch")
    value = record["ValidatedResult"]
    if value is None or not all(
        type(field) is int and 0 <= field < 2**32 for field in value.values()
    ):
        raise ValueError("original validator has no complete computed result")
    last = results[-1]
    if u32(last, 128) != u32(session.boundary, 112):
        raise ValueError("original TMRESULT differs from last actual POST native clock")
    if (
        u32(last, 152),
        u32(last, 156),
        u32(last, 160),
        u32(last, 184),
        u32(last, 192),
        u32(last, 256),
        u32(last, 260),
    ) != (value["Time"], value["Score"], value["NbRespawns"], 1, 8, 0, value["Time"]):
        raise ValueError("full original validator and TMRESULT disagree")
    return {
        "validator": record,
        "validator_array_sha256": digest(envelope),
        "collector_sha256": digest(collector),
        "collector_records": len(results),
        "last_tmresult_hex": last.hex(),
        "computed_finish": value["Time"] != 4294967295 and value["NbCheckpoints"] == 1,
        "validation_acceptance": record["IsValid"],
        "last_actual_post_native_t": u32(session.boundary, 112),
        "clock_is_not_extrapolated_to_windows_race": True,
    }


def main() -> None:
    config_path, receipt_path, output = map(Path, sys.argv[1:4])
    guard_mode = sys.argv[4:] == ["--production-guard"]
    if sys.argv[4:] and not guard_mode:
        raise ValueError("only optional --production-guard diagnostic mode is supported")
    if output.exists():
        raise FileExistsError("fresh original-finish diagnostic output required")
    config, receipt = json.loads(config_path.read_bytes()), json.loads(receipt_path.read_bytes())
    deployment = Path(config["deployment_manifest"]).read_bytes()
    if digest(deployment) != config["deployment_manifest_sha256"]:
        raise ValueError("original deployment manifest identity differs")
    driver_sha = json.loads(deployment)["driverBuildApproval"]["driverSha256"]
    output.mkdir(parents=True)
    with (output / "source-headless_native_finish_probe.py").open("xb") as source_file:
        source_file.write(Path(__file__).read_bytes())
    report = {
        "schema": "tmrl.original-record-natural-finish-diagnostic/v1",
        "finished": False,
        "natural_eof_observed": False,
        "learning_executed": False,
        "client_parity_claim": False,
        "production_gate_claim": False,
        "physical_state_writes": False,
        "original_waveform_policy_action_ids": None,
        "production_guard_mode": guard_mode,
        "budget": {
            "intervals_including_neutral_reset": MAX_INTERVALS,
            "seconds": 120,
            "bytes": MAX_BYTES,
        },
        "requested_original_intervals": 0,
        "actual_original_intervals": 0,
        "steps": 0,
        "sources": {
            "config_sha256": digest(config_path.read_bytes()),
            "v7_receipt_sha256": digest(receipt_path.read_bytes()),
            "reader_source_sha256": digest(Path(__file__).read_bytes()),
        },
    }
    started = time.monotonic()
    session = None
    success = False
    try:
        session = OriginalSession(
            Path(config["runtime_root"]), receipt, output / output.name, 2190, driver_sha
        )
        session.deadline = min(session.deadline, started + 120)
        immutable = (session.out / "parent.inputs.bin").read_bytes()
        if len(immutable) != 4152 * 32:
            raise ValueError("immutable full original record count required")
        report["immutable_original_records_sha256"] = digest(immutable)
        report["initial_post_hex"] = session.reset().hex()
        report["initial_reader_hex"] = session.reader.hex()
        report["administrative_neutral_reset"] = session.step(36, 1)
        session.finish_post_offset = (
            (session.local_posts / f"post-{session.pid}.bin").stat().st_size
        )
        with (output / "steps.jsonl").open("x", encoding="utf-8") as stream:
            while report["requested_original_intervals"] < MAX_INTERVALS - 1:
                if (output / "STOP").exists():
                    raise InterruptedError("natural-finish diagnostic cooperative STOP")
                session.remaining()
                if guard_mode and u32(session.reader, 112) + 5 == session.meta["input_count"]:
                    report["production_guard"] = probe_production_guard(session)
                    report["stop_reason"] = (
                        "unsupported final-reader request rejected synchronously"
                    )
                    success = True
                    break
                if report["steps"] % 20 == 0:
                    size = tree_bytes((output, session.tmp, session.work))
                    report["latest_budget_bytes"] = size
                    if size > MAX_BYTES:
                        raise RuntimeError("natural-finish diagnostic byte budget exhausted")
                available = session.meta["input_count"] - u32(session.reader, 112)
                n = min(5, MAX_INTERVALS - 1 - report["requested_original_intervals"], available)
                if n <= 0:
                    raise RuntimeError("original tape exhausted without literal natural EOF")
                row = session.advance_original(n, immutable)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                report["steps"] += 1
                report["requested_original_intervals"] += n
                report["actual_original_intervals"] += row["actual_intervals"]
                report["last_step"] = row
                if row["eof"]:
                    report["natural_eof_observed"] = True
                    session.connection.close()
                    session.connection = None
                    reaped = session.command("reap")
                    session.episode = None
                    report["natural_reap"] = reaped
                    report["terminal"] = qualify_terminal(session, reaped)
                    report["finished"] = report["terminal"]["computed_finish"]
                    success = True
                    break
            if not report["natural_eof_observed"] and not report.get("production_guard"):
                report["stop_reason"] = (
                    "finite original-interval budget exhausted; no finish fabricated"
                )
                success = True
    except BaseException as exc:
        report["error"] = repr(exc)
        if session is not None:
            report["last_attempt"] = getattr(session, "last_attempt", None)
        raise
    finally:
        try:
            if session is not None:
                session.close(error=not success)
                report["cleanup"] = session.cleanup
                if session.cleanup["remainingOwned"]:
                    raise RuntimeError("original-finish probe left owned processes")
        finally:
            report["wall_seconds_not_a_benchmark"] = time.monotonic() - started
            with (output / "report.json").open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
                stream.write("\n")


if __name__ == "__main__":
    main()
