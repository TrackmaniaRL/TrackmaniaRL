"""Explicit V7 release/drain experiment; no learner credit or invented terminal.

Op2 intentionally closes IPC at an owned POST. Autonomous native completion is
then established separately through normal reap, full original validator JSON,
and the original TMRESULT collector. Production profiles and pins are unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
import time
import zipfile
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from scripts.headless_immediate_probe import ImmediateSession
from trackmaniarl.trackmania.native_headless._runtime import artifacts, codec_v7, env, validator
from trackmaniarl.trackmania.native_headless._runtime.host import u32, u64
from trackmaniarl.trackmania.native_headless._runtime.immediate import (
    ImmediateSession as ProductionImmediateSession,
)
from trackmaniarl.trackmania.native_headless._runtime.outcome import MAP_UID
from trackmaniarl.trackmania.native_headless._runtime.shapes import (
    ParseRecordsSpec,
    RequestSpec,
    ResponseSpec,
)

MAX_BYTES = 64 * 1024 * 1024
SOURCE_ARCHIVE_SHA256 = "863ae95ff8a8d1e5339aea255b2675fd2d09eb652f8f6d9af4f437317bd346b9"
SOURCE_EXPORT = "Trackmania2020RE/analysis/tmrl-shortcut/export-001/jimmy-17715.json"
SOURCE_GHOST = "Trackmania2020RE/analysis/tmrl-shortcut/assets-001/jimmy-17715.Ghost.Gbx"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def chunks(raw: bytes, size: int) -> list[bytes]:
    if len(raw) % size:
        raise ValueError(f"partial {size}-byte evidence record")
    return [raw[offset : offset + size] for offset in range(0, len(raw), size)]


def inspect_native_drain(  # noqa: PLR0913
    stdout: bytes,
    posts_raw: bytes,
    results_raw: bytes,
    pid: int,
    meta: dict[str, Any],
    allowed_audits: tuple[bytes, ...] = (),
) -> dict[str, Any]:
    records = validator.parse_records(
        ParseRecordsSpec(
            text=stdout.decode("utf-8"),
            expected_files={"g.Ghost.Gbx"},
            expected_map_uid=MAP_UID,
            mode="parent",
            allow_inherited_array_close=False,
        )
    )
    blocks = chunks(posts_raw, 256)
    posts, readers, audits = [], [], []
    for raw in blocks:
        if (u32(raw, 8), u32(raw, 16), u32(raw, 20)) != (256, pid, pid):
            raise ValueError("drain stream ownership mismatch")
        if raw[:8] in (b"TMAUD002", b"TMAUD003"):
            audits.append(raw)
        if raw[:8] == b"TMPOST03" and u32(raw, 148) == 75:
            _, native_t = env.qualify_post(raw, pid, meta)
            if not all(math.isfinite(value) for value in struct.unpack_from("<10f", raw, 152)):
                raise ValueError("nonfinite autonomous physical state")
            posts.append((native_t, raw))
        elif raw[:8] == b"TMREAD01" and u32(raw, 148) == 447:
            if u64(raw, 40) != meta["controller"] or u64(raw, 48) != meta["sim"]:
                raise ValueError("autonomous reader owner mismatch")
            p, count, next_t, before_t = (u32(raw, offset) for offset in (112, 116, 120, 124))
            if count != 4152 or p >= count or next_t != before_t + 10:
                raise ValueError("autonomous reader original clock/count mismatch")
            readers.append(
                {
                    "p": p,
                    "before_native_t": before_t,
                    "next_native_t": next_t,
                    "input_record_label_ms": p * 10 + meta["start_offset_ms"],
                }
            )
    if not posts or any(right[0] != left[0] + 10 for left, right in pairwise(posts)):
        raise ValueError("complete consecutive native POST clock stream required")
    if tuple(audits) != allowed_audits:
        raise ValueError("drain write audits differ from explicitly qualified source repair")
    result_records = chunks(results_raw, 768)
    if not 1 <= len(result_records) <= 4096:
        raise ValueError("bounded original result collector required")
    clocks = {native_t for native_t, _ in posts}
    for hit, raw in enumerate(result_records, 1):
        fields = tuple(u32(raw, offset) for offset in (8, 12, 16, 20, 148, 240, 248))
        if (
            raw[:8] != b"TMRESULT"
            or fields != (768, 1, pid, pid, 24575, 13, 1)
            or u64(raw, 24) != hit
        ):
            raise ValueError("complete original own TMRESULT profile required")
        if u64(raw, 40) != meta["controller"] or u64(raw, 48) != meta["sim"]:
            raise ValueError("TMRESULT owner mismatch")
        if u32(raw, 128) not in clocks:
            raise ValueError("TMRESULT clock has no actual native POST")
    record, last = records[0], result_records[-1]
    value = record["ValidatedResult"]
    correlated = False
    if value is not None:
        if not all(type(field) is int and 0 <= field < 2**32 for field in value.values()):
            raise ValueError("computed validator result contains invalid scalars")
        count = value["NbCheckpoints"]
        if not 0 <= count <= 64:
            raise ValueError("computed checkpoint count outside collector bound")
        expected = (value["Time"], value["Score"], value["NbRespawns"], count, count * 8)
        if tuple(u32(last, offset) for offset in (152, 156, 160, 184, 192)) != expected:
            raise ValueError("full validator and final TMRESULT disagree")
        if count and u32(last, 256 + (count - 1) * 8 + 4) != value["Time"]:
            raise ValueError("last checkpoint time differs from computed finish")
        correlated = True
    finish = correlated and value["Time"] != 4294967295 and value["NbCheckpoints"] == 1
    dnf = correlated and value["Time"] == 4294967295
    return {
        "full_original_validator_complete": True,
        "computed_result_correlated": correlated,
        "computed_finish": finish,
        "computed_dnf": dnf,
        "outcome": "physical-finish"
        if finish
        else "computed-DNF"
        if dnf
        else "inconclusive-result",
        "validation_acceptance": record["IsValid"],
        "validator": record,
        "post_count": len(posts),
        "first_native_t": posts[0][0],
        "last_native_t": posts[-1][0],
        "native_clock_grid": 10,
        "windows_clock_extrapolation_claim": False,
        "endpoint_physics6": list(struct.unpack_from("<6f", posts[-1][1], 168)),
        "reader_count": len(readers),
        "qualified_source_repair_audits": len(audits),
        "first_reader": readers[0] if readers else None,
        "last_reader": readers[-1] if readers else None,
        "result_count": len(result_records),
        "last_result_native_t": u32(last, 128),
        "last_result_hex": last.hex(),
    }


class DrainSession(ImmediateSession):
    def remaining(self) -> float:
        now = time.monotonic()
        if now >= getattr(self, "next_budget_scan", 0):
            self.next_budget_scan = now + 1
            self.latest_budget_bytes = sum(
                file.stat().st_size
                for root in (self.out, self.work, self.tmp)
                for file in root.rglob("*")
                if file.is_file()
            )
            if self.latest_budget_bytes > MAX_BYTES:
                raise RuntimeError("explicit drain evidence budget exceeded")
        return cast(float, super().remaining())

    def arm(self) -> None:
        if getattr(self, "suppress_initial_arm", False):
            self.suppress_initial_arm = False
            # The experimental held POST precedes the base session's first reader.
            self.reader = cast(Any, None)
            return
        super().arm()

    def reset_to_post(self) -> bytes:
        self.suppress_initial_arm = True
        result = super().reset()
        if self.seq != 0 or self.epoch != 1 or self.reader is not None:
            raise ValueError("original initial POST must remain held before first ARM")
        return result

    def release_at_post(self) -> dict[str, Any]:
        before, epoch = self.boundary, self.epoch
        self.request(RequestSpec(op=2, n=0, p=0, k=0, befores=(), afters=()))
        same, receipt = self.response(ResponseSpec(kind=2, seq=self.seq, n=0, epoch=epoch))
        if same != before or receipt != bytes(160):
            raise ValueError("literal unchanged-POST release ACK required")
        tail = self.connection.recv(1)
        if tail:
            self.rx.write(tail)
            self.rx.flush()
            raise ValueError("unexpected bytes after explicit release ACK")
        self.connection.close()
        self.connection = None
        return {
            "op": 2,
            "ack_native_t": u32(before, 112),
            "ack_epoch": epoch,
            "intentional_ipc_eof": True,
            "ipc_eof_is_finish": False,
            "future_autonomous_intervals_have_policy_credit": False,
        }

    def repair_source_five(self, source: dict[str, Any]) -> dict[str, Any]:
        self.arm()
        reader, anchor = self.reader, u32(self.boundary, 112)
        p = u32(reader, 112)
        if p != 151 or anchor != 2190:
            raise ValueError("repair applies only to five initial source records151..155")
        original = self.read_mem(u64(reader, 88) - 4 + p * 32, 160)
        befores = [original[index * 32 + 4 : index * 32 + 16] for index in range(5)]
        afters = [bytes.fromhex(raw) for raw in source["five_sgb_hex"]]
        if afters != [struct.pack("<3f", 0, 1, 0)] * 5:
            raise ValueError("source repair must preserve the literal five Jimmy0/1/0 records")
        sequence, epoch = self.seq + 1, self.epoch
        self.request(RequestSpec(op=6, n=5, p=p, k=p, befores=befores, afters=afters))
        committed, receipt = self.response(ResponseSpec(kind=6, seq=sequence, n=5, epoch=epoch))
        expected = struct.pack("<4I", p, p, 5, 2)
        expected += b"".join(before + after for before, after in zip(befores, afters, strict=True))
        expected += bytes(24)
        if committed != reader or receipt != expected:
            raise ValueError("source repair exact current-record receipt mismatch")
        self.boundary, _ = self.response(ResponseSpec(kind=1, seq=sequence, n=5, epoch=epoch + 1))
        if u32(self.boundary, 112) != anchor + 50:
            raise ValueError("source repair must end at literal2240 before next ARM")
        raw = (self.local_posts / f"post-{self.pid}.bin").read_bytes()
        rows = chunks(raw, 256)
        posts = [
            row
            for row in rows
            if row[:8] == b"TMPOST03"
            and u32(row, 148) == 75
            and anchor < u32(row, 112) <= anchor + 50
        ]
        audits = [row for row in rows if row[:8] == b"TMAUD003" and u32(row, 200) == sequence]
        if len(posts) != 5 or len(audits) != 5:
            raise ValueError("five actual source-repair POSTs and audits required")
        for index, (post, audit) in enumerate(zip(posts, audits, strict=True)):
            env.qualify_post(post, self.pid, self.meta)
            gas, brake, steer = struct.unpack_from("<3f", post, 88)
            if (
                u32(post, 112) != anchor + (index + 1) * 10
                or struct.pack("<3f", steer, gas, brake) != afters[index]
            ):
                raise ValueError("source repair actual applied POST differs from source bytes")
            before = original[index * 32 : (index + 1) * 32]
            after = before[:4] + afters[index] + before[16:]
            fields = tuple(
                u32(audit, offset) for offset in (176, 180, 184, 188, 204, 208, 248, 252)
            )
            if fields != (p, p + index, 5, anchor, 1, p, 5, index):
                raise ValueError("source repair audit index/authorization mismatch")
            if (
                tuple(u32(audit, offset) for offset in (8, 12, 16, 20, 212))
                != (256, 2, self.pid, self.pid, 12)
                or u64(audit, 192) != epoch
                or u64(audit, 216) != self.meta["input_base"]
                or u64(audit, 232) != self.meta["input_base"] - 4 + (p + index) * 32
                or u64(audit, 240) != self.meta["input_base"] - 4 + p * 32
            ):
                raise ValueError("source repair audit own identity/epoch/address mismatch")
            if (
                audit[40:72] != before
                or audit[72:104] != after
                or audit[104:136] != reader[152:184]
                or audit[136:168] != reader[152:156] + afters[0] + reader[168:184]
                or audit[168:172] != befores[index][:4]
                or audit[172:176] != afters[index][:4]
            ):
                raise ValueError("source repair changed non-channel/current/shadow bytes")
        return {
            "source": source,
            "n": 5,
            "before_native_t": anchor,
            "after_native_t": anchor + 50,
            "receipt_hex": receipt.hex(),
            "audit_hex": [audit.hex() for audit in audits],
            "actual_post_hex": [post.hex() for post in posts],
            "policy_credit": False,
        }


def source_repair(archive: Path, original: bytes, output: Path) -> dict[str, Any]:
    with archive.open("rb") as stream:
        archive_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if archive_sha != SOURCE_ARCHIVE_SHA256:
        raise ValueError("supplied original source archive hash mismatch")
    with zipfile.ZipFile(archive) as source_zip:
        export, ghost = source_zip.read(SOURCE_EXPORT), source_zip.read(SOURCE_GHOST)
    document = json.loads(export)
    archive_data = document["input_archives"][0]
    packets = archive_data["packets"]
    if len(packets) != 1923 or document["metadata"]["map_uid_literals"] != [MAP_UID]:
        raise ValueError("exact Jimmy17715 source/map binding required")
    mismatch, compared = [], 0
    for index, packet in enumerate(packets):
        if packet["steer_normalized"] is None:
            continue
        desired = struct.pack("<3f", packet["steer_normalized"], packet["accel"], packet["brake"])
        compared += 1
        if original[index * 32 + 4 : index * 32 + 16] != desired:
            mismatch.append(index)
    if mismatch != list(range(151, 156)):
        raise ValueError("expected exactly five altered startup records in the long tape")
    (output / "jimmy-source-export.json").write_bytes(export)
    (output / "jimmy-source.Ghost.Gbx").write_bytes(ghost)
    return {
        "archive_sha256": archive_sha,
        "export_entry": SOURCE_EXPORT,
        "export_sha256": sha(export),
        "ghost_entry": SOURCE_GHOST,
        "ghost_sha256": sha(ghost),
        "packet_count": len(packets),
        "finite_packets_compared": compared,
        "mismatching_indices": mismatch,
        "source_finish_ms": document["metadata"]["result_race_time_ms"],
        "five_sgb_hex": [
            struct.pack(
                "<3f",
                packets[index]["steer_normalized"],
                packets[index]["accel"],
                packets[index]["brake"],
            ).hex()
            for index in mismatch
        ],
    }


def classify_saved_drain(run: Path) -> dict[str, Any]:
    """Feed actual autonomous terminal evidence into the unchanged production classifier."""
    report = json.loads((run / "report.json").read_bytes())
    session_dir = run / run.name
    launch = json.loads((session_dir / "launch-v7.json").read_bytes())
    reaped = report["reaped"]
    episode = next(session_dir.glob("episode-*/metadata.json")).parent
    meta = json.loads((episode / "metadata.json").read_bytes())
    pid = reaped["pid"]
    if pid != meta["server_pid"] or report["policy_credit_intervals"] != 0:
        raise ValueError("actual uncredited drain ownership required")
    posts_raw = (session_dir / "posts" / f"post-{pid}.bin").read_bytes()
    posts = [
        raw for raw in chunks(posts_raw, 256) if raw[:8] == b"TMPOST03" and u32(raw, 148) == 75
    ]
    raw_json = (episode / "stdout-slice.bin").read_bytes()
    prefix = (session_dir / "stdout-prefix.bin").read_bytes()
    envelope = raw_json if raw_json.lstrip().startswith(b"[") else prefix + raw_json
    collector = (session_dir / "posts" / f"result-{pid}.bin").read_bytes()
    profile = codec_v7.PROFILE_ID
    packet = {
        "profile": profile,
        "pid": pid,
        "meta": meta,
        "boundaryHex": posts[-1].hex(),
        "resultEvidence": {
            "profile": profile,
            "pins": launch["pins"],
            "reaped": reaped,
            "validatorArrayHex": envelope.hex(),
            "collectorHex": collector.hex(),
        },
    }
    # Invoke the production method itself; only its readonly context is supplied.
    runtime_root = Path(launch["args"][0])
    for _ in Path(artifacts.ROLE_PATHS["driver"]).parts:
        runtime_root = runtime_root.parent
    if runtime_root / artifacts.ROLE_PATHS["driver"] != Path(launch["args"][0]):
        raise ValueError("actual launch driver must retain its documented runtime-relative role")
    context = SimpleNamespace(
        pins=launch["pins"], settings=SimpleNamespace(runtime_root=runtime_root)
    )
    classified = ProductionImmediateSession.classify_terminal(
        cast(ProductionImmediateSession, context), packet
    )
    return {
        "schema": "tmrl.actual-drain-production-outcome-check/v1",
        "outcome": classified,
        "packet": packet,
        "policy_credit_intervals": 0,
        "is_rl_transition": False,
        "terminal_boundary_source": "last literal autonomous POST; distinct from release ACK",
        "source_report_sha256": sha((run / "report.json").read_bytes()),
        "source_launch_sha256": sha((session_dir / "launch-v7.json").read_bytes()),
        "source_metadata_sha256": sha((episode / "metadata.json").read_bytes()),
        "source_posts_sha256": sha(posts_raw),
        "source_collector_sha256": sha(collector),
        "runtime_root": str(runtime_root),
    }


def main() -> None:
    config_path, receipt_path, output = map(Path, sys.argv[1:4])
    archive_path = Path(sys.argv[4]) if len(sys.argv) == 5 else None
    if len(sys.argv) not in (4, 5):
        raise ValueError("optional supplied source archive is the only fourth argument")
    if output.exists():
        raise FileExistsError("fresh ext4 drain evidence required")
    config, receipt = json.loads(config_path.read_bytes()), json.loads(receipt_path.read_bytes())
    deployment = Path(config["deployment_manifest"]).read_bytes()
    if sha(deployment) != config["deployment_manifest_sha256"]:
        raise ValueError("original deployment manifest changed")
    driver_sha = json.loads(deployment)["driverBuildApproval"]["driverSha256"]
    output.mkdir(parents=True)
    source_bytes = Path(__file__).read_bytes()
    (output / "source-headless_finish_drain.py").write_bytes(source_bytes)
    report: dict[str, Any] = {
        "schema": "tmrl.explicit-original-release-drain/v1",
        "completed": False,
        "learning_executed": False,
        "policy_credit_intervals": 0,
        "production_profile_modified": False,
        "physical_state_writes": False,
        "neutral_reset_applied": False,
        "source_sha256": sha(source_bytes),
        "max_bytes": MAX_BYTES,
    }
    session = None
    started, clean = time.monotonic(), False
    try:
        session = DrainSession(
            Path(config["runtime_root"]), receipt, output / output.name, 2190, driver_sha
        )
        report["initial_post_hex"] = session.reset_to_post().hex()
        original = (session.out / "parent.inputs.bin").read_bytes()
        if session.read_mem(session.meta["input_base"] - 4, len(original)) != original:
            raise ValueError("entire held child tape differs from immutable original")
        report["original_records_sha256"] = sha(original)
        allowed_audits: tuple[bytes, ...] = ()
        if archive_path is not None:
            source = source_repair(archive_path, original, output)
            report["source_five_repair"] = session.repair_source_five(source)
            allowed_audits = tuple(
                bytes.fromhex(raw) for raw in report["source_five_repair"]["audit_hex"]
            )
        report["release"] = session.release_at_post()
        reaped = session.command("reap")
        if not (
            reaped["kind"] == "reaped"
            and reaped["pid"] == session.pid
            and reaped["process_gone"]
            and reaped["parent_unchanged"]
            and reaped["truncated_by_driver"] is False
            and reaped["exit_status"] is None
        ):
            raise ValueError("complete original normal reap proof required")
        session.episode = None
        report["reaped"] = reaped
        child = (session.ep / "stdout-slice.bin").read_bytes()
        prefix = (session.out / "stdout-prefix.bin").read_bytes()
        stdout = child if child.lstrip().startswith(b"[") else prefix + child
        posts = (session.local_posts / f"post-{session.pid}.bin").read_bytes()
        results = (session.local_posts / f"result-{session.pid}.bin").read_bytes()
        report["analysis"] = inspect_native_drain(
            stdout, posts, results, session.pid, session.meta, allowed_audits
        )
        report["evidence_sha256"] = {
            "stdout": sha(stdout),
            "posts": sha(posts),
            "results": sha(results),
        }
        report["completed"] = True
        clean = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        try:
            if session is not None:
                session.close(error=not clean)
                report["cleanup"] = session.cleanup
                if session.cleanup["remainingOwned"]:
                    raise RuntimeError("drain left an owned process")
        finally:
            report["wall_seconds_not_a_benchmark"] = time.monotonic() - started
            with (output / "report.json").open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
                stream.write("\n")


if __name__ == "__main__":
    main()
