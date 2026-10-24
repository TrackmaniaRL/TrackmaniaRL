"""Diagnostic only: inspect native boundaries before administrative warmup.

Never admit these rows to training or relabel the native clock as race time.
"""

import json
import struct
import sys
import time
from pathlib import Path

from trackmaniarl.trackmania.native_headless._runtime import host, ownership
from trackmaniarl.trackmania.native_headless._runtime.features import native16
from trackmaniarl.trackmania.native_headless._runtime.session import make_native_env
from trackmaniarl.trackmania.native_headless.assets import verify_assets
from trackmaniarl.trackmania.native_headless.config import NativeRuntimeConfig


def main() -> None:
    config_path, output = map(Path, sys.argv[1:3])
    output = output.absolute()
    output.mkdir(mode=0o700)
    config = json.loads(config_path.read_bytes())
    settings = NativeRuntimeConfig.from_mapping(
        {
            **config,
            "trace_dir": str(output / "trace"),
            "work_dir": str(Path(config["work_dir"]) / output.name),
            "allow_native": True,
        }
    )
    native = make_native_env(settings, verify_assets(settings))
    session = None
    report = {
        "passed": False,
        "diagnostic_only": True,
        "rows": [],
        "clock_mapping_qualified": False,
    }
    started = time.monotonic()
    try:
        session = native.factory()
        # Intentionally stop before the wrapper's administrative prelude.
        host.NativeSessionV6.reset(session, 0)
        session.number = 0
        session.post_offset = 0
        session.events, session.audits, session.hits = {}, [], set()
        session.pending_provenance = {"source": "diagnostic raw spawn", "actionId": None}
        session.poll_posts()
        report["metadata"] = session.meta
        for ordinal in range(12):
            packet = session.packet()
            raw = bytes.fromhex(packet["boundaryHex"])
            reader = bytes.fromhex(packet["readerHex"])
            report["rows"].append(
                {
                    "native_t": struct.unpack_from("<I", raw, 112)[0],
                    "own_p": struct.unpack_from("<I", reader, 112)[0],
                    "current_record_time": struct.unpack_from("<i", reader, 152)[0],
                    "record_sgb": list(struct.unpack_from("<3f", reader, 156)),
                    "reader_fields": {
                        str(o): struct.unpack_from("<i", reader, o)[0]
                        for o in (120, 124, 144, 248, 252)
                    },
                    "state": native16(host.state76(raw)),
                }
            )
            if ordinal < 11:
                session._advance(36, 1)
        report["passed"] = True
    finally:
        if session is not None:
            session.close()
            report["remaining_owned"] = [
                v
                for v in session.known_owned.values()
                if ownership.same_identity(v, ownership.process_identity(v["pid"]))
            ]
            assert not report["remaining_owned"]
        report["host_seconds"] = time.monotonic() - started
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("passed", "host_seconds", "remaining_owned")}))


if __name__ == "__main__":
    main()
