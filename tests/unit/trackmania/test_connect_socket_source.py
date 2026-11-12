"""Static checks for the checked-in Connect reference; not an in-game test."""

from pathlib import Path


def test_session_reader_drains_buffer_before_treating_hangup_as_end() -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "trackmaniarl/project/openplanet/TrackmaniaRL_Connect.as"
    ).read_text(encoding="utf-8")
    reader = source.split("string ReadJsonLine(", 1)[1].split("void HandleSessionClient(", 1)[0]
    # Openplanet documents that a hangup can coexist with buffered data.
    # A complete JSONL command must be drained before EOF is classified.
    available = reader.index("int available = sock.Available();")
    empty = reader.index("if (available <= 0)")
    hangup = reader.index("if (sock.IsHungUp()) return")
    assert available < empty < hangup
    assert "if (sock is null || sock.IsHungUp())" not in reader
    assert "if (newline + 1 > int(MAX_COMMAND_BYTES))" in reader


def test_experimental_telemetry_pairs_identity_with_legacy_payload() -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "trackmaniarl/project/openplanet/TrackmaniaRL_Connect.as"
    ).read_text(encoding="utf-8")
    server = source.split("void AttestedTelemetryServer()", 1)[1].split("void Main()", 1)[0]
    assert 'Listen("127.0.0.1", ATTESTED_TELEMETRY_PORT)' in server
    assert "if (!g_enableMapSwitch)" in server
    for field in (
        "protocol_version",
        "request_id",
        "instance_epoch",
        "operation_id",
        "map_uid",
        "state",
    ):
        assert f'header["{field}"]' in server
    assert 'string(parsed["command"]) != "frame"' in server
    assert "if (!IsHexToken(requestId))" in server
    assert "sock, 75000, true, streamInstance, streamOperation, streamUid" in server
    reader = source.split("string ReadJsonLine(", 1)[1].split("void HandleSessionClient(", 1)[0]
    assert "!BoundTelemetryCurrent(instance, operation, uid)" in reader
    assert 'header["payload_bytes"] = 132;' in server
    assert 'sock.WriteRaw(Json::Write(header) + "\\n") || !sock.Write(buf)' in server
    assert "if (!CollectTelemetry(buf)) { sock.Close(); break; }" in server
    critical = server.split("string instance = g_switchInstanceEpoch;", 1)[1].split(
        "if (!sock.WriteRaw(", 1
    )[0]
    assert "yield(" not in critical
