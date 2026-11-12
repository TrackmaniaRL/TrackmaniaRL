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


def test_experimental_listener_survives_sequential_clients() -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "trackmaniarl/project/openplanet/TrackmaniaRL_Connect.as"
    ).read_text(encoding="utf-8")
    server = source.split("void AttestedTelemetryServer()", 1)[1].split("void Main()", 1)[0]
    listen = server.index('Listen("127.0.0.1", ATTESTED_TELEMETRY_PORT)')
    accept_loop = server.index("        while (g_enableMapSwitch) {", listen)
    accept = server.index("g_attestedTelemetryServer.Accept();", accept_loop)
    assert listen < accept_loop < accept
    # The client is closed inside the accept loop; the listener is closed only
    # after that loop exits. This is a source check, not an Openplanet run.
    assert (
        "                sock.Close();\n"
        "            }\n"
        "        }\n"
        "        g_attestedTelemetryServer.Close();"
    ) in server


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
    assert "if (!IsHexToken(requestId) || (waitReady && !firstRequest))" in server
    assert 'parsed["wait_ready"].GetType() != Json::Type::Boolean' in server
    assert "while (waitReady && !IsReadyToDrive(true)" in server
    assert "sock, 75000, true, streamInstance, streamOperation, streamUid" in server
    reader = source.split("string ReadJsonLine(", 1)[1].split("void HandleSessionClient(", 1)[0]
    assert "!BoundTelemetryIdentity(instance, operation, uid)" in reader
    assert 'header["payload_bytes"] = 132;' in server
    assert 'sock.WriteRaw(Json::Write(header) + "\\n") || !sock.Write(buf)' in server
    failed_collect = server.split("if (!CollectTelemetry(buf, true)) {", 1)[1].split("}", 1)[0]
    assert "BoundTelemetryIdentity(streamInstance, streamOperation, streamUid);" in failed_collect
    assert "sock.Close(); break;" in failed_collect
    critical = server.split("string instance = g_switchInstanceEpoch;", 1)[1].split(
        "if (!sock.WriteRaw(", 1
    )[0]
    assert "yield(" not in critical


def test_local_view_requirement_is_scoped_to_protocol_three() -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "trackmaniarl/project/openplanet/TrackmaniaRL_Connect.as"
    ).read_text(encoding="utf-8")
    readiness = source.split("bool IsReadyToDrive(", 1)[1].split("string JsonError(", 1)[0]
    assert "bool requireLocalView = false" in readiness
    assert "!requireLocalView || VehicleState::GetViewingPlayer() is player" in readiness
    collect = source.split("bool CollectTelemetry(", 1)[1].split(
        "void AttestedTelemetryServer()", 1
    )[0]
    assert "bool requireLocalView = false" in collect
    assert "requireLocalView && VehicleState::GetViewingPlayer() !is player" in collect
    session = source.split("void HandleSessionClient(", 1)[1].split("void SessionServer()", 1)[0]
    assert "if (!IsReadyToDrive())" in session
    server = source.split("void AttestedTelemetryServer()", 1)[1].split("void Main()", 1)[0]
    assert "CollectTelemetry(buf, true)" in server
    assert server.count("IsReadyToDrive(true)") >= 4
    legacy = source.split("void Main()", 1)[1]
    assert "CollectTelemetry(buf)" in legacy


def test_observed_map_mismatch_persists_across_telemetry_reconnects() -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "trackmaniarl/project/openplanet/TrackmaniaRL_Connect.as"
    ).read_text(encoding="utf-8")
    identity = source.split("bool BoundTelemetryIdentity(", 1)[1].split("string ReadJsonLine(", 1)[
        0
    ]
    # A mismatch poisons the current ready operation, so returning to the old
    # UID cannot make a fresh connection accept that operation again.
    mismatch = identity.split("if (ActiveMapUid() != uid) {", 1)[1].split("}", 1)[0]
    assert 'g_switchState = "indeterminate";' in mismatch
    assert "return false;" in mismatch
    guard = identity.split("if (ActiveMapUid() != uid)", 1)[0]
    for current in (
        "!g_enableMapSwitch",
        'g_switchState != "ready"',
        "g_switchInstanceEpoch != instance",
        "g_switchOperation != operation",
        "g_switchUid != uid",
    ):
        assert current in guard
    assert "return false;" in guard
    server = source.split("void AttestedTelemetryServer()", 1)[1].split("void Main()", 1)[0]
    # Observe the mismatch while idle, waiting for the player, and before and
    # after collection. These source checks do not execute the game engine.
    assert "sock, 75000, true, streamInstance, streamOperation, streamUid" in server
    assert server.count("BoundTelemetryIdentity(streamInstance, streamOperation, streamUid)") >= 4

    reply = source.split("string SwitchReply(", 1)[1].split("void SwitchMapWorker(", 1)[0]
    observation = reply.index('status == "ok" && g_enableMapSwitch && g_switchState == "ready"')
    mismatch = reply.index('uid != g_switchUid) g_switchState = "indeterminate";')
    state_response = reply.index('response["state"] = g_switchState;')
    assert observation < mismatch < state_response


def test_protocol_two_observation_quarantines_only_opted_in_switching() -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "trackmaniarl/project/openplanet/TrackmaniaRL_Connect.as"
    ).read_text(encoding="utf-8")
    handler = source.split("void HandleSessionClient(", 1)[1].split("void SessionServer(", 1)[0]
    observation = handler.index("string uid = ActiveMapUid();")
    guard = handler.index('g_enableMapSwitch && g_switchState == "ready" && uid != g_switchUid')
    quarantine = handler.index('g_switchState = "indeterminate";', guard)
    missing = handler.index('JsonError("no_local_map_loaded")', quarantine)
    assert observation < guard < quarantine < missing
    assert "sock.WriteRaw(JsonOk(uid, false)); return;" in handler
    assert "sock.WriteRaw(JsonOk(uid, true)); return;" in handler
