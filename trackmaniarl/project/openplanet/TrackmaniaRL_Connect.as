// Developer reference snapshot of TrackmaniaRL Connect / SAC_GetData 2.4.0.
// Normal users install the signed plugin through Openplanet's Plugin Manager.
// It serves protocol 2 on 127.0.0.1:9000 (33 float32 values) and 9001 (JSONL).

const uint TELEMETRY_PORT = 9000;
const uint SESSION_PORT = 9001;
const uint ATTESTED_TELEMETRY_PORT = 9002;
const string PROTOCOL_VERSION = "2";
const uint MAX_COMMAND_BYTES = 16 * 1024;

// Experimental developer protocol.  The signed 2.4.0 release does not expose
// this capability. Enable only in an isolated, exclusively controlled game.
[Setting name="Enable experimental map switching"]
bool g_enableMapSwitch = false;
const string SWITCH_PROTOCOL_VERSION = "3";
const uint SWITCH_TIMEOUT_MS = 30000;
string g_switchEpoch = "";
string g_switchInstanceEpoch = "";
string g_switchOperation = "";
string g_switchPath = "";
string g_switchUid = "";
string g_switchState = "idle";
array<string> g_usedSwitchOperations;

Net::Socket@ g_telemetryServer;
Net::Socket@ g_sessionServer;
Net::Socket@ g_attestedTelemetryServer;

void OnDestroy() {
    if (g_telemetryServer !is null) g_telemetryServer.Close();
    if (g_sessionServer !is null) g_sessionServer.Close();
    if (g_attestedTelemetryServer !is null) g_attestedTelemetryServer.Close();
}

void AppendBool(MemoryBuffer@ buf, bool value) { buf.Write(value ? 1.0f : 0.0f); }
void AppendFloat(MemoryBuffer@ buf, float value) { buf.Write(value); }
void AppendInt(MemoryBuffer@ buf, int32 value) { buf.Write(float(value)); }
void AppendVec3(MemoryBuffer@ buf, vec3 value) {
    buf.Write(value.x); buf.Write(value.y); buf.Write(value.z);
}

float RaceTimeMs(CSmScriptPlayer@ player) {
    if (player is null || player.StartTime <= 0) return 0.0f;
    CTrackMania@ app = cast<CTrackMania>(GetApp());
    if (app is null) return 0.0f;
    CSmArenaRulesMode@ rules = cast<CSmArenaRulesMode>(app.PlaygroundScript);
    int64 now = rules is null
        ? int64(app.Network.PlaygroundClientScriptAPI.GameTime)
        : int64(rules.Now);
    return now > player.StartTime ? float(now - player.StartTime) : 0.0f;
}

string ActiveMapUid() {
    CTrackMania@ app = cast<CTrackMania>(GetApp());
    if (app is null) return "";
    CSmArenaClient@ playground = cast<CSmArenaClient>(app.CurrentPlayground);
    if (playground is null || playground.Map is null || playground.Map.MapInfo is null) return "";
    return playground.Map.MapInfo.MapUid;
}

bool IsReadyToDrive() {
    CTrackMania@ app = cast<CTrackMania>(GetApp());
    if (app is null) return false;
    CSmArenaClient@ playground = cast<CSmArenaClient>(app.CurrentPlayground);
    if (playground is null || playground.GameTerminals.Length == 0) return false;
    auto player = cast<CSmPlayer>(playground.GameTerminals[0].GUIPlayer);
    return player !is null && cast<CSmScriptPlayer>(player.ScriptAPI) !is null
        && VehicleState::GetViewingPlayer() is player
        && VehicleState::ViewingPlayerState() !is null;
}

string JsonError(const string &in code) {
    return "{\"status\":\"error\",\"protocol_version\":\"" + PROTOCOL_VERSION
        + "\",\"error\":\"" + code + "\"}\n";
}

string JsonOk(const string &in uid, bool ready) {
    return "{\"status\":\"ok\",\"protocol_version\":\"" + PROTOCOL_VERSION
        + "\",\"map_uid\":\"" + uid + "\",\"ready\":\""
        + (ready ? "true" : "false") + "\"}\n";
}

bool IsHexToken(const string &in value) {
    if (value.Length != 32) return false;
    for (int i = 0; i < int(value.Length); i++) {
        if ("0123456789abcdef".IndexOf(value.SubStr(i, 1)) < 0) return false;
    }
    return true;
}

bool HasString(Json::Value@ value, const string &in key) {
    return value.HasKey(key) && value[key].GetType() == Json::Type::String;
}

string SwitchReply(const string &in status, const string &in error = "") {
    string uid = ActiveMapUid();
    // A successful status or claim also observes the active UID. Retain a
    // mismatch for this operation even if the expected map later returns.
    if (status == "ok" && g_enableMapSwitch && g_switchState == "ready"
        && uid != g_switchUid) g_switchState = "indeterminate";
    auto response = Json::Object();
    response["status"] = status;
    response["protocol_version"] = SWITCH_PROTOCOL_VERSION;
    // Do not disclose lease or operation tokens to a client that failed a check.
    response["instance_epoch"] = g_switchInstanceEpoch;
    response["epoch"] = status == "ok" ? g_switchEpoch : "";
    response["operation_id"] = status == "ok" ? g_switchOperation : "";
    response["state"] = g_switchState;
    if (status == "ok") response["remaining_operations"] = 4096 - int(g_usedSwitchOperations.Length);
    response["map_uid"] = uid;
    response["ready"] = (uid.Length > 0 && IsReadyToDrive()) ? "true" : "false";
    if (error.Length > 0) response["error"] = error;
    return Json::Write(response) + "\n";
}

// At most one coroutine can be launched by a claimed plugin epoch. Any error
// after the operation is accepted is indeterminate, including a timeout after
// PlayMap: the game may complete an asynchronous request later.
void SwitchMapWorker() {
    try {
        auto app = cast<CTrackMania>(GetApp());
        if (app is null || app.ManiaTitleControlScriptAPI is null || !Permissions::PlayLocalMap()) {
            g_switchState = "indeterminate"; return;
        }
        g_switchState = "leaving";
        app.BackToMainMenu();
        uint64 started = Time::Now;
        while (Time::Now - started < SWITCH_TIMEOUT_MS) {
            if (app.ManiaTitleControlScriptAPI.IsReady && app.Switcher.ModuleStack.Length > 0
                && cast<CTrackManiaMenus>(app.Switcher.ModuleStack[0]) !is null
                && ActiveMapUid().Length == 0) break;
            yield();
        }
        if (Time::Now - started >= SWITCH_TIMEOUT_MS) {
            g_switchState = "indeterminate"; return;
        }
        g_switchState = "loading";
        app.ManiaTitleControlScriptAPI.PlayMap(g_switchPath, "", "");
        started = Time::Now;
        uint stableFrames = 0;
        CSmArenaClient@ lastPlayground = null;
        while (Time::Now - started < SWITCH_TIMEOUT_MS) {
            auto playground = cast<CSmArenaClient>(app.CurrentPlayground);
            if (playground !is null && playground is lastPlayground
                && ActiveMapUid() == g_switchUid && IsReadyToDrive()) {
                stableFrames++;
            } else {
                @lastPlayground = playground;
                stableFrames = 0;
            }
            if (stableFrames >= 3) {
                // No more game mutations in this operation after publishing ready.
                g_switchState = "ready"; return;
            }
            yield();
        }
    } catch {
        // Do not reissue a request whose side effects may already be in flight.
    }
    g_switchState = "indeterminate";
}

string HandleSwitchCommand(Json::Value@ request) {
    if (!g_enableMapSwitch) return SwitchReply("error", "switching_disabled");
    if (!HasString(request, "command")) return SwitchReply("error", "invalid_request");
    string command = string(request["command"]);
    if (command == "get_instance") {
        if (request.GetKeys().Length != 2) return SwitchReply("error", "invalid_request");
        if (g_switchInstanceEpoch.Length == 0) return SwitchReply("error", "instance_unavailable");
        auto response = Json::Object();
        response["status"] = "ok";
        response["protocol_version"] = SWITCH_PROTOCOL_VERSION;
        response["instance_epoch"] = g_switchInstanceEpoch;
        return Json::Write(response) + "\n";
    }
    if (!HasString(request, "epoch") || !HasString(request, "instance_epoch"))
        return SwitchReply("error", "invalid_request");
    if (g_switchInstanceEpoch.Length == 0
        || string(request["instance_epoch"]) != g_switchInstanceEpoch)
        return SwitchReply("error", "instance_epoch_mismatch");
    string epoch = string(request["epoch"]);
    if (!IsHexToken(epoch)) return SwitchReply("error", "invalid_epoch");
    if (command == "claim") {
        if (request.GetKeys().Length != 4) return SwitchReply("error", "invalid_request");
        if (g_switchEpoch.Length == 0) g_switchEpoch = epoch;
        if (g_switchEpoch != epoch) return SwitchReply("error", "epoch_mismatch");
        return SwitchReply("ok");
    }
    if (g_switchEpoch.Length == 0 || g_switchEpoch != epoch)
        return SwitchReply("error", "epoch_mismatch");
    if (!HasString(request, "operation_id")) return SwitchReply("error", "invalid_request");
    string operation = string(request["operation_id"]);
    if (!IsHexToken(operation)) return SwitchReply("error", "invalid_operation");
    if (command == "switch_status") {
        if (request.GetKeys().Length != 5) return SwitchReply("error", "invalid_request");
        if (operation != g_switchOperation) return SwitchReply("error", "unknown_operation");
        return SwitchReply("ok");
    }
    if (command != "switch_map" || request.GetKeys().Length != 7
        || !HasString(request, "map_path") || !HasString(request, "expected_map_uid"))
        return SwitchReply("error", "invalid_request");
    string path = string(request["map_path"]);
    string uid = string(request["expected_map_uid"]);
    if (path.Length == 0 || path.Length > 2048 || uid.Length == 0 || uid.Length > 256)
        return SwitchReply("error", "invalid_target");
    if (operation == g_switchOperation) {
        if (path != g_switchPath || uid != g_switchUid)
            return SwitchReply("error", "operation_conflict");
        return SwitchReply("ok");
    }
    // Retain every used ID for the lease lifetime. A late retry of an older
    // operation must never initiate another game load.
    if (g_usedSwitchOperations.Find(operation) >= 0)
        return SwitchReply("error", "stale_operation");
    if (g_switchState != "idle" && g_switchState != "ready")
        return SwitchReply("error", "operation_pending");
    // A manual change or another plugin invalidates the previous confirmation.
    if (g_switchState == "ready" && ActiveMapUid() != g_switchUid) {
        g_switchState = "indeterminate";
        return SwitchReply("error", "active_map_changed");
    }
    if (g_usedSwitchOperations.Length >= 4096)
        return SwitchReply("error", "operation_limit");
    g_usedSwitchOperations.InsertLast(operation);
    g_switchOperation = operation; g_switchPath = path; g_switchUid = uid;
    g_switchState = "accepted";
    try { startnew(SwitchMapWorker); }
    catch { g_switchState = "indeterminate"; }
    return SwitchReply("ok");
}

bool BoundTelemetryIdentity(
    const string &in instance, const string &in operation, const string &in uid
) {
    if (!g_enableMapSwitch || instance.Length == 0 || operation.Length == 0 || uid.Length == 0
        || g_switchState != "ready" || g_switchInstanceEpoch != instance
        || g_switchOperation != operation || g_switchUid != uid) return false;
    // Persist an observed mismatch for this operation, including across socket
    // closes and reconnects. Never poison a newer operation or an intentional stop.
    if (ActiveMapUid() != uid) {
        g_switchState = "indeterminate";
        return false;
    }
    return true;
}

string ReadJsonLine(
    Net::Socket@ sock, uint timeoutMs = 2000, bool requireSwitchEnabled = false,
    const string &in instance = "", const string &in operation = "", const string &in uid = ""
) {
    string request = "";
    uint started = Time::Now;
    while (Time::Now - started < timeoutMs) {
        if (sock is null || (requireSwitchEnabled && !BoundTelemetryIdentity(instance, operation, uid)))
            return "";
        int available = sock.Available();
        if (available <= 0) {
            // Openplanet can report hangup while unread bytes remain buffered.
            // Drain those bytes first so a complete half-closed line is valid.
            if (sock.IsHungUp()) return "";
            yield(); continue;
        }
        // ReadLine keeps an internal application buffer. Checking Available()
        // alone does not bound a line assembled across multiple reads. Keep the
        // accumulator here and never read more than the limit plus one byte.
        int remaining = int(MAX_COMMAND_BYTES) - int(request.Length);
        int take = available < remaining + 1 ? available : remaining + 1;
        request += sock.ReadRaw(take);
        int newline = request.IndexOf("\n");
        if (newline >= 0) {
            if (newline + 1 > int(MAX_COMMAND_BYTES)) return "";
            return request.SubStr(0, newline + 1);
        }
        if (int(request.Length) > int(MAX_COMMAND_BYTES)) return "";
        yield();
    }
    return "";
}

void HandleSessionClient(Net::Socket@ sock) {
    string request = ReadJsonLine(sock);
    if (request.Length == 0) { sock.WriteRaw(JsonError("invalid_or_too_large_request")); return; }
    Json::Value@ parsed = null;
    try { @parsed = Json::Parse(request); } catch { }
    if (parsed is null || parsed.GetType() != Json::Type::Object) {
        sock.WriteRaw(JsonError("invalid_request")); return;
    }
    if (HasString(parsed, "protocol_version")
        && string(parsed["protocol_version"]) == SWITCH_PROTOCOL_VERSION) {
        sock.WriteRaw(HandleSwitchCommand(parsed)); return;
    }
    if (request.IndexOf("\"protocol_version\":\"" + PROTOCOL_VERSION + "\"") < 0) {
        sock.WriteRaw(JsonError("unsupported_protocol")); return;
    }
    if (g_switchState != "idle" && g_switchState != "ready") {
        sock.WriteRaw(JsonError("map_switch_unresolved")); return;
    }
    string uid = ActiveMapUid();
    if (uid.Length == 0) { sock.WriteRaw(JsonError("no_local_map_loaded")); return; }
    if (request.IndexOf("\"command\":\"verify_loaded_map\"") >= 0) {
        sock.WriteRaw(JsonOk(uid, false)); return;
    }
    if (request.IndexOf("\"command\":\"confirm_ready\"") >= 0) {
        if (!IsReadyToDrive()) { sock.WriteRaw(JsonError("player_not_ready")); return; }
        sock.WriteRaw(JsonOk(uid, true)); return;
    }
    sock.WriteRaw(JsonError("unknown_command"));
}

void SessionServer() {
    while (true) {
        @g_sessionServer = Net::Socket();
        if (!g_sessionServer.Listen("127.0.0.1", SESSION_PORT)) { yield(120); continue; }
        while (true) {
            yield();
            Net::Socket@ sock = g_sessionServer.Accept();
            if (sock is null) continue;
            HandleSessionClient(sock);
            sock.Close();
        }
    }
}

// Collect one legacy 33-float payload without yielding. This does not claim
// the game engine's fields form an atomic snapshot.
bool CollectTelemetry(MemoryBuffer@ buf) {
    CTrackMania@ app = cast<CTrackMania>(GetApp());
    CSmArenaClient@ playground = app is null ? null : cast<CSmArenaClient>(app.CurrentPlayground);
    if (playground is null || playground.GameTerminals.Length == 0) return false;
    auto terminal = playground.GameTerminals[0];
    auto player = cast<CSmPlayer>(terminal.GUIPlayer);
    CSmScriptPlayer@ api = player is null ? null : cast<CSmScriptPlayer>(player.ScriptAPI);
    CSceneVehicleVisState@ vis = VehicleState::ViewingPlayerState();
    // Do not combine a spectated vehicle with the local player's telemetry.
    if (api is null || VehicleState::GetViewingPlayer() !is player || vis is null) return false;

    auto raceData = PlayerState::GetRaceData();
    float raceTime = RaceTimeMs(api);
    bool driving = raceTime > 0.0f;
    bool finished = terminal.UISequence_Current == CGamePlaygroundUIConfig::EUISequence::Finish;
    buf.Seek(0, 0);
    AppendFloat(buf, driving ? float(raceData.dPlayerInfo.NumberOfCheckpointsPassed) : 0.0f);
    AppendFloat(buf, driving ? float(raceData.dPlayerInfo.CurrentLapNumber) : 0.0f);
    AppendBool(buf, finished);
    AppendFloat(buf, raceTime);
    AppendVec3(buf, api.Position); AppendVec3(buf, api.Velocity);
    AppendVec3(buf, vis.Dir); AppendVec3(buf, vis.Up);
    AppendFloat(buf, api.Speed); AppendFloat(buf, api.EngineRpm); AppendInt(buf, api.EngineCurGear);
    AppendFloat(buf, vis.FLSlipCoef); AppendFloat(buf, vis.FRSlipCoef);
    AppendFloat(buf, vis.RLSlipCoef); AppendFloat(buf, vis.RRSlipCoef);
    AppendInt(buf, uint(vis.FLGroundContactMaterial)); AppendInt(buf, uint(vis.FRGroundContactMaterial));
    AppendInt(buf, uint(vis.RLGroundContactMaterial)); AppendInt(buf, uint(vis.RRGroundContactMaterial));
    AppendInt(buf, api.WheelsSkiddingCount); AppendInt(buf, api.FlyingDuration);
    AppendFloat(buf, api.AdherenceCoef);
    AppendFloat(buf, api.InputSteer); AppendFloat(buf, api.InputGasPedal); AppendBool(buf, api.InputIsBraking);
    buf.Seek(0, 0);
    return true;
}

// Request/response telemetry is opt-in and separate from protocol 2. Each
// request yields one header and one payload, preventing an unsent frame FIFO.
// Socket writes may still fail; the receiver must reject incomplete records.
void AttestedTelemetryServer() {
    while (true) {
        if (!g_enableMapSwitch) { yield(120); continue; }
        @g_attestedTelemetryServer = Net::Socket();
        if (!g_attestedTelemetryServer.Listen("127.0.0.1", ATTESTED_TELEMETRY_PORT)) {
            yield(120); continue;
        }
        Net::Socket@ sock = null;
        while (g_enableMapSwitch && sock is null) {
            yield(); @sock = g_attestedTelemetryServer.Accept();
        }
        if (sock !is null) {
            MemoryBuffer@ buf = MemoryBuffer(0);
            // A connection belongs to exactly one switch operation, including
            // while waiting between frame requests.
            string streamInstance = g_switchInstanceEpoch;
            string streamOperation = g_switchOperation;
            string streamUid = g_switchUid;
            bool firstRequest = true;
            while (g_enableMapSwitch) {
                string request = ReadJsonLine(
                    sock, 75000, true, streamInstance, streamOperation, streamUid
                );
                Json::Value@ parsed = null;
                try { @parsed = Json::Parse(request); } catch { }
                if (parsed is null || parsed.GetType() != Json::Type::Object
                    || parsed.GetKeys().Length != 3 || !HasString(parsed, "command")
                    || string(parsed["command"]) != "frame" || !HasString(parsed, "request_id")
                    || !parsed.HasKey("wait_ready")
                    || parsed["wait_ready"].GetType() != Json::Type::Boolean) {
                    sock.Close(); break;
                }
                string requestId = string(parsed["request_id"]);
                bool waitReady = bool(parsed["wait_ready"]);
                if (!IsHexToken(requestId) || (waitReady && !firstRequest)) {
                    sock.Close(); break;
                }
                // Only the first reset request may wait for the local player.
                // The same socket and switch identity are retained throughout.
                uint64 readyStarted = Time::Now;
                while (waitReady && !IsReadyToDrive() && Time::Now - readyStarted < 75000) {
                    if (!BoundTelemetryIdentity(streamInstance, streamOperation, streamUid)
                        || sock.IsHungUp()) break;
                    yield();
                }
                if (!BoundTelemetryIdentity(streamInstance, streamOperation, streamUid)
                    || !IsReadyToDrive()) {
                    sock.Close(); break;
                }
                string instance = g_switchInstanceEpoch;
                string operation = g_switchOperation;
                string uid = ActiveMapUid();
                if (!BoundTelemetryIdentity(streamInstance, streamOperation, streamUid)
                    || instance.Length == 0 || operation.Length == 0 || uid.Length == 0
                    || !IsReadyToDrive()) {
                    // Do not hide a map or session transition inside a live
                    // stream: the receiver must establish a new connection.
                    sock.Close(); break;
                }
                if (!CollectTelemetry(buf)) {
                    BoundTelemetryIdentity(streamInstance, streamOperation, streamUid);
                    sock.Close(); break;
                }
                // Do not emit a pair if our own switch identity changed while
                // collecting it. This is not a byte-level attestation of a map.
                if (!BoundTelemetryIdentity(streamInstance, streamOperation, streamUid)
                    || !IsReadyToDrive()) {
                    sock.Close(); break;
                }
                auto header = Json::Object();
                header["protocol_version"] = SWITCH_PROTOCOL_VERSION;
                header["request_id"] = requestId;
                header["instance_epoch"] = instance;
                header["operation_id"] = operation;
                header["map_uid"] = uid;
                header["state"] = "ready";
                header["payload_bytes"] = 132;
                if (!sock.WriteRaw(Json::Write(header) + "\n") || !sock.Write(buf)) break;
                firstRequest = false;
                yield();
            }
            sock.Close();
        }
        g_attestedTelemetryServer.Close();
    }
}

void Main() {
    // A new random identity is required on every plugin load. A client must
    // quarantine any unresolved operation on a changed or missing identity.
    try { g_switchInstanceEpoch = Crypto::RandomBase64(32, true); }
    catch { g_switchInstanceEpoch = ""; }
    startnew(SessionServer);
    startnew(AttestedTelemetryServer);
    while (true) {
        @g_telemetryServer = Net::Socket();
        if (!g_telemetryServer.Listen("127.0.0.1", TELEMETRY_PORT)) { yield(120); continue; }
        Net::Socket@ sock = null;
        while (sock is null) { yield(); @sock = g_telemetryServer.Accept(); }
        MemoryBuffer@ buf = MemoryBuffer(0);
        while (!sock.IsHungUp()) {
            if (!sock.IsReady()) { yield(); continue; }
            if (!CollectTelemetry(buf)) { yield(); continue; }
            if (!sock.Write(buf)) break;
            yield();
        }
        sock.Close(); g_telemetryServer.Close();
    }
}
