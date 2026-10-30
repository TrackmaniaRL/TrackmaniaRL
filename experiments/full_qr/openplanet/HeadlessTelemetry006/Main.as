// Independent request/response diagnostic. No controls, settings or code hooks.
// The payload builder preserves the repository Connect 2.4.0 field order.
const string TELEMETRY006_PROTOCOL = "tmrl.fresh-telemetry/v6";
const string TELEMETRY006_SCHEMA = "tmrl.connect33-float32/v2";
const uint TELEMETRY006_BYTES = 132;
const uint TELEMETRY006_MAX_REQUESTS = 4096;
const uint64 TELEMETRY006_LIFETIME_MS = 120000;
const uint64 TELEMETRY006_IDLE_MS = 5000;
Net::Socket@ g_Telemetry006Server;
Net::Socket@ g_Telemetry006Client;
bool g_Telemetry006Shutdown = false;
uint g_Telemetry006Sequence = 0;
uint64 g_Telemetry006Deadline = 0;
uint64 g_Telemetry006IdleDeadline = 0;
string g_Telemetry006Nonce = "";

void Telemetry006Drop() {
    if (g_Telemetry006Client !is null) {
        try { g_Telemetry006Client.Close(); } catch { }
        @g_Telemetry006Client = null;
    }
}

void Telemetry006Stop() {
    g_Telemetry006Shutdown = true;
    Telemetry006Drop();
    if (g_Telemetry006Server !is null) {
        try { g_Telemetry006Server.Close(); } catch { }
        @g_Telemetry006Server = null;
    }
}
void OnDisabled() { Telemetry006Stop(); }
void OnDestroyed() { Telemetry006Stop(); }
void OnDestroy() { Telemetry006Stop(); }

Json::Value@ Telemetry006Header(const string &in status) {
    auto row = Json::Object();
    row["protocol"] = TELEMETRY006_PROTOCOL;
    row["collector_version"] = "0.0.6";
    row["status"] = status;
    row["nonce"] = g_Telemetry006Nonce;
    row["sequence"] = g_Telemetry006Sequence;
    row["field_schema"] = TELEMETRY006_SCHEMA;
    row["encoding"] = "little-endian-float32";
    row["payload_bytes"] = 0;
    row["field_count"] = 0;
    row["producer_clock"] = "Openplanet.Time.Now/raw";
    row["simulation_phase_qualified"] = false;
    row["applied_input_acknowledgement"] = false;
    row["atomic_snapshot_claimed"] = false;
    row["openplanet_version"] = Meta::OpenplanetVersion();
    row["openplanet_git_commit"] = Meta::OpenplanetVersionGitCommit();
    return row;
}

bool Telemetry006Send(Json::Value@ header) {
    return g_Telemetry006Client !is null &&
        g_Telemetry006Client.WriteRaw(Json::Write(header) + "\n");
}

void Telemetry006Vec3(MemoryBuffer@ buf, vec3 value) {
    buf.Write(value.x); buf.Write(value.y); buf.Write(value.z);
}
void Telemetry006Int(MemoryBuffer@ buf, int32 value) { buf.Write(float(value)); }
void Telemetry006Bool(MemoryBuffer@ buf, bool value) { buf.Write(value ? 1.0f : 0.0f); }

float Telemetry006RaceTime(CTrackMania@ app, CSmScriptPlayer@ player) {
    // Connect's defined pre-start race-time representation is zero. It is not
    // used as a replacement for a missing player or missing clock source.
    if (player.StartTime <= 0) return 0.0f;
    CSmArenaRulesMode@ rules = cast<CSmArenaRulesMode>(app.PlaygroundScript);
    int64 now;
    if (rules !is null) now = int64(rules.Now);
    else {
        if (app.Network is null || app.Network.PlaygroundClientScriptAPI is null)
            throw("race-clock-unavailable");
        now = int64(app.Network.PlaygroundClientScriptAPI.GameTime);
    }
    return now > player.StartTime ? float(now - player.StartTime) : 0.0f;
}

MemoryBuffer@ Telemetry006Build(Json::Value@ header) {
    CTrackMania@ app = cast<CTrackMania>(GetApp());
    CSmArenaClient@ playground = app is null ? null : cast<CSmArenaClient>(app.CurrentPlayground);
    if (playground is null || playground.GameTerminals.Length == 0)
        throw("local-playground-unavailable");
    auto terminal = playground.GameTerminals[0];
    if (terminal is null) throw("local-terminal-unavailable");
    auto player = cast<CSmPlayer>(terminal.GUIPlayer);
    CSmScriptPlayer@ api = player is null ? null : cast<CSmScriptPlayer>(player.ScriptAPI);
    CSceneVehicleVisState@ vis = VehicleState::ViewingPlayerState();
    if (api is null || vis is null) throw("player-or-vehicle-unavailable");
    auto raceData = PlayerState::GetRaceData();
    if (raceData is null || raceData.dPlayerInfo is null) throw("race-data-unavailable");
    if (playground.Map is null || playground.Map.MapInfo is null) throw("local-map-unavailable");
    header["actual_map_uid"] = playground.Map.MapInfo.MapUid;
    header["loaded_game_exe_version"] = app.ManiaPlanetScriptAPI is null
        ? "" : app.ManiaPlanetScriptAPI.ExeVersion;
    float raceTime = Telemetry006RaceTime(app, api);
    bool driving = raceTime > 0.0f;
    bool finished = terminal.UISequence_Current == CGamePlaygroundUIConfig::EUISequence::Finish;
    MemoryBuffer@ buf = MemoryBuffer(0);
    buf.Write(driving ? float(raceData.dPlayerInfo.NumberOfCheckpointsPassed) : 0.0f);
    buf.Write(driving ? float(raceData.dPlayerInfo.CurrentLapNumber) : 0.0f);
    Telemetry006Bool(buf, finished);
    buf.Write(raceTime);
    Telemetry006Vec3(buf, api.Position); Telemetry006Vec3(buf, api.Velocity);
    Telemetry006Vec3(buf, vis.Dir); Telemetry006Vec3(buf, vis.Up);
    buf.Write(api.Speed); buf.Write(api.EngineRpm); Telemetry006Int(buf, api.EngineCurGear);
    buf.Write(vis.FLSlipCoef); buf.Write(vis.FRSlipCoef);
    buf.Write(vis.RLSlipCoef); buf.Write(vis.RRSlipCoef);
    Telemetry006Int(buf, uint(vis.FLGroundContactMaterial)); Telemetry006Int(buf, uint(vis.FRGroundContactMaterial));
    Telemetry006Int(buf, uint(vis.RLGroundContactMaterial)); Telemetry006Int(buf, uint(vis.RRGroundContactMaterial));
    Telemetry006Int(buf, api.WheelsSkiddingCount); Telemetry006Int(buf, api.FlyingDuration);
    buf.Write(api.AdherenceCoef);
    buf.Write(api.InputSteer); buf.Write(api.InputGasPedal); Telemetry006Bool(buf, api.InputIsBraking);
    if (buf.GetSize() != TELEMETRY006_BYTES) throw("unexpected-payload-size");
    return buf;
}

bool Telemetry006ValidNonce(const string &in nonce) {
    if (nonce.Length != 32) return false;
    for (int i = 0; i < nonce.Length; i++)
        if (!"0123456789abcdef".Contains(nonce.SubStr(i, 1))) return false;
    return true;
}

void Telemetry006Request(const string &in line) {
    auto request = Json::Parse(line);
    if (request is null || request.GetType() != Json::Type::Object) throw("invalid-request");
    auto keys = request.GetKeys();
    if (keys is null || keys.Length != 4 || !request.HasKey("protocol") || !request.HasKey("op") ||
        !request.HasKey("nonce") || !request.HasKey("sequence") ||
        request["protocol"].GetType() != Json::Type::String || request["op"].GetType() != Json::Type::String ||
        request["nonce"].GetType() != Json::Type::String || request["sequence"].GetType() != Json::Type::Number)
        throw("invalid-request-shape");
    string protocol = request["protocol"];
    string operation = request["op"];
    string nonce = request["nonce"];
    int sequence = request["sequence"];
    double exactSequence = request["sequence"];
    if (protocol != TELEMETRY006_PROTOCOL || operation != "sample" || !Telemetry006ValidNonce(nonce) ||
        sequence <= 0 || double(sequence) != exactSequence || uint(sequence) != g_Telemetry006Sequence + 1 ||
        uint(sequence) > TELEMETRY006_MAX_REQUESTS ||
        (g_Telemetry006Nonce.Length > 0 && nonce != g_Telemetry006Nonce)) throw("protocol-nonce-or-sequence-mismatch");
    g_Telemetry006Nonce = nonce;
    g_Telemetry006Sequence = uint(sequence);
    uint64 begin = Time::Now;
    auto header = Telemetry006Header("ok");
    MemoryBuffer@ payload;
    try { @payload = Telemetry006Build(header); }
    catch { header["status"] = "unavailable"; header["error"] = "snapshot-unavailable"; }
    uint64 end = Time::Now;
    header["producer_begin_Time_Now_raw"] = begin;
    header["producer_end_Time_Now_raw"] = end;
    if (payload !is null) {
        header["payload_bytes"] = TELEMETRY006_BYTES;
        header["field_count"] = 33;
    }
    if (!Telemetry006Send(header)) throw("response-header-write-failed");
    if (payload !is null) {
        payload.Seek(0, 0);
        // ReadString preserves these exact 132 bytes, including embedded NULs.
        if (!g_Telemetry006Client.WriteRaw(payload.ReadString(TELEMETRY006_BYTES)))
            throw("response-payload-write-failed");
    }
    g_Telemetry006IdleDeadline = Time::Now + TELEMETRY006_IDLE_MS;
    if (g_Telemetry006Sequence >= TELEMETRY006_MAX_REQUESTS) Telemetry006Drop();
}

void Main() {
    @g_Telemetry006Server = Net::Socket();
    if (!g_Telemetry006Server.Listen("127.0.0.1", 19179)) {
        print("HeadlessTelemetry006: loopback19179 unavailable; no server started");
        return;
    }
    print("HeadlessTelemetry006: idle request/response server on loopback19179");
    while (!g_Telemetry006Shutdown) {
        yield();
        try {
            if (g_Telemetry006Client is null) {
                @g_Telemetry006Client = g_Telemetry006Server.Accept();
                if (g_Telemetry006Client is null) continue;
                g_Telemetry006Sequence = 0;
                g_Telemetry006Nonce = "";
                g_Telemetry006Deadline = Time::Now + TELEMETRY006_LIFETIME_MS;
                g_Telemetry006IdleDeadline = Time::Now + TELEMETRY006_IDLE_MS;
                auto hello = Telemetry006Header("hello");
                hello["max_lifetime_ms"] = TELEMETRY006_LIFETIME_MS;
                hello["idle_timeout_ms"] = TELEMETRY006_IDLE_MS;
                hello["max_requests"] = TELEMETRY006_MAX_REQUESTS;
                if (!Telemetry006Send(hello)) throw("hello-write-failed");
            }
            if (g_Telemetry006Client.IsHungUp() || Time::Now >= g_Telemetry006Deadline ||
                Time::Now >= g_Telemetry006IdleDeadline) { Telemetry006Drop(); continue; }
            if (g_Telemetry006Client.Available() > 1024) throw("request-budget-exceeded");
            if (g_Telemetry006Client.IsReady() && g_Telemetry006Client.Available() > 0) {
                string line;
                if (g_Telemetry006Client.ReadLine(line)) {
                    if (line.Length > 1024) throw("request-budget-exceeded");
                    Telemetry006Request(line);
                }
            }
        } catch {
            try {
                auto error = Telemetry006Header("error");
                error["error"] = "protocol-or-runtime-error";
                Telemetry006Send(error);
            } catch { }
            Telemetry006Drop();
        }
    }
    Telemetry006Stop();
}
