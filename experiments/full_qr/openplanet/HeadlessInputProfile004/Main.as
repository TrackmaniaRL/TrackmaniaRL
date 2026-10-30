// Explicit temporary profile, independent of TrackmaniaRL Connect.
// The only game writes are FakeInputLagAvg/Var apply and exact baseline restore.
const string PROFILE004_PROTOCOL = "tmrl.input-profile/v1";
const string PROFILE004_NAME = "no-artificial-lag/v1";
const string PROFILE004_MUTEX = "Global\\TrackmaniaRL.ComparisonController";
const uint PROFILE004_PORT = 19177;
const uint PROFILE004_MAX_BYTES = 8192;
const uint PROFILE004_MAX_TTL_MS = 120000;

Net::Socket@ g_Profile004Server;
Net::Socket@ g_Profile004Client;
CInputPort@ g_Profile004Port;
bool g_Profile004Shutdown = false;
bool g_Profile004Dirty = false;
bool g_Profile004Active = false;
uint g_Profile004BaselineAvg = 0;
float g_Profile004BaselineVar = 0;
uint g_Profile004BaselineRate = 0;
bool g_Profile004BaselinePolling = false;
bool g_Profile004BaselineIgnoreFocus = false;
uint64 g_Profile004Deadline = 0;
uint64 g_Profile004ClientDeadline = 0;
uint g_Profile004Sequence = 0;
string g_Profile004Nonce = "";
string g_Profile004Map = "";
string[] g_Profile004UsedNonces;

string Profile004MapUid() {
    CTrackMania@ app = cast<CTrackMania>(GetApp());
    if (app is null) return "";
    CSmArenaClient@ pg = cast<CSmArenaClient>(app.CurrentPlayground);
    string playgroundUid = "";
    string rootUid = "";
    if (pg !is null && pg.Map !is null && pg.Map.MapInfo !is null)
        playgroundUid = pg.Map.MapInfo.MapUid;
    if (app.RootMap !is null && app.RootMap.MapInfo !is null)
        rootUid = app.RootMap.MapInfo.MapUid;
    // Editor-validation resets can briefly have no CurrentPlayground. RootMap
    // is an actual game map getter; conflicting live maps always fail closed.
    if (playgroundUid.Length > 0 && rootUid.Length > 0 && playgroundUid != rootUid) return "";
    return playgroundUid.Length > 0 ? playgroundUid : rootUid;
}

CInputPort@ Profile004CurrentPort() {
    auto app = GetApp();
    return app is null ? null : app.InputPort;
}

Json::Value@ Profile004Settings(CInputPort@ port) {
    auto row = Json::Object();
    row["available"] = port !is null;
    if (port !is null) {
        row["avg"] = port.FakeInputLagAvg;
        row["var"] = port.FakeInputLagVar;
        row["MaxSampleRate"] = port.MaxSampleRate;
        row["PollingEnabled"] = port.PollingEnabled;
        row["IgnoreFocusForGamePads"] = port.IgnoreFocusForGamePads;
        row["focused"] = port.IsFocused;
    }
    return row;
}

Json::Value@ Profile004Reply(const string &in op) {
    auto row = Json::Object();
    row["protocol"] = PROFILE004_PROTOCOL;
    row["profile"] = PROFILE004_NAME;
    row["op"] = op;
    row["nonce"] = g_Profile004Nonce;
    row["sequence"] = g_Profile004Sequence;
    return row;
}

bool Profile004WriteJournal(const string &in stage, const string &in reason) {
    auto event = Profile004Reply(stage);
    event["schema"] = "tmrl.input-profile-journal/v1";
    event["reason"] = reason;
    event["Time_Now_raw"] = Time::Now;
    event["deadline_Time_Now_raw"] = g_Profile004Deadline;
    event["expected_map_uid"] = g_Profile004Map;
    event["actual_map_uid"] = Profile004MapUid();
    event["baseline_avg"] = g_Profile004BaselineAvg;
    event["baseline_var"] = g_Profile004BaselineVar;
    event["restore_pending"] = g_Profile004Dirty;
    if (stage == "restored") event["restored_verified"] = !g_Profile004Dirty;
    auto actual = Profile004Settings(g_Profile004Port);
    event["actual"] = actual;
    event["plugin_writes_only_profile_fields"] = true;
    event["controller_mutex_attested_by_client"] = PROFILE004_MUTEX;
    event["openplanet_version"] = Meta::OpenplanetVersion();
    auto plugin = Meta::ExecutingPlugin();
    if (plugin !is null) event["actual_plugin_id"] = plugin.ID;
    const string path = IO::FromStorageFolder("input-profile004-" + g_Profile004Nonce + "-" + stage + ".json");
    if (IO::FileExists(path)) return false;
    Json::ToFile(path, event, true);
    return true;
}

bool Profile004Send(Json::Value@ value) {
    return g_Profile004Client !is null && g_Profile004Client.WriteRaw(Json::Write(value) + "\n");
}

Json::Value@ Profile004Restore(const string &in reason) {
    bool restored = !g_Profile004Dirty;
    bool journalWritten = false;
    if (g_Profile004Dirty) {
        // Dirty is set before the first write, so a partial apply is restored too.
        // Attempt each setter independently: one failing must not skip the other.
        if (g_Profile004Port !is null) {
            try { g_Profile004Port.FakeInputLagAvg = g_Profile004BaselineAvg; }
            catch { }
            try { g_Profile004Port.FakeInputLagVar = g_Profile004BaselineVar; }
            catch { }
            try {
                restored = g_Profile004Port.FakeInputLagAvg == g_Profile004BaselineAvg
                    && g_Profile004Port.FakeInputLagVar == g_Profile004BaselineVar;
            } catch { restored = false; }
        }
        if (restored) g_Profile004Dirty = false;
        try { journalWritten = Profile004WriteJournal("restored", reason); }
        catch { journalWritten = false; }
    }
    g_Profile004Active = false;
    // Construct/serialize reply only after attempting both restoration writes.
    auto response = Profile004Reply("restored");
    response["reason"] = reason;
    response["restored_verified"] = restored;
    response["journal_written"] = journalWritten;
    response["baseline_avg"] = g_Profile004BaselineAvg;
    response["baseline_var"] = g_Profile004BaselineVar;
    auto actual = Profile004Settings(g_Profile004Port);
    response["actual"] = actual;
    return response;
}

void Profile004Drop(const string &in reason) {
    if (g_Profile004Dirty || g_Profile004Active) {
        try {
            auto restored = Profile004Restore(reason);
            Profile004Send(restored);
        } catch { }
    }
    if (g_Profile004Client !is null) {
        try { g_Profile004Client.Close(); } catch { }
        @g_Profile004Client = null;
    }
}

bool Profile004Keys(Json::Value@ value, string[]@ expected) {
    if (value is null || value.GetType() != Json::Type::Object) return false;
    auto keys = value.GetKeys();
    if (keys is null || keys.Length != expected.Length) return false;
    for (uint i = 0; i < expected.Length; i++) {
        if (!value.HasKey(expected[i])) return false;
    }
    return true;
}

bool Profile004NonceValid(const string &in nonce) {
    if (nonce.Length != 32) return false;
    for (int i = 0; i < nonce.Length; i++) {
        if ("0123456789abcdef".IndexOf(nonce.SubStr(i, 1)) < 0) return false;
    }
    for (uint i = 0; i < g_Profile004UsedNonces.Length; i++) {
        if (g_Profile004UsedNonces[i] == nonce) return false;
    }
    return g_Profile004UsedNonces.Length < 32;
}

void Profile004Reject(const string &in code) {
    auto response = Profile004Reply("rejected");
    response["error"] = code;
    Profile004Send(response);
    Profile004Drop("protocol-error:" + code);
}

void Profile004Request(const string &in text) {
    auto request = Json::Parse(text);
    if (request is null || request.GetType() != Json::Type::Object) {
        Profile004Reject("object-required"); return;
    }
    if (!request.HasKey("protocol") || string(request["protocol"]) != PROFILE004_PROTOCOL
        || request["protocol"].GetType() != Json::Type::String
        || !request.HasKey("op") || request["op"].GetType() != Json::Type::String
        || !request.HasKey("sequence") || request["sequence"].GetType() != Json::Type::Number) {
        Profile004Reject("protocol-required"); return;
    }
    int sequence = request["sequence"];
    double sequenceExact = request["sequence"];
    if (sequence <= 0 || double(sequence) != sequenceExact || uint(sequence) != g_Profile004Sequence + 1) {
        Profile004Reject("sequence-replay-or-gap"); return;
    }
    g_Profile004Sequence = uint(sequence);
    string op = request["op"];
    if (op == "acquire") {
        string[] keys = {"protocol", "op", "sequence", "nonce", "profile",
            "controller_mutex_held", "expected_map_uid", "ttl_ms", "expected_baseline"};
        string[] baselineKeys = {"avg", "var"};
        if (!Profile004Keys(request, keys) || !Profile004Keys(request["expected_baseline"], baselineKeys)
            || request["nonce"].GetType() != Json::Type::String
            || request["profile"].GetType() != Json::Type::String
            || request["controller_mutex_held"].GetType() != Json::Type::String
            || request["expected_map_uid"].GetType() != Json::Type::String
            || request["ttl_ms"].GetType() != Json::Type::Number
            || request["expected_baseline"]["avg"].GetType() != Json::Type::Number
            || request["expected_baseline"]["var"].GetType() != Json::Type::Number) {
            Profile004Reject("exact-acquire-shape-required"); return;
        }
        if (g_Profile004Active || g_Profile004Dirty) { Profile004Reject("already-active"); return; }
        string nonce = request["nonce"];
        if (!Profile004NonceValid(nonce)) { Profile004Reject("nonce-replay-or-invalid"); return; }
        string profile = request["profile"];
        string mutex = request["controller_mutex_held"];
        string expectedMap = request["expected_map_uid"];
        int ttl = request["ttl_ms"];
        double ttlExact = request["ttl_ms"];
        if (profile != PROFILE004_NAME || mutex != PROFILE004_MUTEX
            || expectedMap.Length == 0 || expectedMap.Length > 128
            || ttl <= 0 || double(ttl) != ttlExact || uint(ttl) > PROFILE004_MAX_TTL_MS) {
            Profile004Reject("profile-scope-or-ttl-invalid"); return;
        }
        auto port = Profile004CurrentPort();
        if (port is null || !port.IsFocused || Profile004MapUid() != expectedMap) {
            Profile004Reject("focus-map-or-port-unavailable"); return;
        }
        int avg = request["expected_baseline"]["avg"];
        double avgExact = request["expected_baseline"]["avg"];
        float variance = request["expected_baseline"]["var"];
        if (avg < 0 || double(avg) != avgExact || uint(avg) != port.FakeInputLagAvg || variance != port.FakeInputLagVar) {
            Profile004Reject("baseline-changed"); return;
        }
        @g_Profile004Port = port;
        g_Profile004BaselineAvg = port.FakeInputLagAvg;
        g_Profile004BaselineVar = port.FakeInputLagVar;
        g_Profile004BaselineRate = port.MaxSampleRate;
        g_Profile004BaselinePolling = port.PollingEnabled;
        g_Profile004BaselineIgnoreFocus = port.IgnoreFocusForGamePads;
        g_Profile004Nonce = nonce;
        g_Profile004Map = expectedMap;
        g_Profile004Deadline = Time::Now + uint(ttl);
        g_Profile004UsedNonces.InsertLast(nonce);
        if (!Profile004WriteJournal("baseline", "acquire")) {
            Profile004Reject("baseline-journal-failed"); return;
        }
        g_Profile004Dirty = true;
        port.FakeInputLagAvg = 0;
        port.FakeInputLagVar = 0.0f;
        if (port.FakeInputLagAvg != 0 || port.FakeInputLagVar != 0.0f) {
            Profile004Reject("apply-readback-failed"); return;
        }
        g_Profile004Active = true;
        if (!Profile004WriteJournal("applied", "acquire")) {
            Profile004Reject("applied-journal-failed"); return;
        }
        auto response = Profile004Reply("applied");
        response["baseline_avg"] = g_Profile004BaselineAvg;
        response["baseline_var"] = g_Profile004BaselineVar;
        response["ttl_ms"] = ttl;
        auto actual = Profile004Settings(port);
        response["actual"] = actual;
        Profile004Send(response);
        return;
    }
    string[] activeKeys = {"protocol", "op", "sequence", "nonce"};
    if (!Profile004Keys(request, activeKeys) || request["nonce"].GetType() != Json::Type::String) {
        Profile004Reject("exact-active-shape-required"); return;
    }
    if (!g_Profile004Active || !request.HasKey("nonce") || string(request["nonce"]) != g_Profile004Nonce) {
        Profile004Reject("active-nonce-required"); return;
    }
    if (op == "release") {
        auto response = Profile004Restore("explicit-release");
        Profile004Send(response);
        Profile004Drop("released");
        return;
    }
    if (op == "status") {
        auto response = Profile004Reply("active");
        auto actual = Profile004Settings(g_Profile004Port);
        response["actual"] = actual;
        response["actual_map_uid"] = Profile004MapUid();
        response["deadline_Time_Now_raw"] = g_Profile004Deadline;
        Profile004Send(response);
        return;
    }
    Profile004Reject("unknown-operation");
}

void Profile004Shutdown(const string &in reason) {
    g_Profile004Shutdown = true;
    Profile004Drop(reason);
    if (g_Profile004Server !is null) {
        try { g_Profile004Server.Close(); } catch { }
        @g_Profile004Server = null;
    }
}

void OnDisabled() { Profile004Shutdown("plugin-disabled"); }
void OnDestroyed() { Profile004Shutdown("plugin-destroyed"); }
void OnDestroy() { Profile004Shutdown("plugin-destroy"); }

void Main() {
    @g_Profile004Server = Net::Socket();
    if (!g_Profile004Server.Listen("127.0.0.1", PROFILE004_PORT)) {
        print("HeadlessInputProfile004: loopback port unavailable; no settings changed");
        return;
    }
    print("HeadlessInputProfile004: ready on loopback19177; idle, no settings changed");
    while (!g_Profile004Shutdown) {
        yield();
        try {
            if (g_Profile004Dirty && !g_Profile004Active) {
                Profile004Drop("retry-incomplete-restore");
                continue;
            }
            if (g_Profile004Active) {
                string reason = "";
                if (Time::Now >= g_Profile004Deadline) reason = "lease-timeout";
                else if (g_Profile004Port is null || !g_Profile004Port.IsFocused) reason = "focus-lost";
                else if (Profile004MapUid() != g_Profile004Map) reason = "map-changed";
                else if (g_Profile004Port.FakeInputLagAvg != 0 || g_Profile004Port.FakeInputLagVar != 0.0f)
                    reason = "profile-settings-changed";
                else if (g_Profile004Port.MaxSampleRate != g_Profile004BaselineRate
                    || g_Profile004Port.PollingEnabled != g_Profile004BaselinePolling
                    || g_Profile004Port.IgnoreFocusForGamePads != g_Profile004BaselineIgnoreFocus)
                    reason = "other-input-settings-changed";
                if (reason.Length > 0) { Profile004Drop(reason); continue; }
            }
            if (g_Profile004Client is null) {
                @g_Profile004Client = g_Profile004Server.Accept();
                if (g_Profile004Client is null) continue;
                g_Profile004Sequence = 0;
                g_Profile004Nonce = "";
                g_Profile004ClientDeadline = Time::Now + 5000;
                auto hello = Profile004Reply("hello");
                auto baseline = Profile004Settings(Profile004CurrentPort());
                hello["baseline"] = baseline;
                hello["actual_map_uid"] = Profile004MapUid();
                hello["max_ttl_ms"] = PROFILE004_MAX_TTL_MS;
                Profile004Send(hello);
            }
            if (g_Profile004Client.IsHungUp()) { Profile004Drop("client-eof"); continue; }
            if (!g_Profile004Active && Time::Now >= g_Profile004ClientDeadline) {
                Profile004Drop("handshake-timeout"); continue;
            }
            if (g_Profile004Client.Available() > int(PROFILE004_MAX_BYTES)) {
                Profile004Reject("request-too-large"); continue;
            }
            if (g_Profile004Client.IsReady() && g_Profile004Client.Available() > 0) {
                string line = "";
                if (g_Profile004Client.ReadLine(line)) {
                    if (line.Length > int(PROFILE004_MAX_BYTES)) Profile004Reject("request-too-large");
                    else Profile004Request(line);
                }
            }
        } catch { Profile004Drop("protocol-or-runtime-error"); }
    }
    Profile004Shutdown("main-exit");
}
