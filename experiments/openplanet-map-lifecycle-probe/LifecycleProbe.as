// Standalone observation probe. It does not start until the user clicks Start.
// Never install this probe in a game or profile used by a running training job.

const uint MAX_SAMPLES = 12000;
const uint MAX_EVENTS_PER_SAMPLE = 64;
const uint MAX_TRACE_BYTES = 2 * 1024 * 1024;
const uint MAX_UID_BYTES = 256;
bool g_probeWindow = false;
bool g_probeRecording = false;
bool g_probeTruncated = false;
uint g_probeFrame = 0;
uint g_probeBytes = 0;
string g_probeStopReason = "not_started";
array<string> g_probeSamples;

string ProbeMapUid(CTrackMania@ app) {
    if (app is null) return "";
    auto playground = cast<CSmArenaClient>(app.CurrentPlayground);
    if (playground is null || playground.Map is null || playground.Map.MapInfo is null) return "";
    return playground.Map.MapInfo.MapUid;
}

bool ProbeLocalReady(CTrackMania@ app) {
    if (app is null) return false;
    auto playground = cast<CSmArenaClient>(app.CurrentPlayground);
    if (playground is null || playground.GameTerminals.Length == 0) return false;
    auto player = cast<CSmPlayer>(playground.GameTerminals[0].GUIPlayer);
    return player !is null && cast<CSmScriptPlayer>(player.ScriptAPI) !is null
        && VehicleState::GetViewingPlayer() is player
        && VehicleState::ViewingPlayerState() !is null;
}

// Match Connect's race-time field when available; zero is not proof that no
// map transition occurred, and fields are not an atomic engine snapshot.
float ProbeRaceTimeMs(CTrackMania@ app) {
    if (app is null) return 0.0f;
    auto playground = cast<CSmArenaClient>(app.CurrentPlayground);
    if (playground is null || playground.GameTerminals.Length == 0) return 0.0f;
    auto player = cast<CSmPlayer>(playground.GameTerminals[0].GUIPlayer);
    auto api = player is null ? null : cast<CSmScriptPlayer>(player.ScriptAPI);
    if (api is null || api.StartTime <= 0) return 0.0f;
    auto rules = cast<CSmArenaRulesMode>(app.PlaygroundScript);
    if (rules is null && (app.Network is null || app.Network.PlaygroundClientScriptAPI is null))
        return 0.0f;
    int64 now = rules is null ? int64(app.Network.PlaygroundClientScriptAPI.GameTime)
        : int64(rules.Now);
    return now > api.StartTime ? float(now - api.StartTime) : 0.0f;
}

void ProbeSample() {
    if (g_probeSamples.Length >= MAX_SAMPLES) {
        g_probeTruncated = true;
        g_probeRecording = false;
        g_probeStopReason = "sample_limit";
        return;
    }
    auto sample = Json::Object();
    sample["frame"] = g_probeFrame++;
    sample["time_ms"] = Time::Now;
    auto app = cast<CTrackMania>(GetApp());
    string uid = ProbeMapUid(app);
    sample["map_uid_truncated"] = uid.Length > int(MAX_UID_BYTES);
    sample["map_uid"] = uid.Length > int(MAX_UID_BYTES) ? uid.SubStr(0, MAX_UID_BYTES) : uid;
    sample["local_ready"] = ProbeLocalReady(app);
    sample["race_time_ms"] = ProbeRaceTimeMs(app);
    auto events = Json::Array();
    bool present = app !is null && app.Network !is null && app.Network.ServerPlugin !is null;
    sample["server_plugin_present"] = present;
    if (present) {
        auto server = app.Network.ServerPlugin;
        sample["map_loaded"] = server.MapLoaded;
        sample["map_unload_requested"] = server.MapUnloadRequested;
        uint count = server.PendingEvents.Length;
        sample["pending_event_count"] = count;
        sample["events_truncated"] = count > MAX_EVENTS_PER_SAMPLE;
        for (uint i = 0; i < count && i < MAX_EVENTS_PER_SAMPLE; i++) {
            auto event = server.PendingEvents[i];
            if (event is null) continue;
            auto kind = event.Type;
            if (kind == CGameServerPluginEvent::EType::MapLoaded
                || kind == CGameServerPluginEvent::EType::MapUnloadRequested
                || kind == CGameServerPluginEvent::EType::MapUnloaded) {
                auto entry = Json::Object();
                entry["type"] = kind == CGameServerPluginEvent::EType::MapLoaded ? "MapLoaded"
                    : kind == CGameServerPluginEvent::EType::MapUnloadRequested
                        ? "MapUnloadRequested" : "MapUnloaded";
                entry["type_value"] = int(kind);
                entry["processed"] = event.HasBeenProcessed;
                events.Add(entry);
            }
        }
    }
    sample["map_events"] = events;
    string line = Json::Write(sample);
    if (line.Length + 1 > MAX_TRACE_BYTES - g_probeBytes) {
        g_probeTruncated = true;
        g_probeRecording = false;
        g_probeStopReason = "byte_limit";
        return;
    }
    g_probeSamples.InsertLast(line);
    g_probeBytes += line.Length + 1;
}

void Main() {
    while (true) {
        if (g_probeRecording) {
            try { ProbeSample(); }
            catch {
                // A failed read must not look like a completed capture, and
                // the sampler remains available for an explicit new attempt.
                g_probeRecording = false;
                g_probeTruncated = true;
                g_probeStopReason = "read_error";
            }
        }
        yield();
    }
}

void RenderMenu() {
    if (UI::MenuItem("TrackmaniaRL lifecycle probe", "", g_probeWindow))
        g_probeWindow = !g_probeWindow;
}

void Render() {
    if (!g_probeWindow) return;
    if (UI::Begin("TrackmaniaRL lifecycle probe", g_probeWindow)) {
        UI::TextWrapped("Read-only sampling; use only in a separate game instance. "
            + "Polling can miss events and is not a map-continuity guarantee.");
        if (UI::Button("Start new capture")) {
            g_probeSamples.RemoveRange(0, g_probeSamples.Length);
            g_probeFrame = 0; g_probeBytes = 0; g_probeTruncated = false;
            g_probeStopReason = "recording"; g_probeRecording = true;
        }
        UI::SameLine();
        if (UI::Button("Stop") && g_probeRecording) {
            g_probeRecording = false; g_probeStopReason = "user";
        }
        UI::SameLine();
        if (UI::Button("Copy JSONL")) {
            auto trailer = Json::Object();
            trailer["probe_status"] = g_probeRecording ? "snapshot_in_progress" : "capture_end";
            trailer["samples"] = g_probeSamples.Length;
            trailer["truncated"] = g_probeTruncated;
            trailer["reason"] = g_probeStopReason;
            string prefix = g_probeSamples.Length == 0 ? "" : Text::Join(g_probeSamples, "\n") + "\n";
            IO::SetClipboard(prefix + Json::Write(trailer) + "\n");
        }
        UI::Text("Samples: " + g_probeSamples.Length + "; status: " + g_probeStopReason);
        UI::TextWrapped("Last 20 samples; Copy JSONL includes the full bounded capture.");
        uint start = g_probeSamples.Length > 20 ? g_probeSamples.Length - 20 : 0;
        for (uint i = start; i < g_probeSamples.Length; i++) UI::Text(g_probeSamples[i]);
    }
    UI::End();
}
