// Independent source-only runtime probe. Not installed by the preparation step.
// Main performs one read-only snapshot and writes one fresh file in its own storage.
// No game input, reset, settings setter, network, memory read/write or code hook.

const uint PROBE_MAX_MEMBER_SLOTS = 8192;
const uint PROBE_MAX_DEVICES = 16;
const uint PROBE_MAX_PLUGINS = 128;
string g_HeadlessRuntimeProbe003Status = "Loaded; capture has not started";
uint g_HeadlessRuntimeProbe003MemberSlots = 0;

Json::Value@ Probe003Vec3(vec3 value) {
    auto values = Json::Array();
    values.Add(Json::Value(value.x));
    values.Add(Json::Value(value.y));
    values.Add(Json::Value(value.z));
    return values;
}

Json::Value@ Probe003DescribeType(const string &in requested) {
    auto row = Json::Object();
    row["requested_name"] = requested;
    const Reflection::MwClassInfo@ info;
    try { @info = Reflection::GetType(requested); }
    catch { row["status"] = "lookup-error"; return row; }
    if (info is null) { row["status"] = "not-found"; return row; }
    row["status"] = "found";
    row["name"] = info.Name;
    row["reported_id_u32"] = info.ID;
    row["reported_size_u32"] = info.Size;
    row["base_available"] = info.BaseType !is null;
    if (info.BaseType !is null) row["base_name"] = info.BaseType.Name;
    auto members = Json::Array();
    auto source = info.Members;
    row["members_available"] = source !is null;
    if (source !is null) {
        row["reported_member_slots"] = source.Length;
        if (source.Length > PROBE_MAX_MEMBER_SLOTS - g_HeadlessRuntimeProbe003MemberSlots) {
            row["status"] = "member-budget-exceeded";
            return row;
        }
        g_HeadlessRuntimeProbe003MemberSlots += source.Length;
        for (uint i = 0; i < source.Length; i++) {
            auto member = Json::Object();
            member["slot"] = i;
            if (source[i] is null) member["status"] = "null-metadata";
            else {
                member["status"] = "reported";
                member["name"] = source[i].Name;
                member["reported_id_u32"] = source[i].ID;
                member["reported_offset_u16"] = source[i].Offset;
            }
            members.Add(member);
        }
    }
    row["members"] = members;
    return row;
}

Json::Value@ Probe003Plugins() {
    auto result = Json::Object();
    auto rows = Json::Array();
    auto plugins = Meta::AllPlugins();
    result["available"] = plugins !is null;
    if (plugins !is null) {
        result["reported_count"] = plugins.Length;
        result["truncated"] = plugins.Length > PROBE_MAX_PLUGINS;
        for (uint i = 0; i < plugins.Length && i < PROBE_MAX_PLUGINS; i++) {
            auto row = Json::Object();
            if (plugins[i] is null) row["status"] = "null-plugin";
            else {
                row["status"] = "loaded";
                row["plugin_id"] = plugins[i].ID;
                row["version"] = plugins[i].Version;
                row["enabled"] = plugins[i].Enabled;
                row["signature_level"] = plugins[i].SignatureLevel;
                row["plugin_type"] = int(plugins[i].Type);
            }
            rows.Add(row);
        }
    }
    result["rows"] = rows;
    return result;
}

Json::Value@ Probe003InputPort(CGameCtnApp@ app) {
    auto row = Json::Object();
    auto port = app.InputPort;
    row["available"] = port !is null;
    if (port is null) return row;
    auto type = Reflection::TypeOf(port);
    if (type !is null) row["runtime_type"] = type.Name;
    row["FakeInputLagAvg"] = port.FakeInputLagAvg;
    row["FakeInputLagVar"] = port.FakeInputLagVar;
    row["MaxSampleRate"] = port.MaxSampleRate;
    row["PollingEnabled"] = port.PollingEnabled;
    row["MinHistoryLength"] = port.MinHistoryLength;
    row["IsFocused"] = port.IsFocused;
    row["IgnoreFocusForGamePads"] = port.IgnoreFocusForGamePads;
    row["EventInStoreCount"] = port.EventInStoreCount;
    row["DevicePlugEventCount"] = port.DevicePlugEventCount;
    row["StatsLatestEvent"] = port.StatsLatestEvent;
    row["StatsDInputEventsLastFrame"] = port.StatsDInputEventsLastFrame;
    row["StatsDInputEventsCount"] = port.StatsDInputEventsCount;
    row["StatsDInputTimedEventsCount"] = port.StatsDInputTimedEventsCount;
    row["StatsDInputWrongTimestampAhead"] = port.StatsDInputWrongTimestampAhead;
    row["StatsDInputWrongTimestampLate"] = port.StatsDInputWrongTimestampLate;
    row["StatsDInputWrongTimestampAvgDelta"] = port.StatsDInputWrongTimestampAvgDelta;
    row["StatsNbMappedInputsReceived"] = port.StatsNbMappedInputsReceived;
    auto devices = Json::Array();
    row["connected_device_count"] = port.ConnectedDevices.Length;
    row["connected_devices_truncated"] = port.ConnectedDevices.Length > PROBE_MAX_DEVICES;
    for (uint i = 0; i < port.ConnectedDevices.Length && i < PROBE_MAX_DEVICES; i++) {
        auto device = port.ConnectedDevices[i];
        auto item = Json::Object();
        item["slot"] = i;
        item["available"] = device !is null;
        if (device !is null) {
            auto deviceType = Reflection::TypeOf(device);
            if (deviceType !is null) item["runtime_type"] = deviceType.Name;
            item["IsDisabled"] = device.IsDisabled;
            item["InputNotAvailable"] = device.InputNotAvailable;
            item["IsUnPlugged"] = device.IsUnPlugged;
            item["MustBePolled"] = device.MustBePolled;
            item["ObjectCount"] = device.ObjectCount;
        }
        devices.Add(item);
    }
    row["connected_devices"] = devices;
    row["timestamp_units"] = "raw API fields; no extra unit interpretation";
    return row;
}

Json::Value@ Probe003Player(CTrackMania@ app) {
    auto row = Json::Object();
    CSmArenaClient@ playground = cast<CSmArenaClient>(app.CurrentPlayground);
    row["playground_available"] = playground !is null;
    if (playground is null || playground.GameTerminals.Length == 0) return row;
    auto terminal = playground.GameTerminals[0];
    row["reader"] = "GameTerminals[0].GUIPlayer.ScriptAPI";
    auto player = cast<CSmPlayer>(terminal.GUIPlayer);
    CSmScriptPlayer@ api = player is null ? null : cast<CSmScriptPlayer>(player.ScriptAPI);
    row["script_api_available"] = api !is null;
    if (api is null) return row;
    row["InputSteer"] = api.InputSteer;
    row["InputGasPedal"] = api.InputGasPedal;
    row["InputIsBraking"] = api.InputIsBraking;
    row["Position"] = Probe003Vec3(api.Position);
    row["Velocity"] = Probe003Vec3(api.Velocity);
    row["StartTime"] = api.StartTime;
    CSmArenaRulesMode@ rules = cast<CSmArenaRulesMode>(app.PlaygroundScript);
    row["rules_now_available"] = rules !is null;
    if (rules !is null) {
        row["rules_Now"] = rules.Now;
        row["race_delta_ms_derived"] = int64(rules.Now) - int64(api.StartTime);
    }
    row["sampling_phase"] = "one Main coroutine read; physical tick phase unqualified";
    return row;
}

void Main() {
    g_HeadlessRuntimeProbe003Status = "Capturing one read-only snapshot";
    try {
        auto result = Json::Object();
        result["schema"] = "tmrl.headless-runtime-probe/v3";
        result["collector_id"] = "headlessRuntimeProbe003";
        result["collector_version"] = "0.0.3";
        result["atomic_snapshot_claimed"] = false;
        result["memory_offsets_interpreted"] = false;
        result["hook_installed"] = false;
        result["game_settings_changed"] = false;
        result["sample_source"] = "actual loaded plugin Main entry point";
        result["Time_Now_raw"] = Time::Now;
        result["openplanet_version"] = Meta::OpenplanetVersion();
        result["openplanet_git_commit"] = Meta::OpenplanetVersionGitCommit();
        auto self = Meta::ExecutingPlugin();
        if (self !is null) {
            result["actual_plugin_id"] = self.ID;
            result["actual_plugin_version"] = self.Version;
            result["actual_plugin_enabled"] = self.Enabled;
            result["actual_plugin_signature_level"] = self.SignatureLevel;
        }
        auto plugins = Probe003Plugins();
        result["loaded_plugins"] = plugins;
        CTrackMania@ app = cast<CTrackMania>(GetApp());
        result["app_available"] = app !is null;
        if (app !is null) {
            result["TimeSinceInitMs"] = app.TimeSinceInitMs;
            result["mania_planet_script_api_available"] = app.ManiaPlanetScriptAPI !is null;
            if (app.ManiaPlanetScriptAPI !is null) {
                result["loaded_game_exe_version"] = app.ManiaPlanetScriptAPI.ExeVersion;
            }
            auto input = Probe003InputPort(app);
            auto player = Probe003Player(app);
            result["input_port"] = input;
            result["player_sample"] = player;
        }
        string[] names = {"CGameCtnApp", "CGameApp", "CTrackMania", "CInputPort",
            "CInputDevice", "CInputDeviceDx8Pad", "CInputDeviceDx8Keyboard",
            "CGameManiaPlanetScriptAPI", "CSmArenaClient", "CSmPlayer",
            "CSmScriptPlayer", "CGameSceneSim"};
        auto records = Json::Array();
        for (uint i = 0; i < names.Length; i++) {
            auto record = Probe003DescribeType(names[i]);
            records.Add(record);
        }
        result["requested_type_count"] = names.Length;
        result["reflection_metadata"] = records;
        result["reflection_member_slots_read"] = g_HeadlessRuntimeProbe003MemberSlots;
        result["reflection_offset_units"] = "raw uint16; no byte/address claim";
        result["reflection_size_units"] = "raw uint32; no byte/address claim";
        const string path = IO::FromStorageFolder("headless-runtime-probe003-" + Time::Now + ".json");
        if (IO::FileExists(path)) {
            g_HeadlessRuntimeProbe003Status = "Output collision; nothing overwritten";
            print("HeadlessRuntimeProbe003: output collision; nothing overwritten");
            return;
        }
        Json::ToFile(path, result, true);
        g_HeadlessRuntimeProbe003Status = "Saved " + path;
        print("HeadlessRuntimeProbe003: saved " + path);
    } catch {
        g_HeadlessRuntimeProbe003Status = "Capture failed; no exception text or user data exported";
        print("HeadlessRuntimeProbe003: capture failed; inspect compile/runtime status");
    }
}
