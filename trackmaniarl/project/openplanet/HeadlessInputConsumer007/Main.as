// Isolated five-second diagnostic. Installation/loading is a separate operation.
// The callback reads integer bit patterns only. It never writes game data,
// changes controls, invokes game getters, yields, formats strings, or performs I/O.
// Time::Now is integer host metadata, explicitly not a simulation clock.
// Dev::Hook itself temporarily replaces exactly eight verified code bytes.
const uint CONSUMER007_MAX_RECORDS = 256;
const uint CONSUMER007_MAX_CALLS = 4096;
const uint CONSUMER007_DURATION_MS = 5000;
const uint64 CONSUMER007_RVA = 0x841734;
const string CONSUMER007_VERSION = "2026-02-02_17_51";
const string CONSUMER007_HEX = "f30f118b301400008b83d8150000440f284c2450440f284424600f287c24700f28b424800000008983081600008b83dc15000039830c160000740d89830c1600";
const string CONSUMER007_PATTERN = "F3 0F 11 8B 30 14 00 00 8B 83 D8 15 00 00 44 0F 28 4C 24 50 44 0F 28 44 24 60 0F 28 7C 24 70 0F 28 B4 24 80 00 00 00 89 83 08 16 00 00 8B 83 DC 15 00 00 39 83 0C 16 00 00 74 0D 89 83 0C 16 00";

Dev::HookInfo@ g_Consumer007Hook;
Net::Socket@ g_Consumer007Server;
Net::Socket@ g_Consumer007Client;
CSmPlayer@ g_Consumer007Owner;
CSmArenaClient@ g_Consumer007Playground;
Json::Value@ g_Consumer007Report;
bool g_Consumer007Shutdown = false;
bool g_Consumer007Attempted = false;
bool g_Consumer007Recording = false;
bool g_Consumer007InsideCallback = false;
bool g_Consumer007Saved = false;
bool g_Consumer007HookEverInstalled = false;
bool g_Consumer007CodeHookAttempted = false;
bool g_Consumer007UnhookError = false;
bool g_Consumer007Restored = false;
bool g_Consumer007Reentered = false;
bool g_Consumer007CallbackIdentityLost = false;
uint g_Consumer007Count = 0;
uint g_Consumer007Calls = 0;
uint g_Consumer007Skipped = 0;
uint64 g_Consumer007Base = 0;
uint64 g_Consumer007End = 0;
uint64 g_Consumer007Target = 0;
uint64 g_Consumer007Participant = 0;
uint64 g_Consumer007Vehicle = 0;
uint64 g_Consumer007Trampoline = 0;
uint64 g_Consumer007Start = 0;
uint64 g_Consumer007Deadline = 0;
uint g_Consumer007EntityClass = 0;
uint64 g_Consumer007ClientDeadline = 0;
string g_Consumer007Nonce = "";
string g_Consumer007OutputPath = "";
string g_Consumer007BeforeHex = "";
string g_Consumer007AfterHex = "";
string g_Consumer007TrampolineHex = "";
string g_Consumer007ExeVersion = "";
uint64[] g_Consumer007Pointers;
uint64[] g_Consumer007Times;
uint[] g_Consumer007StepArguments;
uint[] g_Consumer007Bits;
string g_Consumer007Status = "Source loaded; not armed";
string g_Consumer007Reason = "not-started";

bool Consumer007WithinImage(uint64 ptr, uint64 bytes) {
    return ptr >= g_Consumer007Base && ptr <= g_Consumer007End &&
        bytes <= g_Consumer007End - ptr;
}

string Consumer007Hex(uint64 ptr, uint bytes) {
    string result = "";
    for (uint i = 0; i < bytes; i++)
        result += Text::Format("%02x", uint(Dev::SafeReadUint8(ptr + i)));
    return result;
}

// Parameters are bound by their x86 register names, as documented by Dev::Hook.
// Its trampoline has already executed the exact eight-byte filtered-steer store.
// R13D is a literal simulation-function argument; its clock meaning is unqualified.
void Consumer007Capture(uint64 rbx, uint64 r13) {
    if (!g_Consumer007Recording) return;
    if (g_Consumer007InsideCallback) {
        g_Consumer007Reentered = true;
        g_Consumer007Recording = false;
        return;
    }
    g_Consumer007InsideCallback = true;
    uint64 now = Time::Now;
    if (now >= g_Consumer007Deadline || g_Consumer007Calls >= CONSUMER007_MAX_CALLS ||
            g_Consumer007Count >= CONSUMER007_MAX_RECORDS) {
        g_Consumer007Recording = false;
        g_Consumer007InsideCallback = false;
        return;
    }
    g_Consumer007Calls++;
    if (Dev::SafeReadUint32(g_Consumer007Participant + 0x1110) != g_Consumer007EntityClass ||
            Dev::SafeReadUint64(g_Consumer007Participant + 0x1118) != g_Consumer007Vehicle) {
        g_Consumer007CallbackIdentityLost = true;
        g_Consumer007Recording = false;
        g_Consumer007InsideCallback = false;
        return;
    }
    if (rbx != g_Consumer007Vehicle) {
        g_Consumer007Skipped++;
        g_Consumer007InsideCallback = false;
        return;
    }
    uint index = g_Consumer007Count;
    g_Consumer007Times[index] = now;
    g_Consumer007StepArguments[index] = uint(r13 & 0xffffffff);
    g_Consumer007Pointers[index] = rbx;
    for (uint word = 0; word < 3; word++)
        g_Consumer007Bits[index * 20 + word] = Dev::SafeReadUint32(rbx + 0x98 + word * 4);
    g_Consumer007Bits[index * 20 + 3] = Dev::SafeReadUint32(rbx + 0x1430);
    g_Consumer007Bits[index * 20 + 4] = Dev::SafeReadUint32(rbx + 0x858);
    g_Consumer007Bits[index * 20 + 5] = Dev::SafeReadUint32(rbx + 0x85c);
    g_Consumer007Bits[index * 20 + 6] = Dev::SafeReadUint32(rbx + 0x868);
    for (uint word = 0; word < 3; word++)
        g_Consumer007Bits[index * 20 + 7 + word] = Dev::SafeReadUint32(g_Consumer007Participant + 0xeec + word * 4);
    for (uint word = 0; word < 10; word++)
        g_Consumer007Bits[index * 20 + 10 + word] = Dev::SafeReadUint32(rbx + 0x12e0 + word * 4);
    g_Consumer007Count = index + 1;
    if (g_Consumer007Count >= CONSUMER007_MAX_RECORDS) g_Consumer007Recording = false;
    g_Consumer007InsideCallback = false;
}

void Consumer007Unhook(const string &in reason) {
    g_Consumer007Recording = false;
    g_Consumer007Reason = reason;
    if (g_Consumer007Hook !is null) {
        try {
            Dev::Unhook(g_Consumer007Hook);
            @g_Consumer007Hook = null;
        } catch { g_Consumer007UnhookError = true; }
    }
    // Observe the target even when Hook threw or returned a null handle.
    if (g_Consumer007CodeHookAttempted && Consumer007WithinImage(g_Consumer007Target, 64)) {
        g_Consumer007AfterHex = Consumer007Hex(g_Consumer007Target, 64);
        g_Consumer007Restored = g_Consumer007AfterHex == g_Consumer007BeforeHex &&
            g_Consumer007BeforeHex == CONSUMER007_HEX && g_Consumer007Hook is null &&
            !g_Consumer007UnhookError;
    }
}

void Consumer007Save() {
    if (g_Consumer007Saved || g_Consumer007InsideCallback) return;
    auto report = Json::Object();
    report["schema"] = "tmrl.headless-input-consumer-probe/v7";
    report["collector_id"] = "headlessInputConsumerProbe007";
    report["collector_version"] = "0.0.7";
    report["nonce"] = g_Consumer007Nonce;
    report["status"] = g_Consumer007Status;
    report["stop_reason"] = g_Consumer007Reason;
    report["expected_executable_sha256"] = "3fc7d8cda542beda131c44306b123f4004d07d7e22f512b46b762afc29f6edda";
    report["binary_sha256_checked_by_plugin"] = false;
    report["expected_exe_version"] = CONSUMER007_VERSION;
    report["observed_exe_version"] = g_Consumer007ExeVersion;
    report["openplanet_version"] = Meta::OpenplanetVersion();
    report["openplanet_git_commit"] = Meta::OpenplanetVersionGitCommit();
    report["base_address"] = Text::FormatPointer(g_Consumer007Base);
    report["base_address_end"] = Text::FormatPointer(g_Consumer007End);
    report["target_address"] = Text::FormatPointer(g_Consumer007Target);
    report["target_rva"] = "0x841734";
    report["expected_original_64_hex"] = CONSUMER007_HEX;
    report["observed_before_64_hex"] = g_Consumer007BeforeHex;
    report["observed_after_unhook_64_hex"] = g_Consumer007AfterHex;
    report["exact_original_prefix_verified"] = g_Consumer007BeforeHex == CONSUMER007_HEX;
    report["padding"] = 3;
    report["original_instruction_bytes_copied"] = 8;
    report["trampoline_address"] = Text::FormatPointer(g_Consumer007Trampoline);
    report["observed_original_trampoline_8_hex"] = g_Consumer007TrampolineHex;
    report["requested_push_registers"] = "SSE (1), XMM0..15 save/restore in pinned implementation";
    report["fp_control_preservation"] = "required by Windows x64 ABI; not measured by this collector";
    report["hook_ever_installed"] = g_Consumer007HookEverInstalled;
    report["code_hook_attempted"] = g_Consumer007CodeHookAttempted;
    report["unhook_error"] = g_Consumer007UnhookError;
    report["unhooked"] = g_Consumer007Hook is null;
    report["original_64_bytes_restored"] = g_Consumer007Restored;
    report["callback_reentered"] = g_Consumer007Reentered;
    report["owner_vehicle_identity_lost"] = g_Consumer007CallbackIdentityLost;
    report["expected_vehicle_entity_class"] = "0x0a020000";
    report["expected_vehicle_address"] = Text::FormatPointer(g_Consumer007Vehicle);
    report["owner_vehicle_reference_class_offset"] = "0x1110";
    report["owner_vehicle_reference_payload_offset"] = "0x1118";
    report["expected_participant_address"] = Text::FormatPointer(g_Consumer007Participant);
    report["start_Time_Now_raw"] = g_Consumer007Start;
    report["deadline_Time_Now_raw"] = g_Consumer007Deadline;
    report["callback_invocations"] = g_Consumer007Calls;
    report["non_target_invocations_skipped"] = g_Consumer007Skipped;
    report["record_count"] = g_Consumer007Count;
    report["max_records"] = CONSUMER007_MAX_RECORDS;
    report["max_invocations"] = CONSUMER007_MAX_CALLS;
    report["max_duration_ms"] = CONSUMER007_DURATION_MS;
    report["phase"] = "after original filtered-steer store in physical vehicle update; whole-step PRE/POST unqualified";
    report["sampled_actual_consumer_state"] = true;
    report["same_record_API_getters_invoked"] = false;
    report["issued_command_consumption_acknowledged"] = false;
    report["R13D_is_qualified_simulation_clock"] = false;
    report["raw_bits_order"] = "quaternion4, position3, velocity3 (u32 float bits)";
    report["actual_control_bits_order"] = "gas, brake, steer, filtered_steer (u32 float bits)";
    report["observation_bits_order"] = "steer, gas, braking (literal u32; braking is native integer)";
    report["Time_Now_is_simulation_clock"] = false;
    report["applied_input_acknowledgement"] = false;
    report["game_data_or_controls_written"] = false;
    report["temporary_code_hook_explicitly_requested"] = g_Consumer007Attempted;
    auto rows = Json::Array();
    for (uint index = 0; index < g_Consumer007Count; index++) {
        auto row = Json::Object();
        row["sequence"] = index;
        row["Time_Now_raw"] = g_Consumer007Times[index];
        row["raw_body_address"] = Text::FormatPointer(g_Consumer007Pointers[index]);
        row["R13D_step_argument_raw"] = g_Consumer007StepArguments[index];
        auto actual = Json::Array();
        auto observed = Json::Array();
        auto cached = Json::Array();
        auto raw = Json::Array();
        for (uint word = 0; word < 4; word++)
            actual.Add(Json::Value(g_Consumer007Bits[index * 20 + word]));
        for (uint word = 0; word < 3; word++) {
            observed.Add(Json::Value(g_Consumer007Bits[index * 20 + 4 + word]));
            cached.Add(Json::Value(g_Consumer007Bits[index * 20 + 7 + word]));
        }
        for (uint word = 0; word < 10; word++) {
            raw.Add(Json::Value(g_Consumer007Bits[index * 20 + 10 + word]));
        }
        row["actual_control_bits_u32"] = actual;
        row["vehicle_observed_input_bits_u32"] = observed;
        row["participant_cached_input_bits_u32"] = cached;
        row["raw_body_bits_u32"] = raw;
        rows.Add(row);
    }
    report["records"] = rows;
    @g_Consumer007Report = report;
    string path = IO::FromStorageFolder("headless-input-consumer007-" + Time::Now + ".json");
    if (IO::FileExists(path)) { print("InputConsumerProbe007 output collision; nothing overwritten"); return; }
    Json::ToFile(path, report, true);
    g_Consumer007OutputPath = path;
    g_Consumer007Saved = true;
    print("InputConsumerProbe007 saved " + path);
}

void Consumer007Arm() {
    try {
        g_Consumer007Status = "Preflight";
        g_Consumer007Base = Dev::BaseAddress();
        g_Consumer007End = Dev::BaseAddressEnd();
        g_Consumer007Target = g_Consumer007Base + CONSUMER007_RVA;
        CTrackMania@ app = cast<CTrackMania>(GetApp());
        if (app !is null && app.ManiaPlanetScriptAPI !is null)
            g_Consumer007ExeVersion = app.ManiaPlanetScriptAPI.ExeVersion;
        if (app is null || app.ManiaPlanetScriptAPI is null ||
                app.ManiaPlanetScriptAPI.ExeVersion != CONSUMER007_VERSION ||
                Meta::OpenplanetVersion() != "1.29.14" ||
                Meta::OpenplanetVersionGitCommit() != "6fd4200f") throw("Unqualified runtime version");
        if (app.InputPort is null || !app.InputPort.IsFocused) throw("Game input port must already be focused");
        if (!Consumer007WithinImage(g_Consumer007Target, 64)) throw("Target outside ASLR image");
        g_Consumer007BeforeHex = Consumer007Hex(g_Consumer007Target, 64);
        if (g_Consumer007BeforeHex != CONSUMER007_HEX ||
                Dev::FindPattern(CONSUMER007_PATTERN) != g_Consumer007Target)
            throw("Expected exact physical steering consumer unavailable");
        CSmArenaClient@ pg = cast<CSmArenaClient>(app.CurrentPlayground);
        if (pg is null || pg.GameTerminals.Length == 0) throw("No local playground player");
        auto player = cast<CSmPlayer>(pg.GameTerminals[0].GUIPlayer);
        CSmScriptPlayer@ api = player is null ? null : cast<CSmScriptPlayer>(player.ScriptAPI);
        if (api is null) throw("No local script player");
        if (Dev::GetOffsetNod(api, 0x50) !is player) throw("Managed participant identity mismatch");
        @g_Consumer007Owner = player;
        @g_Consumer007Playground = pg;
        g_Consumer007Participant = Dev::GetOffsetUint64(api, 0x50);
        if (g_Consumer007Participant < 0x10000 ||
                Dev::SafeReadInt32(g_Consumer007Participant + 0x21c) == -1)
            throw("Participant state unavailable");
        g_Consumer007EntityClass = Dev::SafeReadUint32(g_Consumer007Participant + 0x1110);
        g_Consumer007Vehicle = Dev::SafeReadUint64(g_Consumer007Participant + 0x1118);
        if (g_Consumer007EntityClass != 0x0a020000 || g_Consumer007Vehicle < 0x10000 ||
                g_Consumer007Vehicle > 0x00007fffffff0000)
            throw("Qualified local physical vehicle reference unavailable");
        g_Consumer007Pointers.Resize(CONSUMER007_MAX_RECORDS);
        g_Consumer007Times.Resize(CONSUMER007_MAX_RECORDS);
        g_Consumer007StepArguments.Resize(CONSUMER007_MAX_RECORDS);
        g_Consumer007Bits.Resize(CONSUMER007_MAX_RECORDS * 20);
        // Exactly one eight-byte, position-independent MOVSS store is relocated.
        g_Consumer007CodeHookAttempted = true;
        @g_Consumer007Hook = Dev::Hook(g_Consumer007Target, 3, "Consumer007Capture", Dev::PushRegisters::SSE);
        if (g_Consumer007Hook is null) throw("Hook creation failed");
        g_Consumer007HookEverInstalled = true;
        if (Dev::SafeReadUint8(g_Consumer007Target) != 0xe9 ||
                Dev::SafeReadUint8(g_Consumer007Target + 5) != 0x90 ||
                Dev::SafeReadUint8(g_Consumer007Target + 6) != 0x90 ||
                Dev::SafeReadUint8(g_Consumer007Target + 7) != 0x90)
            throw("Hook patch shape mismatch");
        g_Consumer007Trampoline = uint64(int64(g_Consumer007Target) + 5 +
            int64(Dev::SafeReadInt32(g_Consumer007Target + 1)));
        g_Consumer007TrampolineHex = Consumer007Hex(g_Consumer007Trampoline, 8);
        if (g_Consumer007TrampolineHex != CONSUMER007_HEX.SubStr(0, 16))
            throw("Original instruction trampoline mismatch");
        g_Consumer007Start = Time::Now;
        g_Consumer007Deadline = g_Consumer007Start + CONSUMER007_DURATION_MS;
        g_Consumer007Status = "Capturing bounded physical consumer state; no issued-input ACK";
        g_Consumer007Recording = true;
    } catch {
        g_Consumer007Recording = false;
        try { Consumer007Unhook("preflight-or-runtime-error"); } catch { }
        g_Consumer007Status = "Failed closed; inspect preflight and hook evidence";
        try { Consumer007Save(); } catch { }
    }
}

void Consumer007Send(Json::Value@ value) {
    if (g_Consumer007Client !is null) g_Consumer007Client.WriteRaw(Json::Write(value) + "\n");
}

Json::Value@ Consumer007Reply(const string &in operation) {
    auto reply = Json::Object();
    reply["protocol"] = "tmrl.input-consumer-probe/v7";
    reply["op"] = operation;
    reply["nonce"] = g_Consumer007Nonce;
    reply["status"] = g_Consumer007Status;
    reply["hook_installed"] = g_Consumer007Hook !is null;
    reply["record_count"] = g_Consumer007Count;
    return reply;
}

void Consumer007Complete(const string &in reason) {
    // Each cleanup phase is independent. A disk/socket failure cannot skip unhook.
    g_Consumer007Recording = false;
    try { Consumer007Unhook(reason); } catch {
        g_Consumer007Reason = reason + "; unhook-error";
    }
    g_Consumer007Status = g_Consumer007Restored ? "Capture ended; original code restored" : "Original-code restoration unverified";
    try { Consumer007Save(); } catch {
        g_Consumer007Status += "; report-file-write-error";
    }
    try {
        auto reply = Consumer007Reply("complete");
        reply["output_path"] = g_Consumer007OutputPath;
        reply["original_64_bytes_restored"] = g_Consumer007Restored;
        if (g_Consumer007Report !is null) reply["report"] = g_Consumer007Report;
        Consumer007Send(reply);
    } catch { }
}

void Consumer007DropClient() {
    if (g_Consumer007Client !is null) {
        try { g_Consumer007Client.Close(); } catch { }
        @g_Consumer007Client = null;
    }
}

void Consumer007Stop(const string &in reason) {
    g_Consumer007Shutdown = true;
    // Network/disk errors must never skip the original-code restoration attempt.
    try { Consumer007Unhook(reason); } catch {
        g_Consumer007Status = "Unhook failed; original code restoration unverified";
    }
    if (g_Consumer007Attempted) { try { Consumer007Save(); } catch { } }
    try { Consumer007DropClient(); } catch { }
    if (g_Consumer007Server !is null) {
        try { g_Consumer007Server.Close(); } catch { }
        @g_Consumer007Server = null;
    }
}

void OnDisabled() { Consumer007Stop("plugin-disabled"); }
void OnDestroyed() { Consumer007Stop("plugin-destroyed"); }
void OnDestroy() { Consumer007Stop("plugin-destroy"); }

void Consumer007Request(const string &in line) {
    auto request = Json::Parse(line);
    if (request.GetType() != Json::Type::Object || !request.HasKey("protocol") ||
            !request.HasKey("op") || !request.HasKey("nonce") ||
            request["protocol"].GetType() != Json::Type::String ||
            request["op"].GetType() != Json::Type::String ||
            request["nonce"].GetType() != Json::Type::String ||
            string(request["protocol"]) != "tmrl.input-consumer-probe/v7") throw("Invalid request");
    string operation = request["op"];
    string nonce = request["nonce"];
    if (nonce.Length != 32) throw("An explicit fresh 32-character nonce is required");
    if (operation == "arm") {
        if (g_Consumer007Attempted || g_Consumer007Hook !is null) throw("One arm attempt per plugin load");
        if (!request.HasKey("acknowledge_temporary_code_hook") ||
                request["acknowledge_temporary_code_hook"].GetType() != Json::Type::Boolean ||
                !bool(request["acknowledge_temporary_code_hook"]))
            throw("Explicit temporary code hook acknowledgement required");
        g_Consumer007Attempted = true;
        g_Consumer007Nonce = nonce;
        Consumer007Arm();
        if (g_Consumer007Hook is null || !g_Consumer007Recording) {
            auto failed = Consumer007Reply("failed");
            if (g_Consumer007Report !is null) failed["report"] = g_Consumer007Report;
            Consumer007Send(failed);
            return;
        }
        auto armed = Consumer007Reply("armed");
        armed["max_duration_ms"] = CONSUMER007_DURATION_MS;
        armed["max_records"] = CONSUMER007_MAX_RECORDS;
        armed["deadline_Time_Now_raw"] = g_Consumer007Deadline;
        Consumer007Send(armed);
        return;
    }
    if (!g_Consumer007Attempted || nonce != g_Consumer007Nonce) throw("Matching arm nonce required");
    if (operation == "stop") {
        if (g_Consumer007Hook !is null) Consumer007Complete("explicit-stop");
        else Consumer007Send(Consumer007Reply("stopped"));
        return;
    }
    if (operation == "status") {
        auto status = Consumer007Reply(g_Consumer007Saved ? "complete" : "status");
        status["output_path"] = g_Consumer007OutputPath;
        if (g_Consumer007Report !is null) status["report"] = g_Consumer007Report;
        Consumer007Send(status);
        return;
    }
    throw("Unknown operation");
}

void Main() {
    @g_Consumer007Server = Net::Socket();
    if (!g_Consumer007Server.Listen("127.0.0.1", 19180)) {
        g_Consumer007Status = "Loopback port unavailable; no hook installed";
        return;
    }
    g_Consumer007Status = "Idle; explicit loopback arm with nonce required";
    print("InputConsumerProbe007 idle on loopback19180; no hook installed");
    while (!g_Consumer007Shutdown) {
        yield();
        try {
            if (g_Consumer007Hook !is null) {
                CTrackMania@ app = cast<CTrackMania>(GetApp());
                string reason = "";
                if (app is null || cast<CSmArenaClient>(app.CurrentPlayground) !is g_Consumer007Playground)
                    reason = "playground-changed";
                else if (g_Consumer007Playground.GameTerminals.Length == 0 ||
                        g_Consumer007Playground.GameTerminals[0].GUIPlayer !is g_Consumer007Owner)
                    reason = "owner-changed";
                else if (app.InputPort is null || !app.InputPort.IsFocused) reason = "focus-lost";
                else if (g_Consumer007Client is null || g_Consumer007Client.IsHungUp()) reason = "client-eof";
                else if (g_Consumer007Reentered) reason = "callback-reentrancy";
                else if (g_Consumer007CallbackIdentityLost) reason = "owner-vehicle-identity-lost";
                else if (g_Consumer007Count >= CONSUMER007_MAX_RECORDS) reason = "record-cap";
                else if (g_Consumer007Calls >= CONSUMER007_MAX_CALLS) reason = "invocation-cap";
                else if (Time::Now >= g_Consumer007Deadline || !g_Consumer007Recording) reason = "duration-cap";
                if (reason.Length > 0) {
                    Consumer007Complete(reason);
                    Consumer007DropClient();
                    continue;
                }
            }
            if (g_Consumer007Client is null) {
                @g_Consumer007Client = g_Consumer007Server.Accept();
                if (g_Consumer007Client is null) continue;
                g_Consumer007ClientDeadline = Time::Now + 5000;
                auto hello = Consumer007Reply("hello");
                hello["one_arm_attempt_per_plugin_load"] = true;
                hello["max_duration_ms"] = CONSUMER007_DURATION_MS;
                hello["max_records"] = CONSUMER007_MAX_RECORDS;
                Consumer007Send(hello);
            }
            if (g_Consumer007Client.IsHungUp() ||
                    (g_Consumer007Hook is null && Time::Now >= g_Consumer007ClientDeadline)) {
                Consumer007DropClient(); continue;
            }
            if (g_Consumer007Client.Available() > 4096) throw("Request byte budget exceeded");
            if (g_Consumer007Client.IsReady() && g_Consumer007Client.Available() > 0) {
                string line = "";
                if (g_Consumer007Client.ReadLine(line)) {
                    if (line.Length > 4096) throw("Request line budget exceeded");
                    Consumer007Request(line);
                }
            }
        } catch {
            try {
                if (g_Consumer007Hook !is null) Consumer007Complete("protocol-or-runtime-error");
            } catch { }
            try { Consumer007DropClient(); } catch { }
        }
    }
    Consumer007Stop("main-exit");
}
