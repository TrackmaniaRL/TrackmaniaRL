// Isolated five-second diagnostic. Installation/loading is a separate operation.
// The callback reads integer bit patterns only. It never writes game data,
// changes controls, invokes game getters, yields, formats strings, or performs I/O.
// Time::Now is integer host metadata, explicitly not a simulation clock.
// Dev::Hook itself temporarily replaces exactly seven verified code bytes.
const uint PHYSICAL005_MAX_RECORDS = 256;
const uint PHYSICAL005_MAX_CALLS = 4096;
const uint PHYSICAL005_DURATION_MS = 5000;
const uint64 PHYSICAL005_RVA = 0x7d0cb0;
const string PHYSICAL005_VERSION = "2026-02-02_17_51";
// SHA-256 of the lowercase hexadecimal form of the verified 64-byte window.
const string PHYSICAL005_HEX_SHA256 = "7af8dd5a17f1b8df59a90487dc15253b7f6af2bad9b3d24405476a76e4898b9f";

Dev::HookInfo@ g_Physical005Hook;
Net::Socket@ g_Physical005Server;
Net::Socket@ g_Physical005Client;
CSmPlayer@ g_Physical005Owner;
CSmArenaClient@ g_Physical005Playground;
Json::Value@ g_Physical005Report;
bool g_Physical005Shutdown = false;
bool g_Physical005Attempted = false;
bool g_Physical005Recording = false;
bool g_Physical005InsideCallback = false;
bool g_Physical005Saved = false;
bool g_Physical005HookEverInstalled = false;
bool g_Physical005CodeHookAttempted = false;
bool g_Physical005UnhookError = false;
bool g_Physical005Restored = false;
bool g_Physical005Reentered = false;
bool g_Physical005CallbackIdentityLost = false;
uint g_Physical005Count = 0;
uint g_Physical005Calls = 0;
uint g_Physical005Skipped = 0;
uint64 g_Physical005Base = 0;
uint64 g_Physical005End = 0;
uint64 g_Physical005Target = 0;
uint64 g_Physical005Participant = 0;
uint64 g_Physical005Slot = 0;
uint64 g_Physical005Trampoline = 0;
uint64 g_Physical005Start = 0;
uint64 g_Physical005Deadline = 0;
uint g_Physical005TypeIndex = 0;
uint64 g_Physical005ClientDeadline = 0;
string g_Physical005Nonce = "";
string g_Physical005OutputPath = "";
string g_Physical005BeforeHex = "";
string g_Physical005VerifiedPattern = "";
bool g_Physical005OriginalVerified = false;
string g_Physical005AfterHex = "";
string g_Physical005TrampolineHex = "";
string g_Physical005ExeVersion = "";
uint64[] g_Physical005Pointers;
uint64[] g_Physical005Times;
uint[] g_Physical005Bits;
string g_Physical005Status = "Source loaded; not armed";
string g_Physical005Reason = "not-started";

bool Physical005WithinImage(uint64 ptr, uint64 bytes) {
    return ptr >= g_Physical005Base && ptr <= g_Physical005End &&
        bytes <= g_Physical005End - ptr;
}

string Physical005Hex(uint64 ptr, uint bytes) {
    string result = "";
    for (uint i = 0; i < bytes; i++)
        result += Text::Format("%02x", uint(Dev::SafeReadUint8(ptr + i)));
    return result;
}

// Build both representations from the same bounded memory snapshot. The pattern
// is used only after its hexadecimal form has passed the pinned SHA-256 guard.
void Physical005ReadSignature(uint64 ptr) {
    string observed = "";
    string pattern = "";
    for (uint i = 0; i < 64; i++) {
        uint value = uint(Dev::SafeReadUint8(ptr + i));
        observed += Text::Format("%02x", value);
        if (i != 0) pattern += " ";
        pattern += Text::Format("%02X", value);
    }
    if (observed.Length != 128 || Crypto::Sha256(observed) != PHYSICAL005_HEX_SHA256)
        throw("Original instruction hash mismatch");
    g_Physical005BeforeHex = observed;
    g_Physical005VerifiedPattern = pattern;
    g_Physical005OriginalVerified = true;
}

// Parameters are bound by their x86 register names, as documented by Dev::Hook.
// Its trampoline has already executed the stolen seven-byte MOVUPS load.
// RDX/R8/R9 are unchanged and no original destination store has run yet.
void Physical005Capture(uint64 rdx, uint64 r8, uint64 r9) {
    if (!g_Physical005Recording) return;
    if (g_Physical005InsideCallback) {
        g_Physical005Reentered = true;
        g_Physical005Recording = false;
        return;
    }
    g_Physical005InsideCallback = true;
    uint64 now = Time::Now;
    if (now >= g_Physical005Deadline || g_Physical005Calls >= PHYSICAL005_MAX_CALLS ||
            g_Physical005Count >= PHYSICAL005_MAX_RECORDS) {
        g_Physical005Recording = false;
        g_Physical005InsideCallback = false;
        return;
    }
    g_Physical005Calls++;
    if (Dev::SafeReadUint64(g_Physical005Slot) != g_Physical005Target) {
        g_Physical005CallbackIdentityLost = true;
        g_Physical005Recording = false;
        g_Physical005InsideCallback = false;
        return;
    }
    if (r8 != g_Physical005Participant + 0xe14 || r9 != g_Physical005Participant + 0xe30 ||
            rdx < 0x10000 || rdx > 0x00007fffffff0000) {
        g_Physical005Skipped++;
        g_Physical005InsideCallback = false;
        return;
    }
    uint index = g_Physical005Count;
    g_Physical005Times[index] = now;
    g_Physical005Pointers[index * 3] = rdx;
    g_Physical005Pointers[index * 3 + 1] = r8;
    g_Physical005Pointers[index * 3 + 2] = r9;
    for (uint word = 0; word < 10; word++)
        g_Physical005Bits[index * 20 + word] = Dev::SafeReadUint32(rdx + 0x12e0 + word * 4);
    for (uint word = 0; word < 7; word++)
        g_Physical005Bits[index * 20 + 10 + word] = Dev::SafeReadUint32(r8 + word * 4);
    for (uint word = 0; word < 3; word++)
        g_Physical005Bits[index * 20 + 17 + word] = Dev::SafeReadUint32(r9 + word * 4);
    g_Physical005Count = index + 1;
    if (g_Physical005Count >= PHYSICAL005_MAX_RECORDS) g_Physical005Recording = false;
    g_Physical005InsideCallback = false;
}

void Physical005Unhook(const string &in reason) {
    g_Physical005Recording = false;
    g_Physical005Reason = reason;
    if (g_Physical005Hook !is null) {
        try {
            Dev::Unhook(g_Physical005Hook);
            @g_Physical005Hook = null;
        } catch { g_Physical005UnhookError = true; }
    }
    // Observe the target even when Hook threw or returned a null handle.
    if (g_Physical005CodeHookAttempted && Physical005WithinImage(g_Physical005Target, 64)) {
        g_Physical005AfterHex = Physical005Hex(g_Physical005Target, 64);
        g_Physical005Restored = g_Physical005AfterHex == g_Physical005BeforeHex &&
            g_Physical005OriginalVerified && Crypto::Sha256(g_Physical005BeforeHex) == PHYSICAL005_HEX_SHA256 && g_Physical005Hook is null &&
            !g_Physical005UnhookError;
    }
}

void Physical005Save() {
    if (g_Physical005Saved || g_Physical005InsideCallback) return;
    auto report = Json::Object();
    report["schema"] = "tmrl.headless-physical-copy-probe/v5";
    report["collector_id"] = "headlessPhysicalProbe005";
    report["collector_version"] = "0.0.5";
    report["nonce"] = g_Physical005Nonce;
    report["status"] = g_Physical005Status;
    report["stop_reason"] = g_Physical005Reason;
    report["expected_executable_sha256"] = "3fc7d8cda542beda131c44306b123f4004d07d7e22f512b46b762afc29f6edda";
    report["binary_sha256_checked_by_plugin"] = false;
    report["expected_exe_version"] = PHYSICAL005_VERSION;
    report["observed_exe_version"] = g_Physical005ExeVersion;
    report["openplanet_version"] = Meta::OpenplanetVersion();
    report["openplanet_git_commit"] = Meta::OpenplanetVersionGitCommit();
    report["base_address"] = Text::FormatPointer(g_Physical005Base);
    report["base_address_end"] = Text::FormatPointer(g_Physical005End);
    report["target_address"] = Text::FormatPointer(g_Physical005Target);
    report["target_rva"] = "0x7d0cb0";
    report["expected_original_64_hex"] = g_Physical005OriginalVerified ? g_Physical005BeforeHex : "";
    report["observed_before_64_hex"] = g_Physical005BeforeHex;
    report["observed_after_unhook_64_hex"] = g_Physical005AfterHex;
    report["exact_original_prefix_verified"] = g_Physical005OriginalVerified && Crypto::Sha256(g_Physical005BeforeHex) == PHYSICAL005_HEX_SHA256;
    report["padding"] = 2;
    report["original_instruction_bytes_copied"] = 7;
    report["trampoline_address"] = Text::FormatPointer(g_Physical005Trampoline);
    report["observed_original_trampoline_7_hex"] = g_Physical005TrampolineHex;
    report["requested_push_registers"] = "SSE (1), XMM0..15 save/restore in pinned implementation";
    report["fp_control_preservation"] = "required by Windows x64 ABI; not measured by this collector";
    report["hook_ever_installed"] = g_Physical005HookEverInstalled;
    report["code_hook_attempted"] = g_Physical005CodeHookAttempted;
    report["unhook_error"] = g_Physical005UnhookError;
    report["unhooked"] = g_Physical005Hook is null;
    report["original_64_bytes_restored"] = g_Physical005Restored;
    report["callback_reentered"] = g_Physical005Reentered;
    report["callback_pointer_identity_lost"] = g_Physical005CallbackIdentityLost;
    report["callback_type_index"] = g_Physical005TypeIndex;
    report["verified_callback_slot_address"] = Text::FormatPointer(g_Physical005Slot);
    report["expected_participant_address"] = Text::FormatPointer(g_Physical005Participant);
    report["start_Time_Now_raw"] = g_Physical005Start;
    report["deadline_Time_Now_raw"] = g_Physical005Deadline;
    report["callback_invocations"] = g_Physical005Calls;
    report["non_target_invocations_skipped"] = g_Physical005Skipped;
    report["record_count"] = g_Physical005Count;
    report["max_records"] = PHYSICAL005_MAX_RECORDS;
    report["max_invocations"] = PHYSICAL005_MAX_CALLS;
    report["max_duration_ms"] = PHYSICAL005_DURATION_MS;
    report["phase"] = "after original first quaternion load; before every original output store";
    report["destination_samples_are_PRE_copy"] = true;
    report["raw_bits_order"] = "quaternion4, position3, velocity3 (u32 float bits)";
    report["destination_bits_order"] = "pose quaternion4, pose position3, velocity3 (u32 float bits)";
    report["Time_Now_is_simulation_clock"] = false;
    report["applied_input_acknowledgement"] = false;
    report["game_data_or_controls_written"] = false;
    report["temporary_code_hook_explicitly_requested"] = g_Physical005Attempted;
    auto rows = Json::Array();
    for (uint index = 0; index < g_Physical005Count; index++) {
        auto row = Json::Object();
        row["sequence"] = index;
        row["Time_Now_raw"] = g_Physical005Times[index];
        row["raw_body_address"] = Text::FormatPointer(g_Physical005Pointers[index * 3]);
        row["pose_destination_address"] = Text::FormatPointer(g_Physical005Pointers[index * 3 + 1]);
        row["velocity_destination_address"] = Text::FormatPointer(g_Physical005Pointers[index * 3 + 2]);
        auto raw = Json::Array();
        auto previous = Json::Array();
        for (uint word = 0; word < 10; word++) {
            raw.Add(Json::Value(g_Physical005Bits[index * 20 + word]));
            previous.Add(Json::Value(g_Physical005Bits[index * 20 + 10 + word]));
        }
        row["raw_body_bits_u32"] = raw;
        row["destination_PRE_copy_bits_u32"] = previous;
        rows.Add(row);
    }
    report["records"] = rows;
    @g_Physical005Report = report;
    string path = IO::FromStorageFolder("headless-physical-probe005-" + Time::Now + ".json");
    if (IO::FileExists(path)) { print("PhysicalProbe005 output collision; nothing overwritten"); return; }
    Json::ToFile(path, report, true);
    g_Physical005OutputPath = path;
    g_Physical005Saved = true;
    print("PhysicalProbe005 saved " + path);
}

void Physical005Arm() {
    try {
        g_Physical005Status = "Preflight";
        g_Physical005Base = Dev::BaseAddress();
        g_Physical005End = Dev::BaseAddressEnd();
        g_Physical005Target = g_Physical005Base + PHYSICAL005_RVA;
        CTrackMania@ app = cast<CTrackMania>(GetApp());
        if (app !is null && app.ManiaPlanetScriptAPI !is null)
            g_Physical005ExeVersion = app.ManiaPlanetScriptAPI.ExeVersion;
        if (app is null || app.ManiaPlanetScriptAPI is null ||
                app.ManiaPlanetScriptAPI.ExeVersion != PHYSICAL005_VERSION ||
                Meta::OpenplanetVersion() != "1.29.14" ||
                Meta::OpenplanetVersionGitCommit() != "6fd4200f") throw("Unqualified runtime version");
        if (app.InputPort is null || !app.InputPort.IsFocused) throw("Game input port must already be focused");
        if (!Physical005WithinImage(g_Physical005Target, 64)) throw("Target outside ASLR image");
        Physical005ReadSignature(g_Physical005Target);
        if (Dev::FindPattern(g_Physical005VerifiedPattern) != g_Physical005Target)
            throw("Expected exact physical copy function unavailable");
        CSmArenaClient@ pg = cast<CSmArenaClient>(app.CurrentPlayground);
        if (pg is null || pg.GameTerminals.Length == 0) throw("No local playground player");
        auto player = cast<CSmPlayer>(pg.GameTerminals[0].GUIPlayer);
        CSmScriptPlayer@ api = player is null ? null : cast<CSmScriptPlayer>(player.ScriptAPI);
        if (api is null) throw("No local script player");
        if (Dev::GetOffsetNod(api, 0x50) !is player) throw("Managed participant identity mismatch");
        @g_Physical005Owner = player;
        @g_Physical005Playground = pg;
        g_Physical005Participant = Dev::GetOffsetUint64(api, 0x50);
        if (g_Physical005Participant < 0x10000 ||
                Dev::SafeReadInt32(g_Physical005Participant + 0x21c) == -1)
            throw("Participant state unavailable");
        if (!Physical005WithinImage(g_Physical005Base + 0x20540d8, 4)) throw("Type index outside image");
        g_Physical005TypeIndex = Dev::SafeReadUint32(g_Physical005Base + 0x20540d8);
        if (g_Physical005TypeIndex >= 128) throw("Unqualified entity type index");
        uint64 table = Dev::SafeReadUint64(g_Physical005Base + 0x1fab3f0);
        g_Physical005Slot = table + uint64(g_Physical005TypeIndex) * 0x120 + 0x80;
        if (!Physical005WithinImage(g_Physical005Slot, 8) ||
                Dev::SafeReadUint64(g_Physical005Slot) != g_Physical005Target)
            throw("Physical callback table identity unproved");
        g_Physical005Pointers.Resize(PHYSICAL005_MAX_RECORDS * 3);
        g_Physical005Times.Resize(PHYSICAL005_MAX_RECORDS);
        g_Physical005Bits.Resize(PHYSICAL005_MAX_RECORDS * 20);
        // Exactly one seven-byte, position-independent MOVUPS is relocated.
        // Recheck the same verified snapshot immediately before modifying code.
        if (Physical005Hex(g_Physical005Target, 64) != g_Physical005BeforeHex)
            throw("Original instruction window changed before hook");
        g_Physical005CodeHookAttempted = true;
        @g_Physical005Hook = Dev::Hook(g_Physical005Target, 2, "Physical005Capture", Dev::PushRegisters::SSE);
        if (g_Physical005Hook is null) throw("Hook creation failed");
        g_Physical005HookEverInstalled = true;
        if (Dev::SafeReadUint8(g_Physical005Target) != 0xe9 ||
                Dev::SafeReadUint8(g_Physical005Target + 5) != 0x90 ||
                Dev::SafeReadUint8(g_Physical005Target + 6) != 0x90)
            throw("Hook patch shape mismatch");
        g_Physical005Trampoline = uint64(int64(g_Physical005Target) + 5 +
            int64(Dev::SafeReadInt32(g_Physical005Target + 1)));
        g_Physical005TrampolineHex = Physical005Hex(g_Physical005Trampoline, 7);
        if (g_Physical005TrampolineHex != g_Physical005BeforeHex.SubStr(0, 14))
            throw("Original instruction trampoline mismatch");
        g_Physical005Start = Time::Now;
        g_Physical005Deadline = g_Physical005Start + PHYSICAL005_DURATION_MS;
        g_Physical005Status = "Capturing bounded physical-copy PRE state";
        g_Physical005Recording = true;
    } catch {
        g_Physical005Recording = false;
        try { Physical005Unhook("preflight-or-runtime-error"); } catch { }
        g_Physical005Status = "Failed closed; inspect preflight and hook evidence";
        try { Physical005Save(); } catch { }
    }
}

void Physical005Send(Json::Value@ value) {
    if (g_Physical005Client !is null) g_Physical005Client.WriteRaw(Json::Write(value) + "\n");
}

Json::Value@ Physical005Reply(const string &in operation) {
    auto reply = Json::Object();
    reply["protocol"] = "tmrl.physical-probe/v5";
    reply["op"] = operation;
    reply["nonce"] = g_Physical005Nonce;
    reply["status"] = g_Physical005Status;
    reply["hook_installed"] = g_Physical005Hook !is null;
    reply["record_count"] = g_Physical005Count;
    return reply;
}

void Physical005Complete(const string &in reason) {
    // Each cleanup phase is independent. A disk/socket failure cannot skip unhook.
    g_Physical005Recording = false;
    try { Physical005Unhook(reason); } catch {
        g_Physical005Reason = reason + "; unhook-error";
    }
    g_Physical005Status = g_Physical005Restored ? "Capture ended; original code restored" : "Original-code restoration unverified";
    try { Physical005Save(); } catch {
        g_Physical005Status += "; report-file-write-error";
    }
    try {
        auto reply = Physical005Reply("complete");
        reply["output_path"] = g_Physical005OutputPath;
        reply["original_64_bytes_restored"] = g_Physical005Restored;
        if (g_Physical005Report !is null) reply["report"] = g_Physical005Report;
        Physical005Send(reply);
    } catch { }
}

void Physical005DropClient() {
    if (g_Physical005Client !is null) {
        try { g_Physical005Client.Close(); } catch { }
        @g_Physical005Client = null;
    }
}

void Physical005Stop(const string &in reason) {
    g_Physical005Shutdown = true;
    // Network/disk errors must never skip the original-code restoration attempt.
    try { Physical005Unhook(reason); } catch {
        g_Physical005Status = "Unhook failed; original code restoration unverified";
    }
    if (g_Physical005Attempted) { try { Physical005Save(); } catch { } }
    try { Physical005DropClient(); } catch { }
    if (g_Physical005Server !is null) {
        try { g_Physical005Server.Close(); } catch { }
        @g_Physical005Server = null;
    }
}

void OnDisabled() { Physical005Stop("plugin-disabled"); }
void OnDestroyed() { Physical005Stop("plugin-destroyed"); }
void OnDestroy() { Physical005Stop("plugin-destroy"); }

void Physical005Request(const string &in line) {
    auto request = Json::Parse(line);
    if (request.GetType() != Json::Type::Object || !request.HasKey("protocol") ||
            !request.HasKey("op") || !request.HasKey("nonce") ||
            request["protocol"].GetType() != Json::Type::String ||
            request["op"].GetType() != Json::Type::String ||
            request["nonce"].GetType() != Json::Type::String ||
            string(request["protocol"]) != "tmrl.physical-probe/v5") throw("Invalid request");
    string operation = request["op"];
    string nonce = request["nonce"];
    if (nonce.Length != 32) throw("An explicit fresh 32-character nonce is required");
    if (operation == "arm") {
        if (g_Physical005Attempted || g_Physical005Hook !is null) throw("One arm attempt per plugin load");
        if (!request.HasKey("acknowledge_temporary_code_hook") ||
                request["acknowledge_temporary_code_hook"].GetType() != Json::Type::Boolean ||
                !bool(request["acknowledge_temporary_code_hook"]))
            throw("Explicit temporary code hook acknowledgement required");
        g_Physical005Attempted = true;
        g_Physical005Nonce = nonce;
        Physical005Arm();
        if (g_Physical005Hook is null || !g_Physical005Recording) {
            auto failed = Physical005Reply("failed");
            if (g_Physical005Report !is null) failed["report"] = g_Physical005Report;
            Physical005Send(failed);
            return;
        }
        auto armed = Physical005Reply("armed");
        armed["max_duration_ms"] = PHYSICAL005_DURATION_MS;
        armed["max_records"] = PHYSICAL005_MAX_RECORDS;
        armed["deadline_Time_Now_raw"] = g_Physical005Deadline;
        Physical005Send(armed);
        return;
    }
    if (!g_Physical005Attempted || nonce != g_Physical005Nonce) throw("Matching arm nonce required");
    if (operation == "stop") {
        if (g_Physical005Hook !is null) Physical005Complete("explicit-stop");
        else Physical005Send(Physical005Reply("stopped"));
        return;
    }
    if (operation == "status") {
        auto status = Physical005Reply(g_Physical005Saved ? "complete" : "status");
        status["output_path"] = g_Physical005OutputPath;
        if (g_Physical005Report !is null) status["report"] = g_Physical005Report;
        Physical005Send(status);
        return;
    }
    throw("Unknown operation");
}

void Main() {
    @g_Physical005Server = Net::Socket();
    if (!g_Physical005Server.Listen("127.0.0.1", 19178)) {
        g_Physical005Status = "Loopback port unavailable; no hook installed";
        return;
    }
    g_Physical005Status = "Idle; explicit loopback arm with nonce required";
    print("PhysicalProbe005 idle on loopback19178; no hook installed");
    while (!g_Physical005Shutdown) {
        yield();
        try {
            if (g_Physical005Hook !is null) {
                CTrackMania@ app = cast<CTrackMania>(GetApp());
                string reason = "";
                if (app is null || cast<CSmArenaClient>(app.CurrentPlayground) !is g_Physical005Playground)
                    reason = "playground-changed";
                else if (g_Physical005Playground.GameTerminals.Length == 0 ||
                        g_Physical005Playground.GameTerminals[0].GUIPlayer !is g_Physical005Owner)
                    reason = "owner-changed";
                else if (app.InputPort is null || !app.InputPort.IsFocused) reason = "focus-lost";
                else if (g_Physical005Client is null || g_Physical005Client.IsHungUp()) reason = "client-eof";
                else if (g_Physical005Reentered) reason = "callback-reentrancy";
                else if (g_Physical005CallbackIdentityLost) reason = "callback-identity-lost";
                else if (g_Physical005Count >= PHYSICAL005_MAX_RECORDS) reason = "record-cap";
                else if (g_Physical005Calls >= PHYSICAL005_MAX_CALLS) reason = "invocation-cap";
                else if (Time::Now >= g_Physical005Deadline || !g_Physical005Recording) reason = "duration-cap";
                if (reason.Length > 0) {
                    Physical005Complete(reason);
                    Physical005DropClient();
                    continue;
                }
            }
            if (g_Physical005Client is null) {
                @g_Physical005Client = g_Physical005Server.Accept();
                if (g_Physical005Client is null) continue;
                g_Physical005ClientDeadline = Time::Now + 5000;
                auto hello = Physical005Reply("hello");
                hello["one_arm_attempt_per_plugin_load"] = true;
                hello["max_duration_ms"] = PHYSICAL005_DURATION_MS;
                hello["max_records"] = PHYSICAL005_MAX_RECORDS;
                Physical005Send(hello);
            }
            if (g_Physical005Client.IsHungUp() ||
                    (g_Physical005Hook is null && Time::Now >= g_Physical005ClientDeadline)) {
                Physical005DropClient(); continue;
            }
            if (g_Physical005Client.Available() > 4096) throw("Request byte budget exceeded");
            if (g_Physical005Client.IsReady() && g_Physical005Client.Available() > 0) {
                string line = "";
                if (g_Physical005Client.ReadLine(line)) {
                    if (line.Length > 4096) throw("Request line budget exceeded");
                    Physical005Request(line);
                }
            }
        } catch {
            try {
                if (g_Physical005Hook !is null) Physical005Complete("protocol-or-runtime-error");
            } catch { }
            try { Physical005DropClient(); } catch { }
        }
    }
    Physical005Stop("main-exit");
}
