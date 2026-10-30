// Read-only reflection metadata for the native vehicle visual-state format.
// No input, reset, settings, network, hook or Dev memory API is used.
void Main() {
    auto result = Json::Object();
    result["kind"] = "MODEL_REPLAY_VIS_SCHEMA1125_READ_ONLY";
    result["offset_semantics"] = "raw reflection values; independently verify before interpreting as byte offsets";
    const Reflection::MwClassInfo@ info;
    try { @info = Reflection::GetType("CSceneVehicleVisState"); }
    catch { result["status"] = "lookup-error"; print("VIS_SCHEMA1125:" + Json::Write(result)); return; }
    if (info is null) {
        result["status"] = "not-found";
        print("VIS_SCHEMA1125:" + Json::Write(result));
        return;
    }
    result["name"] = info.Name;
    result["id"] = info.ID;
    result["size"] = info.Size;
    auto source = info.Members;
    auto members = Json::Array();
    if (source is null || source.Length > 256) {
        result["status"] = "member-metadata-unavailable-or-over-budget";
        print("VIS_SCHEMA1125:" + Json::Write(result));
        return;
    }
    for (uint i = 0; i < source.Length; i++) {
        if (source[i] is null) continue;
        auto row = Json::Object();
        row["name"] = source[i].Name;
        row["id"] = source[i].ID;
        row["raw_offset_u16"] = source[i].Offset;
        members.Add(row);
    }
    result["members"] = members;
    result["status"] = "REFLECTION_ONLY_NOT_AN_ENCODER_OR_OFFSET_PROOF";
    print("VIS_SCHEMA1125:" + Json::Write(result));
}
