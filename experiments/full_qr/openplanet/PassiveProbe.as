// Bounded diagnostic observer. It has no input, reset, map-load or memory-write API.
// Its separate socket emits only the 23 channels consumed by the historical V5
// encoder and the actual UI Finish flag. It never claims to emit Connect33.
Net::Socket@ g_probeServer;
Net::Socket@ g_probeClient;

void OnDestroy() {
    if (g_probeClient !is null) g_probeClient.Close();
    if (g_probeServer !is null) g_probeServer.Close();
}

string Number(float value) { return Text::Format("%.9g", value); }
string V3(vec3 value) {
    return Number(value.x) + "," + Number(value.y) + "," + Number(value.z);
}

void Main() {
    const uint started = Time::Now;
    @g_probeServer = Net::Socket();
    if (!g_probeServer.Listen("127.0.0.1", 19041)) return;
    print("FullQRPassive041: bounded read-only observer started");
    uint sequence = 0;
    while (Time::Now - started < 90000) {
        yield();
        if (g_probeClient is null) {
            @g_probeClient = g_probeServer.Accept();
            continue;
        }
        if (g_probeClient.IsHungUp()) break;
        auto app = cast<CTrackMania>(GetApp());
        auto pg = app is null ? null : cast<CSmArenaClient>(app.CurrentPlayground);
        if (pg is null || pg.GameTerminals.Length == 0 || pg.Map is null) continue;
        auto terminal = pg.GameTerminals[0];
        auto player = cast<CSmPlayer>(terminal.GUIPlayer);
        auto api = player is null ? null : cast<CSmScriptPlayer>(player.ScriptAPI);
        auto vis = VehicleState::ViewingPlayerState();
        if (api is null || vis is null || pg.Map.MapInfo is null) continue;
        auto rules = cast<CSmArenaRulesMode>(app.PlaygroundScript);
        int64 now = rules is null
            ? int64(app.Network.PlaygroundClientScriptAPI.GameTime) : int64(rules.Now);
        float raceTime = api.StartTime > 0 && now > api.StartTime ? float(now - api.StartTime) : 0.0f;
        bool finished = terminal.UISequence_Current == CGamePlaygroundUIConfig::EUISequence::Finish;
        string frame = Number(raceTime) + "," + V3(api.Position) + "," + V3(api.Velocity)
            + "," + V3(vis.Dir) + "," + Number(api.Speed) + "," + Number(api.EngineRpm)
            + "," + api.EngineCurGear + "," + Number(vis.FLSlipCoef) + "," + Number(vis.FRSlipCoef)
            + "," + Number(vis.RLSlipCoef) + "," + Number(vis.RRSlipCoef)
            + "," + api.WheelsSkiddingCount + "," + api.FlyingDuration
            + "," + Number(api.AdherenceCoef) + "," + Number(api.InputSteer)
            + "," + Number(api.InputGasPedal) + "," + (api.InputIsBraking ? "1" : "0");
        string packet = "{\"schema\":\"tmrl.windows-passive/v1\",\"sequence\":" + sequence++
            + ",\"map_uid\":\"" + pg.Map.MapInfo.MapUid + "\",\"finished\":"
            + (finished ? "true" : "false") + ",\"frame23\":[" + frame + "]}\n";
        if (!g_probeClient.WriteRaw(packet)) break;
    }
    OnDestroy();
    print("FullQRPassive041: observer closed");
}
