// Read-only render provenance. Does not drive, reset, patch, or send telemetry.
// The policy crop excludes the barcode and the game's clock/HUD.
uint frameSequence = 0;
void Barcode(uint value, float y) {
    for (uint bit = 0; bit < 32; bit++) {
        nvg::BeginPath();
        nvg::Rect(8 + bit * 4, y, 4, 4);
        nvg::FillColor((value & (uint(1) << bit)) != 0 ? vec4(1,1,1,1) : vec4(0,0,0,1));
        nvg::Fill();
    }
}
void Render() {
    auto app = cast<CTrackMania>(GetApp());
    if (app is null) return;
    auto pg = cast<CSmArenaClient>(app.CurrentPlayground);
    if (pg is null || pg.Map is null || pg.Map.MapInfo is null) return;
    if (pg.Map.MapInfo.MapUid != "oqIJ5rQDRrNwLPTh9H2p_W4tLof" || pg.GameTerminals.Length == 0) return;
    auto player = cast<CSmPlayer>(pg.GameTerminals[0].GUIPlayer);
    if (player is null) return;
    auto api = cast<CSmScriptPlayer>(player.ScriptAPI);
    auto rules = cast<CSmArenaRulesMode>(app.PlaygroundScript);
    if (api is null || rules is null || api.StartTime <= 0) return;
    int raceTime = int(rules.Now) - int(api.StartTime);
    uint ticks = uint(raceTime);
    frameSequence++;
    Barcode(0x1167CAFE, 8);
    Barcode(ticks, 12);
    Barcode(frameSequence, 16);
    Barcode(ticks ^ frameSequence ^ 0x1167CAFE, 20);
}
