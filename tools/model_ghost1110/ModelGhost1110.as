// Visual playback/export only. No controller, training, teacher or physicalization.
const string ReplayPath = "Replays/My Replays/TrackmaniaRL-Model-17.715.Replay.Gbx";
bool Saved = false;

void Main() {
    Saved = true;
    trace("MODEL_GHOST1110 READY externally verified local replay " + ReplayPath);
    auto app = cast<CTrackMania>(GetApp());
    for (uint i = 0; i < app.Switcher.ModuleStack.Length; i++) {
        auto viewer = cast<CGameCtnMediaClipViewer>(app.Switcher.ModuleStack[i]);
        if (viewer !is null) trace("MODEL_GHOST1110 CLIENT viewer_speed=" + viewer.PlaySpeed + " status=" + int(viewer.ViewerStatus));
    }
    if (app.CurrentPlayground !is null && app.CurrentPlayground.ReplayRecord !is null) {
        auto record = app.CurrentPlayground.ReplayRecord;
        trace("MODEL_GHOST1110 CLIENT duration=" + record.Duration + " ghost_count=" + record.Ghosts.Length);
    }
}

void CloseView() { cast<CTrackMania>(GetApp()).BackToMainMenu(); }

void Watch() {
    if (!Saved) return;
    auto app = cast<CTrackMania>(GetApp());
    app.BackToMainMenu();
    uint start = Time::Now;
    while (Time::Now - start < 20000) {
        auto menu = app.MenuManager.MenuCustom_CurrentManiaApp;
        if (menu !is null && menu.TitleControl.IsReady) {
            trace("MODEL_GHOST1110 VIEW " + ReplayPath);
            menu.TitleControl.ViewReplay(ReplayPath); return;
        }
        yield();
    }
    warn("MODEL_GHOST1110 FAIL replay menu readiness");
}

void RenderMenu() {
    if (UI::MenuItem("Watch model 17.715 (headless ghost)", "", false, Saved)) startnew(Watch);
    if (UI::MenuItem("Close model playback")) startnew(CloseView);
}
