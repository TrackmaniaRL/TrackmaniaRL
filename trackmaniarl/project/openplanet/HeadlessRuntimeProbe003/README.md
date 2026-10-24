> Historical home-directory prefixes are shown as `<USER_HOME>` placeholders. Replace them with your explicit local paths; original private receipts are not redistributed.

# Headless Runtime Probe 003

Independent source-only folder plugin, prepared against the selected local
Openplanet API metadata. Static validation is not an AngelScript compile test.
Preparation does not install or load this plugin or modify TrackmaniaRL Connect.

When explicitly loaded by the operator, `Main()` performs one read-only snapshot
and writes a new `headless-runtime-probe003-<Time::Now>.json` inside this plugin's
own storage. Existing files are not replaced. There is no menu step or continuous
collector. The console receives only this probe's success/failure message.

Capture includes actual runtime Openplanet version/commit, the executing plugin's
identity/status, up to128 loaded plugin identifiers/versions/status, the game
`ManiaPlanetScriptAPI.ExeVersion`, readable input-port configuration and counters,
up to16 input-device types/status, one GUI-player ScriptAPI input/physical sample,
and reflection metadata for12 explicitly requested types. Device names/identifiers,
player/account identifiers, logs, credentials and network data are not read.

All game fields are read. This source contains no `Dev`, network, authentication,
input-injection, reset, input-device poll, game-setting assignment or hook call.
The snapshot does not claim atomicity or a physical input-reader phase. Reflection
sizes/offsets remain their declared raw integers, with no guessed memory units.

The selected Core API declares runtime loading independently of Connect:

```
Meta::LoadPlugin(absoluteFolderPath, Meta::PluginSource::UserFolder, Meta::PluginType::Folder)
```

The operator can instead load the fresh folder/`.op` through Openplanet's developer
interface. Use an independent identifier such as `headlessRuntimeProbe003`; the
actual identifier returned by `Meta::ExecutingPlugin()` is saved in the result.
Runtime loading compiles the source in that installation. No standalone compiler
has been verified. `Meta::ReloadPlugin` and `Meta::UnloadPlugin` also exist in the
selected API; this collector never calls any of those lifecycle methods itself.

Selected Core metadata also declares `Dev::Hook`/`Unhook`, procedure interception,
and raw/offset read/write functions. They are not needed for the readable settings
and are deliberately absent from this collector. No `Dump*` or compile function
was found among that dump's global Core functions; this does not prove the absence
of a developer UI capability or an external compiler.

Read-only local validation:

```
python -m scripts.headless_runtime_probe_validate --api-root <USER_HOME>/OpenplanetNext
```

After an actual load, retain the fresh output, the source/manifest hashes and a
separate host-side record of the loaded EXE/Openplanet module identities. A disk
API dump stamp or an EXE file hash alone does not prove which binary was loaded.
