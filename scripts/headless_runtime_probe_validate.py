"""Read-only static API check for the separately prepared runtime probe source.

This neither compiles AngelScript nor loads a plugin, starts a game, reads game
logs or installs files. Only the three selected generated API metadata files and
the two probe source files are read. Optional output is created exclusively.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

PROBE = (
    Path(__file__).resolve().parents[1] / "trackmaniarl/project/openplanet/HeadlessRuntimeProbe003"
)
CORE_FUNCTIONS = (
    ("", "GetApp"),
    ("Json", "Object"),
    ("Json", "Array"),
    ("Json", "ToFile"),
    ("IO", "FromStorageFolder"),
    ("IO", "FileExists"),
    ("Meta", "ExecutingPlugin"),
    ("Meta", "AllPlugins"),
    ("Meta", "OpenplanetVersion"),
    ("Meta", "OpenplanetVersionGitCommit"),
    ("Reflection", "GetType"),
    ("Reflection", "TypeOf"),
)
GAME_MEMBERS = {
    "CGameApp": ("InputPort", "TimeSinceInitMs"),
    "CGameCtnApp": ("CurrentPlayground", "PlaygroundScript"),
    "CTrackMania": ("ManiaPlanetScriptAPI",),
    "CGameManiaPlanetScriptAPI": ("ExeVersion",),
    "CInputPort": (
        "FakeInputLagAvg",
        "FakeInputLagVar",
        "MaxSampleRate",
        "PollingEnabled",
        "MinHistoryLength",
        "IsFocused",
        "IgnoreFocusForGamePads",
        "EventInStoreCount",
        "DevicePlugEventCount",
        "StatsLatestEvent",
        "StatsDInputEventsLastFrame",
        "StatsDInputEventsCount",
        "StatsDInputTimedEventsCount",
        "StatsDInputWrongTimestampAhead",
        "StatsDInputWrongTimestampLate",
        "StatsDInputWrongTimestampAvgDelta",
        "StatsNbMappedInputsReceived",
        "ConnectedDevices",
    ),
    "CInputDevice": (
        "IsDisabled",
        "InputNotAvailable",
        "IsUnPlugged",
        "MustBePolled",
        "ObjectCount",
    ),
    "CSmArenaClient": ("GameTerminals",),
    "CGameTerminal": ("GUIPlayer",),
    "CSmPlayer": ("ScriptAPI",),
    "CSmScriptPlayer": (
        "InputSteer",
        "InputGasPedal",
        "InputIsBraking",
        "Position",
        "Velocity",
        "StartTime",
    ),
    "CSmArenaRulesMode": ("Now",),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _member(classes: dict[str, Any], type_name: str, member_name: str) -> dict[str, Any] | None:
    visited = set()
    while type_name in classes and type_name not in visited:
        visited.add(type_name)
        current = classes[type_name]
        found = [member for member in current.get("m", []) if member.get("n") == member_name]
        if found:
            return {"declaring_type": type_name, "member": member_name, "type": found[0].get("t")}
        type_name = current.get("p", "")
    return None


def validate(api_root: Path) -> dict[str, Any]:
    paths = {
        name: api_root / name
        for name in ("OpenplanetCore.json", "OpenplanetNext.json", "Openplanet.h")
    }
    core = json.loads(paths["OpenplanetCore.json"].read_bytes())
    engine = json.loads(paths["OpenplanetNext.json"].read_bytes())
    header = paths["Openplanet.h"].read_text(encoding="utf-8")
    source = (PROBE / "Main.as").read_text(encoding="utf-8")
    functions = {(item.get("ns", ""), item["name"]): item for item in core["functions"]}
    missing_functions = ["::".join(key) for key in CORE_FUNCTIONS if key not in functions]
    classes = {
        name: value for namespace in engine["ns"].values() for name, value in namespace.items()
    }
    checked_members, missing_members = [], []
    for name, members in GAME_MEMBERS.items():
        for member in members:
            found = _member(classes, name, member)
            if found is None:
                missing_members.append(name + "." + member)
            else:
                checked_members.append(found)
    forbidden = [
        item
        for item in (
            "Dev::",
            "Net::",
            "Auth::",
            "ReadHardwareCurState",
            "DeviceHotPlugUpdate",
            ".InstanceId",
            ".InstanceName",
            ".DeviceModelId",
            ".LocalUser",
            ".Login",
            ".AccountId",
        )
        if item in source
    ]
    writes = re.findall(r"\b(?:port|device|app|api|terminal|player)\.\w+\s*=(?!=)", source)
    capability_names = {
        "LoadPlugin",
        "ReloadPlugin",
        "UnloadPlugin",
        "Hook",
        "Unhook",
        "InterceptProc",
        "ResetInterceptProc",
    }
    capabilities = [
        {"namespace": item.get("ns", ""), "decl": item["decl"]}
        for item in core["functions"]
        if item["name"] in capability_names
    ]
    compile_apis = [item["decl"] for item in core["functions"] if "compile" in item["name"].lower()]
    stamps_match = engine["op"] in core["op"] and engine["mp"] in header and core["op"] in header
    return {
        "schema": "tmrl.runtime-probe-static-validation/v1",
        "passed": not (missing_functions or missing_members or forbidden or writes)
        and stamps_match,
        "source_id": "headlessRuntimeProbe003",
        "game_launched": False,
        "plugin_installed": False,
        "plugin_loaded": False,
        "angel_script_compiled": False,
        "runtime_capture_performed": False,
        "api_dump_stamp": core["op"],
        "engine_dump_stamp": engine["mp"],
        "api_dump_stamps_match": stamps_match,
        "api_dump_files": {
            name: {"path": str(path), "sha256": _sha(path)} for name, path in paths.items()
        },
        "source_files": {
            name: {"path": str(PROBE / name), "sha256": _sha(PROBE / name)}
            for name in ("Main.as", "info.toml")
        },
        "checked_core_function_count": len(CORE_FUNCTIONS),
        "missing_core_functions": missing_functions,
        "checked_game_members": checked_members,
        "missing_game_members": missing_members,
        "forbidden_calls_found": forbidden,
        "game_property_assignments_found": writes,
        "runtime_load_and_hook_declarations": capabilities,
        "compile_functions_in_selected_core_dump": compile_apis,
        "standalone_compiler_verified": False,
        "loaded_module_evidence": None,
        "note": (
            "Static declaration/source validation only; "
            "runtime compilation and output remain unverified."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.api_root)
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    print(encoded, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
