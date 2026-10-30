"""Source only: actual autonomous17.712s poses and acknowledged model commands."""
import json,math,struct,tarfile,argparse,hashlib
from pathlib import Path
from typing import Any
import numpy as np
from tools.ghost_divergence1103.prepare_archived_inputs import audited_controls

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read_json(path): return json.loads(path.read_text(encoding='utf-8-sig'))

def extract(study: Path) -> dict[str, Any]:
    evaluation = study / "autonomous-line1114/final-evaluation1116"
    result = read_json(evaluation / "evaluation-result.json")
    assert result["status"] == "PASS_FIVE_AUTONOMOUS_GREEDY_TRIALS"
    assert result["expert_file_access_denied"]
    assert result["updates_during_evaluation"] == 0
    assert result["trials"][-1]["finished"]
    assert result["trials"][-1]["finish_time_s"] == 17.712
    archive = next((evaluation / "closed-traces").rglob("session-00000006.tar.gz"))
    receipt = read_json(archive.with_suffix("").with_suffix(".json"))
    assert sha(archive) == receipt["archive_sha256"]
    with tarfile.open(archive, "r:gz") as source:

        def member(name: str) -> bytes:
            stream = source.extractfile(name)
            assert stream is not None
            return stream.read()

        meta = json.loads(member("trace/episode-0000/metadata.json"))
        reset = json.loads(member("trace/episode-0000/uncredited-neutral-reset-v7.json"))
        pid = int(meta["server_pid"])
        origin = int(reset["resetNativeT"])
        assert origin == 2200
        posts = member(f"trace/posts/post-{pid}.bin")
        goals = member(f"trace/posts/goal-receipt-{pid}.bin")
    assert len(posts) % 256 == len(goals) % 96 == 0
    commits = audited_controls(posts, pid)
    frames: list[dict[str, Any]] = []
    for at in range(0, len(posts), 256):
        row = posts[at : at + 256]
        if row[:8] != b"TMPOST03" or struct.unpack_from("<I", row, 148)[0] != 75:
            continue
        assert struct.unpack_from("<4I", row, 8) == (256, 3, pid, pid)
        clock = struct.unpack_from("<I", row, 112)[0]
        if clock < origin:
            continue
        assert clock == struct.unpack_from("<I", row, 116)[0]
        physical = struct.unpack_from("<10f", row, 152)
        gas, brake, steer = struct.unpack_from("<3f", row, 88)
        if clock > origin:
            assert commits[clock] == struct.pack("<3f", steer, gas, brake)
        assert all(math.isfinite(x) for x in physical)
        frames.append(
            {
                "t": clock - origin,
                "q": physical[:4],
                "p": physical[4:7],
                "v": physical[7:10],
                "sgb": [steer, gas, brake],
            }
        )
    assert [f["t"] for f in frames] == list(range(0, 17751, 10))
    goal_times = set()
    for at in range(0, len(goals), 96):
        row = goals[at : at + 96]
        assert row[:8] == b"TMGR0001"
        assert struct.unpack_from("<4I", row, 8) == (96, 1, pid, pid)
        _, last_goal, count = struct.unpack_from("<QII", row, 40)
        if count:
            goal_times.add(last_goal - origin)
    assert goal_times == {17712}
    # Confirm that archived controls are the actual network decisions of trial5.
    actions_file = evaluation / "actions-and-outcomes.jsonl"
    actions = [json.loads(line)["action"] for line in actions_file.read_text().splitlines()]
    assert len(actions) == 5 * 355
    with np.load(study / "training-data1091/macro-catalog.npz", allow_pickle=False) as data:
        commands = data["programs"][actions[-355:]].reshape(-1)
    for frame, command in zip(frames[1:], commands, strict=True):
        steer, variant = divmod(int(command), 6)
        gas, brake = divmod(variant, 3)
        expected = np.asarray([(steer - 6) / 6, gas, min(brake, 1)], dtype=np.float32)
        assert np.array_equal(np.asarray(frame["sgb"], dtype=np.float32), expected)
    return {
        "frames": frames,
        "goal_ms": 17712,
        "physical_end_ms": 17750,
        "provenance": {
            "archive": str(archive),
            "archive_sha256": sha(archive),
            "policy_sha256": result["policy_only_sha256"],
            "evaluation_sha256": sha(evaluation / "evaluation-result.json"),
            "model_action_log_sha256": sha(actions_file),
            "native_pid": pid,
            "actual_model_commands_verified": len(commands),
            "teacher_files_used": False,
            "new_simulations": 0,
        },
    }



if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError('Refuse source export overwrite')
 result=extract(a.study);a.output.write_text(json.dumps(result,allow_nan=False));print(json.dumps({'status':'VERIFIED_ACTUAL_NEURAL_POLICY_EXPORT','goal_ms':result['goal_ms'],'frames':len(result['frames']),'provenance':result['provenance']}))
