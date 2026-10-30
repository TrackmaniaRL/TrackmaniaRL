//! Diagnostic serializer of already recorded controls; acceptance requires a real server replay.
use std::{fs, io::Write, path::Path};
use gbx::{container::{Container, RACE_TIME_CHUNK, write_gbx, write_result},
          tape::{Encoding, Packet, StateEnc, Tape, unpack_word_pub}};
use serde_json::{json, Value};

fn require(ok: bool, why: &str) -> Result<(), String> {
    if ok { Ok(()) } else { Err(why.to_owned()) }
}
fn sha(raw: &[u8]) -> String {
    gbx::sha::sha256(raw).iter().map(|v| format!("{v:02x}")).collect()
}
fn fresh(path: &Path, raw: &[u8]) -> Result<(), String> {
    fs::OpenOptions::new().write(true).create_new(true).open(path)
        .and_then(|mut f| f.write_all(raw)).map_err(|e| e.to_string())
}
fn bits(value: &Value) -> Result<u32, String> {
    value.as_u64().and_then(|v| u32::try_from(v).ok()).ok_or("u32 channel bits required".into())
}
fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    require(args.len() == 4, "usage: SOURCE_GHOST OWN_CAPTURE_PLAN FRESH_OUTPUT_DIR")?;
    let source = fs::read(&args[1]).map_err(|e| e.to_string())?;
    let plan_raw = fs::read(&args[2]).map_err(|e| e.to_string())?;
    let plan: Value = serde_json::from_slice(&plan_raw).map_err(|e| e.to_string())?;
    let byte_controls = plan["schema"] == "tmrl.recorded-native-i8-controls/v1";
    require(byte_controls || plan["schema"] == "tmrl.recorded-native-controls/v1", "capture plan schema")?;
    require(plan["source_ghost_sha256"].as_str() == Some(&sha(&source)), "source hash")?;
    let out = Path::new(&args[3]); require(!out.exists(), "fresh output required")?;
    let original = Container::load(&args[1])?;
    require(original.embedded_map().is_none(), "pure ghost required")?;
    let mut tape = Tape::from_body(original.body())?; tape.verbatim_is_identity()?;
    require(tape.archives.len() == 1 && tape.n() == 16152, "pinned archive/count")?;
    require(tape.archives[0].start_offset_ms == -1510, "pinned archive clock")?;
    let records = plan["records"].as_array().ok_or("records array")?;
    require(!records.is_empty() && records.len() <= 15001, "bounded recorded controls")?;
    let mut expected_index = 151usize;
    for record in records {
        let index = record["index"].as_u64().ok_or("record index")? as usize;
        require(index == expected_index && index < 15152, "contiguous owned record range")?;
        expected_index += 1;
        let sgb = record["sgb_bits"].as_array().ok_or("channel bit array")?;
        require(sgb.len() == 3, "three channels")?;
        let steer_bits = bits(&sgb[0])?; let steer = f32::from_bits(steer_bits);
        let gas = f32::from_bits(bits(&sgb[1])?); let brake = f32::from_bits(bits(&sgb[2])?);
        require(steer.is_finite() && (-1.0..=1.0).contains(&steer)
                && (gas == 0.0 || gas == 1.0) && (brake == 0.0 || brake == 1.0), "channel domain")?;
        // Preserve the already established neutral bootstrap packet's mode2.
        // Mode12's actual engine interpretation is deliberately left to the live diagnostic.
        let mode = if byte_controls || index == 151 { 2 } else { 12 };
        let encoded_steer = if byte_controls {
            let value = record["steer_i8"].as_i64()
                .filter(|v| (-127..=127).contains(v)).ok_or("signed byte steer required")?;
            require(((value as f32) / 127.0).to_bits() == steer_bits,
                    "signed byte must reproduce actual recorded float32 steering")?;
            (value as i8 as u8) as u32
        } else if index == 151 { 0 } else { steer_bits };
        require(index != 151 || (steer == 0.0 && gas == 0.0 && brake == 0.0), "neutral bootstrap")?;
        let (word0, flags) = unpack_word_pub(mode);
        tape.archives[0].packets[index] = Packet {
            word0, flags, mode: mode as u32, state: StateEnc::Lit(mode), mouse: None,
            vsame: false, steer: encoded_steer,
            accel: gas as u32, brake: brake as u32, tri: None,
        };
    }
    let declared = &plan["computed_result"];
    require(declared["NbCheckpoints"] == 1 && declared["NbRespawns"] == 0 && declared["Score"] == 0,
            "one-CP no-respawn observed result required")?;
    let time = declared["Time"].as_i64().filter(|v| *v > 0 && *v <= 150000).ok_or("bounded observed finish")? as i32;
    let mut result = original.result().ok_or("typed result")?;
    require(result.entries.len() == 1 && result.race_ms == 160000 && result.entries[0].0 == 160000,
            "pinned placeholder result")?;
    result.race_ms = time; result.entries[0].0 = time;
    let mut body = original.body().to_vec();
    // A generated recording must not inherit the template's unrelated host
    // timestamps. Only independently captured, real UTC endpoints are allowed;
    // no simulated race duration is substituted for host elapsed time.
    let original_validation: Value = serde_json::from_str(
        &gbx::manifest::validation_manifest(original.body()).ok_or("typed validation chunk")?
    ).map_err(|e| e.to_string())?;
    let mut recording_metadata = json!(null);
    if let Some(recording) = plan.get("actual_recording") {
        require(byte_controls, "recording metadata only for proven signed-byte encoding")?;
        let start_ns = recording["start_unix_ns"].as_u64().ok_or("actual UTC start")?;
        let end_ns = recording["end_unix_ns"].as_u64().ok_or("actual UTC end")?;
        require(end_ns > start_ns && end_ns-start_ns >= time as u64*1_000_000
            && end_ns-start_ns <= 180_000_000_000, "bounded actual recording interval")?;
        require(recording["receipt_sha256"].as_str().map(|s|s.len()) == Some(64),
                "captured paced replay evidence required")?;
        let start = i32::try_from(start_ns/1_000_000_000).map_err(|e|e.to_string())?;
        let end = i32::try_from(end_ns/1_000_000_000).map_err(|e|e.to_string())?;
        let chunks: Vec<_> = original.chunks().into_iter().filter(|c|c.0==0x0309_202D).collect();
        require(chunks.len()==1, "unique typed validation metadata")?;
        let at = chunks[0].2;
        let size = chunks[0].3;
        require(size>=28 && body[at..at+4]==[0,0,0,0], "supported external-input metadata")?;
        let version_len = u32::from_le_bytes(body[at+4..at+8].try_into().unwrap()) as usize;
        let clock_at = at.checked_add(20).and_then(|v|v.checked_add(version_len)).ok_or("metadata offset")?;
        require(clock_at+8<=at+size, "typed metadata clock bounds")?;
        require(i32::from_le_bytes(body[clock_at..clock_at+4].try_into().unwrap()) as i64
                    == original_validation["walltime_start"].as_i64().ok_or("source wall start")?
                && i32::from_le_bytes(body[clock_at+4..clock_at+8].try_into().unwrap()) as i64
                    == original_validation["walltime_end"].as_i64().ok_or("source wall end")?,
                "independent typed wall clock readback")?;
        body[clock_at..clock_at+4].copy_from_slice(&start.to_le_bytes());
        body[clock_at+4..clock_at+8].copy_from_slice(&end.to_le_bytes());
        recording_metadata = json!({"start_unix_seconds":start,"end_unix_seconds":end,
            "source":recording,"kind":"actual-host-UTC-from-paced-diagnostic"});
    }
    let times = original.declared_times(); require(!times.is_empty(), "declared copies")?;
    for (at, value) in times {
        require(value == 160000 && original.chunks().iter().any(|c| c.0 == RACE_TIME_CHUNK && c.2 == at && c.3 == 4),
                "typed declared copy")?;
        body[at..at+4].copy_from_slice(&time.to_le_bytes());
    }
    body = write_result(&body, &result)?;
    let body = tape.splice_into(&body, Encoding::Explicit)?;
    fs::create_dir(out).map_err(|e| e.to_string())?;
    let path = out.join("generated.Ghost.Gbx");
    write_gbx(&original.gbx, body.clone(), path.to_str().ok_or("utf8 output")?)?;
    let disk = Container::load(path.to_str().unwrap())?;
    require(disk.body() == body && disk.gbx.header_bytes_u() == original.gbx.header_bytes_u(), "body/header readback")?;
    let roundtrip = Tape::from_body(disk.body())?;
    require(roundtrip.archives.len() == 1 && roundtrip.n() == tape.n(), "input readback count")?;
    for record in records {
        let index = record["index"].as_u64().unwrap() as usize;
        let actual = &roundtrip.archives[0].packets[index];
        let expected = &tape.archives[0].packets[index];
        require(actual.word0 == expected.word0 && actual.flags == expected.flags && actual.mode == expected.mode
                && actual.steer == expected.steer && actual.accel == expected.accel && actual.brake == expected.brake
                && !actual.respawn(), "literal input readback")?;
    }
    require(disk.result().ok_or("result readback")?.encode() == result.encode(), "typed result readback")?;
    let manifest = json!({"schema":"tmrl.generated-control-codec/v1", "source_sha256":sha(&source),
        "plan_sha256":sha(&plan_raw),"ghost_sha256":sha(&fs::read(&path).map_err(|e|e.to_string())?),
        "recorded_controls":records.len(),"first_index":151,"end_exclusive":expected_index,
        "declared_observed_ms":time,"mode12_engine_semantics_qualified":false,
        "encoding":if byte_controls {"signed-i8-mode2"} else {"unqualified-mode12"},
        "template_validation_metadata":original_validation,"actual_recording_metadata":recording_metadata,
        "native_validation_runs":0,"training_qualified":false});
    fresh(&out.join("manifest.json"), &serde_json::to_vec_pretty(&manifest).map_err(|e|e.to_string())?)?;
    println!("{}", manifest); Ok(())
}
fn main() {
    if let Err(error) = run() { eprintln!("generated-control codec refused: {error}"); std::process::exit(2); }
}
