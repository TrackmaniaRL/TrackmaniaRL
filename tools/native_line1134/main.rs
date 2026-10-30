//! Diagnostic input tapes for CPU native physics search, never visual replays.
//! Preserve every field outside the explicitly enumerated racing S/G/B edits.
use gbx::container::{Container, write_gbx};
use gbx::tape::{Encoding, Packet, Tape};
use serde_json::{json, Value};
use std::path::Path;

fn motors(value: &Value) -> Result<Vec<u8>, String> {
    value.as_array().ok_or("motor array")?.iter().map(|v| {
        let n = v.as_u64().ok_or("integer motor ID")?;
        if n >= 78 || n % 3 == 2 || ![0, 6, 12].contains(&(n / 6)) {
            return Err("exact digital, constant-per-10ms canonical motor IDs only".into());
        }
        Ok(n as u8)
    }).collect()
}
fn sgb(n: u8) -> (u32, u32, u32) {
    let s = match n / 6 { 0 => -127i8, 6 => 0, 12 => 127, _ => unreachable!() };
    ((s as u8) as u32, ((n % 6) / 3) as u32, (n % 3) as u32)
}
fn signature(p: &Packet) -> (u32, u32, u32, Option<(u16,u16)>, Option<[u8;4]>, u32,u32,u32) {
    (p.word0,p.flags,p.mode,p.mouse,p.tri,p.steer,p.accel,p.brake)
}
fn main() -> Result<(),String> {
    let a: Vec<String> = std::env::args().collect();
    if a.len()!=4 {return Err("native-line1134 NATIVE_SOURCE SPEC.json FRESH_DIRECTORY".into());}
    let c = Container::load(&a[1])?;
    if c.gbx.class_id!=0x03092000 || c.result().ok_or("native result missing")?.race_ms!=17712
        || !c.uids().iter().any(|(_,s)|s=="oqIJ5rQDRrNwLPTh9H2p_W4tLof") {
        return Err("bound native17.712/map99162 source only".into());
    }
    let tape = Tape::from_file(&a[1])?;
    tape.verbatim_is_identity()?;
    if tape.archives.len()!=1 || tape.archives[0].start_offset_ms!=0 || tape.n()!=2072 {
        return Err("bound observed-clock native archive required".into());
    }
    let spec:Value=serde_json::from_slice(&std::fs::read(&a[2]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    let baseline=motors(&spec["baseline"])?;
    let active=spec["active_ticks"].as_u64().ok_or("active prefix")? as usize;
    if baseline.len()!=1775 || active!=1771 {return Err("bound full racing prefix required".into());}
    let source=&tape.archives[0].packets;
    if source[0].steer_i8()!=0 || source[0].accel!=0 || source[0].brake!=0 {
        return Err("t0 must remain native neutral initialization".into());
    }
    for (i,&m) in baseline[..active].iter().enumerate() {
        let p=&source[i+1];
        if p.word0!=2 || p.flags!=0 || p.mode!=2 || p.respawn() || (p.steer,p.accel,p.brake)!=sgb(m) {
            return Err(format!("baseline/native observation-clock mismatch at motor{} -> packet{}",i,i+1));
        }
    }
    let candidates=spec["candidates"].as_array().ok_or("candidates")?;
    if candidates.is_empty() || candidates.len()>4096 {return Err("bounded candidate count 1..4096".into());}
    let root=Path::new(&a[3]);std::fs::create_dir(root).map_err(|e|e.to_string())?;
    let mut report=Vec::new();
    for item in candidates {
        let name=item["name"].as_str().ok_or("candidate name")?;
        if name.is_empty() || name.len()>80 || !name.bytes().all(|b|b.is_ascii_alphanumeric() || b==b'-') {
            return Err("simple unique candidate names only".into());
        }
        let ticks=motors(&item["ticks"])?;
        if ticks.len()!=baseline.len() || ticks[active..]!=baseline[active..] {return Err("no clock change or padding edits".into());}
        let mut edited=tape.clone();let mut changes=Vec::new();
        for i in 0..active {
            if ticks[i]!=baseline[i] {
                let p=&mut edited.archives[0].packets[i+1];
                (p.steer,p.accel,p.brake)=sgb(ticks[i]);
                changes.push(i);
            }
        }
        let output=root.join(format!("INPUTONLY-{name}.Ghost.Gbx"));
        if output.exists() {return Err("refuse candidate overwrite".into());}
        write_gbx(&c.gbx,edited.splice_into(c.body(),Encoding::Explicit)?,output.to_str().ok_or("output encoding")?)?;
        let back=Tape::from_file(output.to_str().unwrap())?;
        let metadata=Container::load(output.to_str().unwrap())?;
        if back.n()!=tape.n() || back.archives[0].start_offset_ms!=0 || metadata.result().ok_or("result lost")?.race_ms!=17712 {
            return Err("metadata, packet count or clock altered".into());
        }
        for (p,q) in back.archives[0].packets.iter().zip(&edited.archives[0].packets) {
            if signature(p)!=signature(q) {return Err("serialized input or nonvehicle fields changed".into());}
        }
        report.push(json!({"name":name,"file":output,"changed_motor_indices":changes,
            "native_packet_offset":1,"all2072PacketsReadBack":true,"timeMetadataUnchanged":true}));
    }
    std::fs::write(root.join("encoding-proof.json"),serde_json::to_vec_pretty(&json!({
        "status":"INPUT_ONLY_DIAGNOSTICS_NOT_VISUAL_REPLAYS_OR_CLAIMED_RESULTS",
        "native_packet_ms":10,"initial_packet_preserved":true,"racing_offset":1,
        "source_result_metadata_ms":17712,"actual_candidate_goal_must_come_from_native_physics":true,
        "candidates":report
    })).unwrap()).map_err(|e|e.to_string())?;
    println!("{} diagnostic input tapes, exact roundtrip",report.len());Ok(())
}
