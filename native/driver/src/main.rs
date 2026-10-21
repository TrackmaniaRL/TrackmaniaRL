//! Retained early validator parent, bounded fork children, native G2/V4; no probe or fallback.
use fk::{
    session::{Checkpoint, Engine, Session},
    tape::Tape,
};
use forkoracle::forksrv::{parse_ready, read_frame, write_frame, BranchReq};
use serde_json::{json, Value};
use std::io::{BufRead, Write};
use std::os::fd::AsRawFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::time::{Duration, Instant};
use std::{
    fs::File,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
};
fn write(p: &Path, b: impl AsRef<[u8]>) -> Result<(), String> {
    if p.exists() {
        return Err("refuses overwrite".into());
    }
    std::fs::write(p, b).map_err(|e| e.to_string())
}
fn mem(pid: i32, a: u64, n: usize) -> Result<Vec<u8>, String> {
    readable(pid, a, n as u64)?;
    let f = File::open(format!("/proc/{}/mem", pid)).map_err(|e| e.to_string())?;
    let mut b = vec![0; n];
    f.read_exact_at(&mut b, a)
        .map_err(|e| format!("read {:x}: {}", a, e))?;
    Ok(b)
}
fn u64at(b: &[u8], i: usize) -> u64 {
    u64::from_le_bytes(b[i..i + 8].try_into().unwrap())
}
fn u32at(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(b[i..i + 4].try_into().unwrap())
}
fn loads(elf: &[u8]) -> Vec<(u64, u64, u64)> {
    let ph = u64at(elf, 32) as usize;
    let size = u16::from_le_bytes(elf[54..56].try_into().unwrap()) as usize;
    let count = u16::from_le_bytes(elf[56..58].try_into().unwrap()) as usize;
    (0..count)
        .filter_map(|i| {
            let at = ph + i * size;
            if u32at(elf, at) == 1 {
                Some((u64at(elf, at + 8), u64at(elf, at + 16), u64at(elf, at + 32)))
            } else {
                None
            }
        })
        .collect()
}
fn readable(pid: i32, a: u64, n: u64) -> Result<(), String> {
    let end = a.checked_add(n).ok_or("address overflow")?;
    let maps = std::fs::read_to_string(format!("/proc/{}/maps", pid)).map_err(|e| e.to_string())?;
    if maps.lines().any(|line| {
        let x: Vec<&str> = line.split_whitespace().collect();
        let r: Vec<&str> = x[0].split('-').collect();
        x[1].starts_with('r')
            && u64::from_str_radix(r[0], 16).unwrap() <= a
            && end <= u64::from_str_radix(r[1], 16).unwrap()
    }) {
        Ok(())
    } else {
        Err(format!("unreadable range {:x}/{}", a, n))
    }
}

#[derive(Debug, PartialEq, Eq)]
struct EarlyOwned {
    controller: u64,
    sim: u64,
    playground: u64,
    players: u64,
    participant: u64,
    vehicle: u64,
}
fn ptr(pid: i32, a: u64) -> Result<u64, String> {
    let p = u64at(&mem(pid, a, 8)?, 0);
    if p < 4096 || p >= 0x0000800000000000 || p % 8 != 0 {
        return Err(format!("invalid pointer at{:x}: {:x}", a, p));
    }
    Ok(p)
}
fn owned_at(pid: i32, controller: u64, sim: u64) -> Result<EarlyOwned, String> {
    if controller < 4096
        || sim < 4096
        || controller >= 0x0000800000000000
        || sim >= 0x0000800000000000
    {
        return Err("missing validator capture".into());
    }
    if ptr(pid, controller + 0x1a70)? != sim {
        return Err("controller captured sim mismatch".into());
    }
    let playground = ptr(pid, sim + 0x18)?;
    if u32at(&mem(pid, playground + 0x668, 4)?, 0) != 1 {
        return Err("single validator player required".into());
    }
    let players = ptr(pid, playground + 0x660)?;
    let participant = ptr(pid, players)?;
    if u32at(&mem(pid, participant + 0x1110, 4)?, 0) != 0x032e2000 {
        return Err("primary vehicle class mismatch".into());
    }
    let vehicle = ptr(pid, participant + 0x1118)?;
    Ok(EarlyOwned {
        controller,
        sim,
        playground,
        players,
        participant,
        vehicle,
    })
}
fn owned(s: &Session) -> Result<EarlyOwned, String> {
    owned_at(
        s.srv.pid(),
        s.srv.validator_controller,
        s.srv.validation_sim,
    )
}
extern "C" {
    fn kill(pid: i32, sig: i32) -> i32;
    fn getsockopt(fd: i32, level: i32, option: i32, value: *mut i32, len: *mut u32) -> i32;
}
fn birth(pid: i32) -> Option<String> {
    let s = std::fs::read_to_string(format!("/proc/{}/stat", pid)).ok()?;
    let tail = &s[s.rfind(')')? + 1..];
    tail.split_whitespace().nth(19).map(str::to_owned)
}
struct Branch {
    pid: i32,
    birth: String,
    sock: UnixStream,
    dir: PathBuf,
    stdout_start: usize,
}
impl Drop for Branch {
    fn drop(&mut self) {
        if birth(self.pid).as_deref() == Some(&self.birth) {
            unsafe {
                kill(self.pid, 9);
            }
        }
    }
}
fn emit(v: Value) {
    println!("{}", v);
    std::io::stdout().flush().unwrap();
}
fn guard(pid: i32, e: &EarlyOwned, base: u64, n: usize) -> Result<(Vec<u8>, Vec<u8>), String> {
    if owned_at(pid, e.controller, e.sim)? != *e {
        return Err("ownership changed".into());
    }
    let mut b = mem(pid, e.sim + 0x48, 4)?;
    b.extend(mem(pid, e.vehicle + 0x12e0, 40)?);
    b.extend(mem(pid, e.vehicle + 0x15d8, 8)?);
    b.extend(mem(pid, e.vehicle + 0x1608, 12)?);
    b.extend(mem(pid, e.vehicle + 0x98, 12)?);
    b.extend(mem(pid, e.vehicle + 0x1430, 4)?);
    Ok((
        b,
        mem(
            pid,
            base.checked_sub(4).ok_or("input underflow")?,
            n.checked_mul(32).ok_or("input overflow")?,
        )?,
    ))
}
fn go(sock: &mut UnixStream, clock: u64, physical: u64, path: &Path) -> Result<(), String> {
    let pb = path.to_str().ok_or("utf8path")?.as_bytes();
    let mut p = Vec::new();
    p.push(b'G');
    p.extend(2u32.to_le_bytes());
    p.extend(1u64.to_le_bytes());
    p.extend(200000u32.to_le_bytes());
    p.extend(44u32.to_le_bytes());
    p.extend(0u32.to_le_bytes());
    p.extend(0u64.to_le_bytes());
    p.extend(0u32.to_le_bytes());
    p.extend(0u32.to_le_bytes());
    p.extend((pb.len() as u32).to_le_bytes());
    p.extend(pb);
    for (a, n) in [(clock, 4u32), (physical, 40u32)] {
        p.extend(a.to_le_bytes());
        p.extend(n.to_le_bytes());
    }
    write_frame(sock, &p).map_err(|e| e.to_string())?;
    if read_frame(sock).as_deref() != Some(b"GO") {
        return Err("child GO failed".into());
    }
    Ok(())
}
// Preview203: path configuration only. No physics/wire/ownership/deadline change.
fn canonical_existing(path: &Path, directory: bool) -> Result<(), String> {
    if !path.is_absolute()
        || path.components().any(|c| {
            matches!(
                c,
                std::path::Component::ParentDir | std::path::Component::CurDir
            )
        })
    {
        return Err("absolute canonical path required".into());
    }
    if std::fs::canonicalize(path).map_err(|e| e.to_string())? != path {
        return Err("path alias rejected".into());
    }
    for part in path.ancestors() {
        if std::fs::symlink_metadata(part)
            .map_err(|e| e.to_string())?
            .file_type()
            .is_symlink()
        {
            return Err("symlink ancestor rejected".into());
        }
    }
    if directory && !path.is_dir() || !directory && !path.is_file() {
        return Err("wrong path type".into());
    }
    Ok(())
}
fn runtime_root() -> Result<PathBuf, String> {
    let root = PathBuf::from(
        std::env::var_os("TMRE_RUNTIME_ROOT").ok_or("explicit TMRE_RUNTIME_ROOT required")?,
    );
    canonical_existing(&root, true)?;
    Ok(root)
}
fn fresh_work(root: &Path, work: &Path) -> Result<(), String> {
    if !work.is_absolute()
        || work.exists()
        || !work.starts_with(root.join("build/server-probe"))
        || work.components().any(|c| {
            matches!(
                c,
                std::path::Component::ParentDir | std::path::Component::CurDir
            )
        })
    {
        return Err("fresh bounded work required".into());
    }
    let existing = work
        .ancestors()
        .find(|p| p.exists())
        .ok_or("existing work ancestor required")?;
    canonical_existing(existing, true)?;
    Ok(())
}
fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        return Err("normalstart-bootstrap FRESH_WORK EXISTING_OUT".into());
    }
    let root = runtime_root()?;
    let work = PathBuf::from(&args[1]);
    let out = PathBuf::from(&args[2]);
    fresh_work(&root, &work)?;
    canonical_existing(&out, true)?;
    let tape_path = PathBuf::from(&args[3]);
    let shim_path = PathBuf::from(&args[4]);
    canonical_existing(&tape_path, false)?;
    canonical_existing(&shim_path, false)?;
    if !tape_path.starts_with(root.join("analysis/tmrl-shortcut"))
        || !shim_path.starts_with(root.join("build/forkoracle"))
    {
        return Err("bounded tape/shim required".into());
    }
    let assets = root.join("analysis/tmrl-shortcut/assets-001");
    let engine = Engine {
        server: root.join("build/server-probe/linux-validate-map2-001"),
        map: assets.join("tmrl-test.Map.Gbx"),
        shim: shim_path,
        work: work.join("root"),
        work_is_temporary: false,
    };
    let tape = Tape::load(tape_path.to_str().unwrap())?;
    tape.codec_is_lossless()?;
    let mut session = Session::start(&engine, tape, Checkpoint::Clock(1000))?;
    let pid = session.srv.pid();
    write(&out.join("ready.json"),serde_json::to_vec_pretty(&json!({"server_pid":pid,"requested_lroundf_checkpoint":1000,"ready_lroundf":session.srv.clock,"input_base":session.srv.base,"controller":session.srv.validator_controller,"sim":session.srv.validation_sim,"probe_called":false,"clock_locator_called":false,"keylocate_and_tape_verification_called":true})).unwrap())?;
    let early = owned(&session)?;
    let participant = early.participant;
    let elf = std::fs::read(engine.server.join("TrackmaniaServer")).map_err(|e| e.to_string())?;
    if &elf[..4] != b"\x7fELF" {
        return Err("ELF required".into());
    }
    let segments = loads(&elf);
    let zero: Vec<_> = segments.iter().filter(|(off, _, _)| *off == 0).collect();
    if zero.len() != 1 {
        return Err("unique offset0 ELF load required".into());
    }
    let maps = std::fs::read_to_string(format!("/proc/{}/maps", pid)).map_err(|e| e.to_string())?;
    let mut bases = Vec::new();
    for line in maps.lines() {
        let x: Vec<&str> = line.split_whitespace().collect();
        if x.len() >= 6
            && x[5].ends_with("/TrackmaniaServer")
            && u64::from_str_radix(x[2], 16).ok() == Some(0)
        {
            let lo = u64::from_str_radix(x[0].split('-').next().unwrap(), 16)
                .map_err(|e| e.to_string())?;
            bases.push(lo - zero[0].1);
        }
    }
    if bases.len() != 1 {
        return Err("unique main module required".into());
    }
    let base = bases[0];
    let vptr = ptr(pid, participant)?;
    if vptr != base + 0x1c33938 {
        return Err("CSmPlayer vptr mismatch".into());
    }
    let getid = ptr(pid, vptr + 0x20)?;
    let descriptor = ptr(pid, vptr + 0x18)?;
    if getid != base + 0x121d190 || descriptor != base + 0x121b970 {
        return Err("getter slots mismatch".into());
    }
    let mut proofs = Vec::new();
    for rva in [0x121d190u64, 0x121b970] {
        let segment = segments
            .iter()
            .find(|(_, va, n)| *va <= rva && rva + 16 <= va + n)
            .ok_or("getter outside ELF")?;
        let off = (segment.0 + rva - segment.1) as usize;
        let live = mem(pid, base + rva, 16)?;
        if live != elf[off..off + 16] {
            return Err("live getter bytes differ".into());
        }
        proofs.push(json!({"rva":rva,"bytesHex":live.iter().map(|v|format!("{:02x}",v)).collect::<String>()}));
    }
    let simclock = early.sim + 0x48;
    let physical = early.vehicle + 0x12e0;
    let c1 = mem(pid, simclock, 4)?;
    let p1 = mem(pid, physical, 40)?;
    let f: Vec<f32> = p1
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let qnorm = f[..4]
        .iter()
        .map(|v| (*v as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    if !f.iter().all(|v| v.is_finite()) || (qnorm - 1.).abs() > 0.001 {
        return Err("invalid early physical state".into());
    }
    let again = owned(&session)?;
    let c2 = mem(pid, simclock, 4)?;
    let p2 = mem(pid, physical, 40)?;
    if early != again || c1 != c2 || p1 != p2 {
        return Err("paused reads changed".into());
    }
    write(
        &out.join("initial.snapshot.bin"),
        [c1.clone(), p1.clone()].concat(),
    )?;
    let metadata = json!({"schema":"tmre.early-owned-bootstrap/v2","server_pid":pid,"controller":early.controller,"sim":early.sim,"playground":early.playground,"players":early.players,"participant":participant,"vehicle":early.vehicle,"input_base":session.srv.base,"input_count":session.tape.n(),"start_offset_ms":session.tape.start_offset_ms,"requested_lroundf_checkpoint":1000,"ready_lroundf":session.srv.clock,"native_sim_clock":u32at(&c1,0),"native_sim_clock_address":simclock,"physical_address":physical,"module_slide":base,"participant_vptr":vptr,"getid":getid,"descriptor_getter":descriptor,"getter_instruction_proofs":proofs,"qnorm":qnorm,"state_f32":f,"same_owned_chain_clock_physical_two_reads":true,"probe_calls":0,"clock_locator_calls":0,"qualify_calls":0,"late_fallback":false,"host_data_memory_writes":false,"shim_action_capability":true,"clock_source":"NativeSimClock = exact sim+0x48; not old Gclock or race time","sampler":"NativeSimState52: u64hook/u32NativeSimClock/physical40","segments":[[simclock,4],[physical,40]],"snapshot_scope":"partial readonly 44B clock/physical; NOT full snapshot/endtick/reset","normalstart_reset_verified":false});
    write(
        &out.join("metadata.json"),
        serde_json::to_vec_pretty(&metadata).unwrap(),
    )?;

    let tree_dir = PathBuf::from(std::env::var("TMRE_TREE_DIR").map_err(|_| "tree dir required")?);
    if !tree_dir.starts_with("/tmp") || !tree_dir.is_dir() {
        return Err("existing private tmp tree dir required".into());
    }
    let tree_path = tree_dir.join("branch.sock");
    if tree_path.exists() {
        return Err("fresh branch socket required".into());
    }
    let listener = UnixListener::bind(&tree_path).map_err(|e| e.to_string())?;
    listener.set_nonblocking(true).map_err(|e| e.to_string())?;
    let baseline = guard(pid, &early, session.srv.base, session.tape.n())?;
    write(&out.join("parent.guard.bin"), &baseline.0)?;
    write(&out.join("parent.inputs.bin"), &baseline.1)?;
    let stdout_path = session.srv.dir.join("stdout.log");
    let original_prefix = std::fs::read(&stdout_path).map_err(|e| e.to_string())?;
    if !["", "["].contains(
        &std::str::from_utf8(&original_prefix)
            .map_err(|e| e.to_string())?
            .trim(),
    ) {
        return Err("unexpected parent stdout prefix".into());
    }
    write(&out.join("stdout-prefix.bin"), &original_prefix)?;
    let mut active: Option<Branch> = None;
    emit(
        json!({"kind":"ready","parent_pid":pid,"metadata":metadata,"driver_pid":std::process::id()}),
    );
    for line in std::io::stdin().lock().lines() {
        let line = line.map_err(|e| e.to_string())?;
        if line.len() > 4096 {
            return Err("oversize command".into());
        }
        let cmd: Value = serde_json::from_str(&line).map_err(|e| e.to_string())?;
        match cmd["op"].as_str().ok_or("op required")? {
            "spawn" => {
                if active.is_some() {
                    return Err("active child exists".into());
                }
                let number = cmd["episode"].as_u64().ok_or("episode required")?;
                if number > 100000 {
                    return Err("episode out of bounds".into());
                }
                let ep = out.join(format!("episode-{:04}", number));
                if ep.exists() {
                    return Err("episode directory exists".into());
                }
                std::fs::create_dir(&ep).map_err(|e| e.to_string())?;
                if guard(pid, &early, session.srv.base, session.tape.n())? != baseline {
                    return Err("parent changed before spawn".into());
                }
                let stdout_start = std::fs::metadata(&stdout_path)
                    .map_err(|e| e.to_string())?
                    .len() as usize;
                let socket = tree_path.to_str().unwrap();
                let req = BranchReq {
                    from: 0,
                    recs: &[],
                    stop_after_lroundf: 1,
                    sock: socket,
                    trace_path: "",
                    segs: &[],
                    sample_stride: 1,
                    sample_max: 0,
                    key: (0, 0),
                };
                let child_pid = session.srv.branch(&req)?;
                if child_pid <= 0 {
                    return Err("invalid branch pid".into());
                }
                let deadline = Instant::now() + Duration::from_secs(90);
                let mut sock = loop {
                    match listener.accept() {
                        Ok((s, _)) => break s,
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            if Instant::now() > deadline {
                                unsafe {
                                    kill(child_pid, 9);
                                }
                                return Err("child accept timeout".into());
                            }
                            std::thread::sleep(Duration::from_millis(1));
                        }
                        Err(e) => return Err(e.to_string()),
                    }
                };
                sock.set_read_timeout(Some(Duration::from_secs(180)))
                    .map_err(|e| e.to_string())?;
                sock.set_write_timeout(Some(Duration::from_secs(90)))
                    .map_err(|e| e.to_string())?;
                let mut cred = [0i32; 3];
                let mut len = 12u32;
                if unsafe { getsockopt(sock.as_raw_fd(), 1, 17, cred.as_mut_ptr(), &mut len) } != 0
                    || len != 12
                    || cred[0] != child_pid
                {
                    unsafe {
                        kill(child_pid, 9);
                    }
                    return Err("branch peer mismatch".into());
                }
                let hello = read_frame(&mut sock).ok_or("missing READY")?;
                let (base, clock, named_pid) =
                    parse_ready(std::str::from_utf8(&hello).map_err(|e| e.to_string())?)?;
                if base != session.srv.base || clock != 1001 || named_pid != Some(child_pid) {
                    unsafe {
                        kill(child_pid, 9);
                    }
                    return Err("branch ready mismatch".into());
                }
                active = Some(Branch {
                    pid: child_pid,
                    birth: birth(child_pid).ok_or("missing child")?,
                    sock,
                    dir: ep.clone(),
                    stdout_start,
                });
                let child = active.as_mut().unwrap();
                if owned_at(child_pid, early.controller, early.sim)? != early
                    || ptr(child_pid, participant)? != vptr
                    || ptr(child_pid, vptr + 0x20)? != getid
                    || ptr(child_pid, vptr + 0x18)? != descriptor
                {
                    return Err("child typed identity mismatch".into());
                }
                let child_guard = guard(child_pid, &early, base, session.tape.n())?;
                write(&ep.join("initial.guard.bin"), &child_guard.0)?;
                if child_guard.1 != baseline.1 {
                    return Err("child initial tape mismatch".into());
                }
                write(&ep.join("ready.bin"), &hello)?;
                let mut child_meta = metadata.clone();
                child_meta["server_pid"] = json!(child_pid);
                child_meta["ready_lroundf"] = json!(clock);
                child_meta["native_sim_clock"] = json!(u32at(&child_guard.0, 0));
                child_meta["parent_pid"] = json!(pid);
                child_meta["stdout_start"] = json!(stdout_start);
                write(
                    &ep.join("metadata.json"),
                    serde_json::to_vec_pretty(&child_meta).unwrap(),
                )?;
                go(&mut child.sock, simclock, physical, &ep.join("samples.bin"))?;
                emit(
                    json!({"kind":"spawned","episode":number,"pid":child_pid,"clock":clock,"directory":ep,"metadata":child_meta}),
                );
            }
            "reap" | "truncate" => {
                let mut child = active.take().ok_or("no active child")?;
                let truncated = cmd["op"] == "truncate";
                if truncated && birth(child.pid).as_deref() == Some(&child.birth) {
                    unsafe {
                        kill(child.pid, 9);
                    }
                }
                let deadline = Instant::now() + Duration::from_secs(180);
                while birth(child.pid).as_deref() == Some(&child.birth) {
                    if Instant::now() > deadline {
                        return Err("child exit timeout".into());
                    }
                    std::thread::sleep(Duration::from_millis(1));
                }
                let stdout = std::fs::read(&stdout_path).map_err(|e| e.to_string())?;
                if stdout.len() < child.stdout_start {
                    return Err("stdout shrank".into());
                }
                write(
                    &child.dir.join("stdout-slice.bin"),
                    &stdout[child.stdout_start..],
                )?;
                let after = guard(pid, &early, session.srv.base, session.tape.n())?;
                if after != baseline {
                    return Err("retained parent changed".into());
                }
                write(&child.dir.join("parent-after.guard.bin"), &after.0)?;
                write(&child.dir.join("parent-after.inputs.bin"), &after.1)?;
                let result = json!({"kind":"reaped","pid":child.pid,"truncated_by_driver":truncated,"parent_unchanged":true,"process_gone":true,"exit_status":null,"exit_status_reason":"server parent SIGCHLD ignored; driver is not waitpid owner","stdout_start":child.stdout_start,"stdout_end":stdout.len()});
                write(
                    &child.dir.join("result.json"),
                    serde_json::to_vec_pretty(&result).unwrap(),
                )?;
                child.birth.clear();
                emit(result);
            }
            "close" => {
                if active.is_some() {
                    return Err("cannot close parent with active child".into());
                }
                if guard(pid, &early, session.srv.base, session.tape.n())? != baseline {
                    return Err("parent changed at close".into());
                }
                emit(json!({"kind":"closed","parent_unchanged":true}));
                break;
            }
            _ => return Err("unknown op".into()),
        }
    }
    drop(active);
    drop(listener);
    std::fs::remove_file(&tree_path).map_err(|e| e.to_string())?;
    Ok(())
}
