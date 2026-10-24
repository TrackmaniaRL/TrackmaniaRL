//! Actual 9ce390 prepared-contact delivery, with one emulated RET.
//! No contact flags, motor callbacks or complete event coverage are invented.
//! The batch clock is native callback coverage, NOT future physical-event coverage.
use super::*;
include!("prepared_impact_signature.rs");
static BASE: AtomicUsize = AtomicUsize::new(0);
static PID: AtomicUsize = AtomicUsize::new(0);
static TID: AtomicUsize = AtomicUsize::new(0);
static FD: AtomicI32 = AtomicI32::new(-1);
static SEQ: AtomicUsize = AtomicUsize::new(0);
static LAST_CALLBACK: AtomicUsize = AtomicUsize::new(0);
static LAST_EVENT: AtomicUsize = AtomicUsize::new(0);
const RET_RVA: usize = 0x9ce395;

pub unsafe fn configure(base: usize) {
    let mut actual = [0u8; 20];
    if !read(base + RET_RVA, &mut actual) || !same(&actual, &IMPACT_RET_SIGNATURE)
        || IMPACT_RET_SIGNATURE[0] != 0xc3 { fail(930) }
    BASE.store(base, Ordering::SeqCst);
}

pub unsafe fn activate() {
    let base = BASE.load(Ordering::SeqCst);
    if base == 0 { return }
    let at = base + RET_RVA;
    let mut actual = [0u8; 20];
    if !read(at, &mut actual) || !same(&actual, &IMPACT_RET_SIGNATURE) { fail(931) }
    let page = at & !4095usize;
    if sc(10, page, 4096, 7, 0, 0, 0) != 0 { fail(932) }
    core::ptr::write_volatile(at as *mut u8, 0xcc);
    if sc(10, page, 4096, 5, 0, 0, 0) != 0 { fail(933) }
}

unsafe fn output(pid: usize, tid: usize) -> i32 {
    if PID.load(Ordering::SeqCst) != pid {
        let old = FD.swap(-1, Ordering::SeqCst);
        if old >= 0 { sc(3, old as usize, 0, 0, 0, 0, 0); }
        let n = DIR_LEN.load(Ordering::SeqCst);
        if n == 0 || n > 240 { fail(934) }
        let mut path = [0u8; 320];
        copy_volatile(&(&*core::ptr::addr_of!(DIR))[..n], &mut path[..n]);
        let prefix = b"/prepared-impact-";
        copy_volatile(prefix, &mut path[n..n+prefix.len()]);
        let mut digits = [0u8; 24];
        let mut x = pid; let mut count = 0;
        loop { digits[count] = b'0' + (x % 10) as u8; count += 1; x /= 10; if x == 0 { break } }
        let start = n + prefix.len();
        for i in 0..count { path[start+i] = digits[count-i-1]; }
        copy_volatile(b".bin", &mut path[start+count..start+count+4]);
        let fd = sc(257, (-100isize) as usize, path.as_ptr() as usize, 0x800c1, 0o600, 0, 0);
        if fd < 0 { fail(935) }
        PID.store(pid, Ordering::SeqCst); TID.store(tid, Ordering::SeqCst);
        FD.store(fd as i32, Ordering::SeqCst); SEQ.store(0, Ordering::SeqCst);
        LAST_CALLBACK.store(0, Ordering::SeqCst); LAST_EVENT.store(0, Ordering::SeqCst);
    }
    if TID.load(Ordering::SeqCst) != tid { fail(936) }
    FD.load(Ordering::SeqCst)
}

unsafe fn identity(base: usize) -> (usize, usize, usize, u32) {
    let sim = VALIDATOR_SIM.load(Ordering::SeqCst);
    let controller = VALIDATOR_CONTROLLER.load(Ordering::SeqCst);
    let clock = need(dword(sim.wrapping_add(0x48)), 937);
    let arena = need(word(sim.wrapping_add(0x18)), 937);
    let players = need(word(arena.wrapping_add(0x660)), 937);
    let player = need(word(players), 937);
    let vehicle = need(word(player.wrapping_add(0x1118)), 937);
    if word(controller.wrapping_add(0x1a70)) != Some(sim)
        || dword(arena.wrapping_add(0x668)) != Some(1)
        || word(player) != Some(base+0x1c33938)
        || dword(player.wrapping_add(0x1110)) != Some(0x032e2000)
        || vehicle < 4096 || clock % 10 != 0 || clock > 162200 { fail(938) }
    (sim, player, vehicle, clock)
}

pub unsafe fn handle(info: *const u8, ctx: *mut c_void) -> bool {
    let base = BASE.load(Ordering::SeqCst);
    if base == 0 || ctx.is_null() { return false }
    let g = (ctx as *mut usize).add(5);
    let at = base + RET_RVA;
    if *g.add(16) != at+1 { return false }
    if info.is_null() { fail(939) }
    let code = *(info.add(8) as *const i32);
    if (code != 1 && code != 128) || DEPTH.fetch_add(1, Ordering::SeqCst) != 0
        || *g.add(17) & 0x10100 != 0 { fail(940) }
    let mut actual = [0u8; 20];
    if !read(at, &mut actual) || actual[0] != 0xcc
        || !same(&actual[1..], &IMPACT_RET_SIGNATURE[1..]) { fail(941) }
    let (sim, player, vehicle, clock) = identity(base);
    // Linux gregs: RDI8, RSI9, RDX12, RCX14, R8=0, R9=1, RSP15.
    if *g.add(9) != vehicle { fail(942) }
    let source = *g.add(12);
    let owner = need(word(vehicle+8), 943);
    let count = need(dword(vehicle+0x380), 943);
    if owner < 4096 || count != 4 { fail(944) }
    let return_address = need(word(*g.add(15)), 945);
    let pid = sc(39,0,0,0,0,0,0) as usize;
    let tid = sc(186,0,0,0,0,0,0) as usize;
    let fd = output(pid, tid);
    let sequence = SEQ.fetch_add(1, Ordering::SeqCst)+1;
    if sequence > 100000 { fail(946) }
    let mut data = [0u8; 512];
    copy_volatile(b"TMIH0001", &mut data[..8]);
    w32(&mut data,8,512); w32(&mut data,12,1); w32(&mut data,16,pid as u32);
    w32(&mut data,20,tid as u32); w64(&mut data,24,sequence);
    w32(&mut data,32,0); w32(&mut data,36,clock);
    w32(&mut data,40,*g.add(8) as u32); w32(&mut data,44,*g.add(14) as u32);
    w64(&mut data,48,vehicle); w64(&mut data,56,at);
    w64(&mut data,64,return_address); w64(&mut data,72,source);
    w64(&mut data,80,*g.add(8)); w64(&mut data,88,*g.add(0));
    if !read(source,&mut data[96..144]) || !read(owner,&mut data[152..184])
        || !read(owner+0x1b0,&mut data[184..200]) { fail(947) }
    w64(&mut data,144,owner);
    let wheel_index = *g.add(14) as u32;
    if wheel_index != u32::MAX {
        if wheel_index >= count { fail(948) }
        let wheel = need(word(vehicle+0x388+wheel_index as usize*8), 948);
        w64(&mut data,200,wheel);
        if !read(wheel,&mut data[208..224]) { fail(949) }
    }
    w64(&mut data,224,player); w64(&mut data,232,sim);
    w64(&mut data,240,VALIDATOR_CONTROLLER.load(Ordering::SeqCst));
    w32(&mut data,248,31); w32(&mut data,252,count);
    w64(&mut data,256,base); w64(&mut data,264,sim+0x48);
    if !read(vehicle+0x12e0,&mut data[272..312]) { fail(950) }
    w64(&mut data,312,*g.add(1));
    let mut again = [0u8;48];
    if !read(source,&mut again) || !same(&again,&data[96..144])
        || dword(sim+0x48) != Some(clock) || word(player+0x1118) != Some(vehicle) { fail(951) }
    let mut offset = 0;
    while offset < data.len() {
        let n = sc(1,fd as usize,data[offset..].as_ptr() as usize,data.len()-offset,0,0,0);
        if n == -4 { continue } if n <= 0 { fail(952) } offset += n as usize;
    }
    LAST_CALLBACK.store(clock as usize, Ordering::SeqCst);
    LAST_EVENT.store(*g.add(8) as u32 as usize, Ordering::SeqCst);
    // Original C3: pop the return address without touching flags or FP registers.
    *g.add(16) = return_address;
    *g.add(15) = need((*g.add(15)).checked_add(8), 953);
    DEPTH.store(0, Ordering::SeqCst);
    true
}

pub fn delivery_snapshot() -> (usize, usize, usize) {
    // Factual counters only; not completeness of every engine contact path.
    (SEQ.load(Ordering::SeqCst), LAST_CALLBACK.load(Ordering::SeqCst),
     LAST_EVENT.load(Ordering::SeqCst))
}
