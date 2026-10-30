//! Experimental pinned-ELF observer. Never writes engine objects.
//! The vendor cache producer operates on an observer-owned shadow cache.
//! Requires the separate V8 long150 build and its source/asset receipts.
use super::*;

static VEHICLE_PID: AtomicUsize = AtomicUsize::new(0);
static VEHICLE_TID: AtomicUsize = AtomicUsize::new(0);
static VEHICLE_FD: AtomicI32 = AtomicI32::new(-1);
static VEHICLE_LAST: AtomicUsize = AtomicUsize::new(0);
static VEHICLE_SEQUENCE: AtomicUsize = AtomicUsize::new(0);
static mut SHADOW: [u8; 140] = [0; 140];
const PRODUCER: usize = 0x9d6ad0;
const PRODUCER_PREFIX: [u8; 20] = [
    0x55,0x48,0x89,0xe5,0x41,0x57,0x41,0x56,0x53,0x48,
    0x83,0xec,0x28,0xf3,0x0f,0x11,0x45,0xdc,0x49,0x89,
];

unsafe fn output(pid: usize, fixture: bool) -> i32 {
    let mut path = [0u8; 320];
    let n = DIR_LEN.load(Ordering::SeqCst);
    if n == 0 || n > 240 { fail(701) }
    let dir = &*core::ptr::addr_of!(DIR);
    path[..n].copy_from_slice(&dir[..n]);
    let prefix: &[u8] = if fixture { b"/vehicle-fixture-" } else { b"/vehicle-" };
    let start = n + prefix.len();
    path[n..start].copy_from_slice(prefix);
    let mut digits = [0u8; 24];
    let mut value = pid;
    let mut count = 0;
    loop {
        digits[count] = b'0' + (value % 10) as u8;
        count += 1;
        value /= 10;
        if value == 0 { break }
    }
    for i in 0..count { path[start+i] = digits[count-i-1]; }
    let end = start+count;
    path[end..end+4].copy_from_slice(b".bin");
    // openat, fresh regular file, close-on-exec. No environment/allocation here.
    let fd = sc(257,(-100isize) as usize,path.as_ptr() as usize,0x800c1,0o600,0,0);
    if fd < 0 { fail(702) }
    fd as i32
}

type CacheProducer = unsafe extern "C" fn(*mut u8,*const u8,f32);

unsafe fn synthetic_flying_fixture(pid: usize, visual: &[u8;864], producer: CacheProducer) {
    // Deliberately changed COPY, never a trajectory sample or engine-object write.
    let mut input = *visual;
    let mut cache = [0u8;140];
    let original = u32::from_le_bytes(input[0x88..0x8c].try_into().unwrap());
    let forced = (original | 0x10000) & !8u32;
    w32(&mut input,0x88,forced);w32(&mut input,0x1bc,0);
    producer(cache.as_mut_ptr(),input.as_ptr(),10.0f32 / 1000.0);
    let first = u32::from_le_bytes(cache[0x5c..0x60].try_into().unwrap());
    producer(cache.as_mut_ptr(),input.as_ptr(),10.0f32 / 1000.0);
    let second = u32::from_le_bytes(cache[0x5c..0x60].try_into().unwrap());
    w32(&mut input,0x88,forced | 8);
    producer(cache.as_mut_ptr(),input.as_ptr(),10.0f32 / 1000.0);
    let reset = u32::from_le_bytes(cache[0x5c..0x60].try_into().unwrap());
    if (first,second,reset) != (10,20,0) { fail(718) }
    let mut record = [0u8;64];record[..8].copy_from_slice(b"TMVF0001");
    w32(&mut record,8,64);w32(&mut record,12,1);w32(&mut record,16,pid as u32);
    w32(&mut record,20,1); // explicit synthetic-fixture marker
    w32(&mut record,24,first);w32(&mut record,28,second);w32(&mut record,32,reset);
    w32(&mut record,36,original);w32(&mut record,40,forced);w32(&mut record,44,PRODUCER as u32);
    let fd = output(pid,true);let mut offset = 0;
    while offset < record.len() {
        let n = sc(1,fd as usize,record[offset..].as_ptr() as usize,record.len()-offset,0,0,0);
        if n == -4 { continue }if n <= 0 { fail(719) }offset += n as usize;
    }
    sc(3,fd as usize,0,0,0,0,0);
}

pub unsafe fn at_reader(event: &[u8;256]) {
    let sim = VALIDATOR_SIM.load(Ordering::SeqCst);
    let module = MODULE_BASE.load(Ordering::SeqCst);
    let clock = need(dword(sim.wrapping_add(0x48)),703) as usize;
    if clock < 2190 { return }
    if clock > 162200 || clock % 10 != 0 { fail(704) }
    let pid = sc(39,0,0,0,0,0,0) as usize;
    let tid = sc(186,0,0,0,0,0,0) as usize;
    if u32::from_le_bytes(event[148..152].try_into().unwrap()) & 0x1bf != 0x1bf
        || u32::from_le_bytes(event[124..128].try_into().unwrap()) as usize != clock
    { fail(705) }
    if VEHICLE_PID.load(Ordering::SeqCst) != pid {
        let old = VEHICLE_FD.swap(-1,Ordering::SeqCst);
        if old >= 0 { sc(3,old as usize,0,0,0,0,0); }
        VEHICLE_PID.store(pid,Ordering::SeqCst);
        VEHICLE_TID.store(tid,Ordering::SeqCst);
        VEHICLE_LAST.store(0,Ordering::SeqCst);
        VEHICLE_SEQUENCE.store(0,Ordering::SeqCst);
        let cache = &mut *core::ptr::addr_of_mut!(SHADOW);
        cache.fill(0);
        cache[0x14..0x18].copy_from_slice(&1.0f32.to_le_bytes());
        VEHICLE_FD.store(output(pid,false),Ordering::SeqCst);
    }
    if VEHICLE_TID.load(Ordering::SeqCst) != tid { fail(706) }
    let last = VEHICLE_LAST.load(Ordering::SeqCst);
    if last == clock { return }
    if last != 0 && last.checked_add(10) != Some(clock) { fail(707) }
    let controller = VALIDATOR_CONTROLLER.load(Ordering::SeqCst);
    let pg = need(word(sim.wrapping_add(0x18)),708);
    let players = need(word(pg.wrapping_add(0x660)),708);
    let participant = need(word(players),708);
    let vehicle = need(word(participant.wrapping_add(0x1118)),708);
    if word(controller.wrapping_add(0x1a70)) != Some(sim)
        || dword(pg.wrapping_add(0x668)) != Some(1)
        || word(participant) != Some(module+0x1c33938)
        || dword(participant.wrapping_add(0x1110)) != Some(0x032e2000)
        || vehicle < 4096
    { fail(709) }
    let descriptor = need(word(vehicle),716);
    if dword(descriptor.wrapping_add(0x3c)) != Some(4) { fail(717) }
    let mut signature = [0u8;20];
    if !read(module+PRODUCER,&mut signature) || signature != PRODUCER_PREFIX { fail(710) }
    let mut visual = [0u8;864];
    let mut again = [0u8;864];
    let mut live_cache = [0u8;140];
    let global = need(word(module+0x1d98a28),711);
    if !read(vehicle+0x848,&mut visual)
        || !read(participant+0xec0,&mut live_cache)
        || !read(vehicle+0x848,&mut again)
        || !same(&visual,&again)
        || dword(sim+0x48) != Some(clock as u32)
        || u32::from_le_bytes(visual[12..16].try_into().unwrap()) as usize != clock
    { fail(712) }
    // Read-only source copy, observer-owned destination, fixed complete entrypoint.
    // The full 1542-byte function writes only its 140-byte destination and calls atan2f.
    let producer: CacheProducer = core::mem::transmute(module+PRODUCER);
    let cache = &mut *core::ptr::addr_of_mut!(SHADOW);
    let delta = if last == 0 { 0 } else { clock-last };
    producer(cache.as_mut_ptr(),visual.as_ptr(),delta as f32 / 1000.0);
    if last == 0 { synthetic_flying_fixture(pid,&visual,producer); }
    // Calling the vendor math must leave the actual source and engine state intact.
    if !read(vehicle+0x848,&mut again) || !same(&visual,&again)
        || dword(sim+0x48) != Some(clock as u32)
        || word(participant+0x1118) != Some(vehicle)
        || word(module+0x1d98a28) != Some(global)
    { fail(713) }
    let sequence = VEHICLE_SEQUENCE.fetch_add(1,Ordering::SeqCst)+1;
    if sequence > 16200 { fail(714) }
    let mut record = [0u8;1280];
    record[..8].copy_from_slice(b"TMVH0001");
    w32(&mut record,8,1280);w32(&mut record,12,1);
    w32(&mut record,16,pid as u32);w32(&mut record,20,tid as u32);
    w64(&mut record,24,sequence);w32(&mut record,32,clock as u32);
    w32(&mut record,36,delta as u32);w64(&mut record,40,participant);
    w64(&mut record,48,vehicle);w64(&mut record,56,global);
    record[64..204].copy_from_slice(cache);
    w32(&mut record,204,1);record[208..1072].copy_from_slice(&visual);
    record[1072..1212].copy_from_slice(&live_cache);
    w32(&mut record,1240,PRODUCER as u32);
    w32(&mut record,1248,clock as u32);w32(&mut record,1252,clock as u32);
    w32(&mut record,1256,clock as u32);w32(&mut record,1260,clock as u32);
    w64(&mut record,1264,vehicle+0x848);w64(&mut record,1272,module+PRODUCER);
    let fd = VEHICLE_FD.load(Ordering::SeqCst);
    let mut offset = 0;
    while offset < record.len() {
        let n = sc(1,fd as usize,record[offset..].as_ptr() as usize,record.len()-offset,0,0,0);
        if n == -4 { continue }
        if n <= 0 { fail(715) }
        offset += n as usize;
    }
    VEHICLE_LAST.store(clock,Ordering::SeqCst);
}
