//! Separate measured arena-resolved coefficient stream. No engine writes/defaults.
use super::*;
include!("adherence_signature.rs");
static PID: AtomicUsize = AtomicUsize::new(0);
static TID: AtomicUsize = AtomicUsize::new(0);
static FD: AtomicI32 = AtomicI32::new(-1);
static LAST: AtomicUsize = AtomicUsize::new(0);
static SEQ: AtomicUsize = AtomicUsize::new(0);

unsafe fn output(pid: usize) -> i32 {
    let n = DIR_LEN.load(Ordering::SeqCst);
    if n == 0 || n > 240 { fail(960) }
    let mut path = [0u8;320];
    let dir = &*core::ptr::addr_of!(DIR);
    path[..n].copy_from_slice(&dir[..n]);
    let prefix = b"/adherence-";path[n..n+prefix.len()].copy_from_slice(prefix);
    let start=n+prefix.len();let mut digits=[0u8;24];let mut value=pid;let mut count=0;
    loop { digits[count]=b'0'+(value%10) as u8;count+=1;value/=10;if value==0 {break} }
    for i in 0..count {path[start+i]=digits[count-i-1];}
    let end=start+count;path[end..end+4].copy_from_slice(b".bin");
    let fd=sc(257,(-100isize) as usize,path.as_ptr() as usize,0x800c1,0o600,0,0);
    if fd<0 {fail(961)}fd as i32
}

pub unsafe fn at_reader(event: &[u8;256]) {
    let base=MODULE_BASE.load(Ordering::SeqCst);
    let sim=VALIDATOR_SIM.load(Ordering::SeqCst);
    let controller=VALIDATOR_CONTROLLER.load(Ordering::SeqCst);
    let clock=need(dword(sim+0x48),962) as usize;
    if clock<2190 {return}
    if clock>162200 || clock%10!=0 {fail(962)}
    let pid=sc(39,0,0,0,0,0,0) as usize;
    let tid=sc(186,0,0,0,0,0,0) as usize;
    if PID.load(Ordering::SeqCst)!=pid {
        let old=FD.swap(-1,Ordering::SeqCst);if old>=0 {sc(3,old as usize,0,0,0,0,0);}
        PID.store(pid,Ordering::SeqCst);TID.store(tid,Ordering::SeqCst);
        LAST.store(0,Ordering::SeqCst);SEQ.store(0,Ordering::SeqCst);
        FD.store(output(pid),Ordering::SeqCst);
    }
    let last=LAST.load(Ordering::SeqCst);
    if last==clock {return}
    if TID.load(Ordering::SeqCst)!=tid || (last!=0 && last+10!=clock)
        || u32::from_le_bytes(event[124..128].try_into().unwrap()) as usize!=clock
        || u32::from_le_bytes(event[148..152].try_into().unwrap())&0x1bf!=0x1bf
        || word(controller+0x1a70)!=Some(sim) {fail(963)}
    for (rva,signature) in ADHERENCE_SIGNATURES {
        let mut actual=[0u8;20];
        if !read(base+rva,&mut actual) || actual!=signature {fail(964)}
    }
    let arena=need(word(sim+0x18),965);
    let players=need(word(arena+0x660),965);
    let participant=need(word(players),965);
    let vehicle=need(word(participant+0x1118),965);
    if dword(arena+0x668)!=Some(1) || word(participant)!=Some(base+0x1c33938)
        || word(base+0x1c33938+0x110)!=Some(base+0x121c7a0)
        || dword(participant+0x1110)!=Some(0x032e2000) || vehicle<4096 {fail(966)}
    let context=need(word(participant+0xc60),967);
    let first=need(dword(participant+0x28c),967);
    let second=need(dword(participant+0x280),967);
    let entity=if first!=0x0ff00000 {first} else if second!=0x0ff00000 {second}
        else if context==0 {need(dword(participant+0x278),967)}
        else {
            let mut kind=[0u8;1];if !read(context+0xc,&mut kind) {fail(967)}
            let mut table=[0u8;4];if !read(base+0x3c19ac,&mut table) {fail(967)}
            let slot=if kind[0]<4 {table[kind[0] as usize] as usize} else {6};
            if slot>6 {fail(967)}
            need(dword(participant+if slot==6 {0x278} else {0x294+4*slot}),967)
        };
    if entity&0xff000000!=0x02000000 {fail(968)}
    let pool=need(word(arena+0x7c8),968);
    let mut header=[0u8;56];if !read(pool+0x8e0,&mut header) {fail(969)}
    let capacity=u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
    let count=u32::from_le_bytes(header[8..12].try_into().unwrap()) as usize;
    let flags=u64::from_le_bytes(header[32..40].try_into().unwrap()) as usize;
    let keys=u64::from_le_bytes(header[40..48].try_into().unwrap()) as usize;
    let values=u64::from_le_bytes(header[48..56].try_into().unwrap()) as usize;
    if capacity==0 || capacity>65536 || !capacity.is_power_of_two() || count==0 || count>capacity
        || flags<4096 || keys<4096 || values<4096 {fail(969)}
    let mut slot=(entity as usize)&(capacity-1);let origin=slot;let mut increment=1;
    let mut found=false;
    for _ in 0..capacity {
        let flag=(need(dword(flags+(slot>>4)*4),970)>>((slot&15)*2))&3;
        if flag&2!=0 {break}
        if flag==0 && dword(keys+slot*4)==Some(entity) {found=true;break}
        slot=(slot+increment)&(capacity-1);increment+=1;if slot==origin {break}
    }
    if !found {fail(970)}
    let mut reference=[0u8;16];
    if !read(values+slot*16,&mut reference) {fail(971)}
    let tag=u32::from_le_bytes(reference[..4].try_into().unwrap());
    let generation=u32::from_le_bytes(reference[4..8].try_into().unwrap());
    let physical=u64::from_le_bytes(reference[8..].try_into().unwrap()) as usize;
    if tag!=0x0a020000 || generation==0xffffffff || physical!=vehicle {fail(971)}
    let coefficient=need(dword(physical+0x1760),972);
    if !f32::from_bits(coefficient).is_finite() {fail(972)}
    let global=need(word(base+0x1d98a28),972);
    if global!=0 && global!=arena {fail(972)}
    let mut header_again=[0u8;56];let mut ref_again=[0u8;16];
    if !read(pool+0x8e0,&mut header_again) || header!=header_again
        || !read(values+slot*16,&mut ref_again) || reference!=ref_again
        || dword(physical+0x1760)!=Some(coefficient) || word(sim+0x18)!=Some(arena)
        || word(participant+0x1118)!=Some(vehicle) || dword(sim+0x48)!=Some(clock as u32)
        || word(participant+0xc60)!=Some(context) {fail(973)}
    let sequence=SEQ.fetch_add(1,Ordering::SeqCst)+1;
    if sequence>16200 {fail(974)}
    let mut record=[0u8;128];record[..8].copy_from_slice(b"TMAD0001");
    w32(&mut record,8,128);w32(&mut record,12,1);w32(&mut record,16,pid as u32);w32(&mut record,20,tid as u32);
    w64(&mut record,24,sequence);w32(&mut record,32,clock as u32);w32(&mut record,36,if last==0 {0} else {10});
    w64(&mut record,40,participant);w64(&mut record,48,vehicle);w64(&mut record,56,arena);w64(&mut record,64,pool);
    w32(&mut record,72,entity);w32(&mut record,76,slot as u32);record[80..96].copy_from_slice(&reference);
    w32(&mut record,96,coefficient);w32(&mut record,100,0x121c7a0);w32(&mut record,104,0x1195d70);
    w32(&mut record,108,0xa48730);w64(&mut record,112,global);w32(&mut record,120,clock as u32);
    let fd=FD.load(Ordering::SeqCst);let mut offset=0;
    while offset<record.len() {
        let n=sc(1,fd as usize,record[offset..].as_ptr() as usize,record.len()-offset,0,0,0);
        if n == -4 {continue}if n<=0 {fail(975)}offset+=n as usize;
    }
    LAST.store(clock,Ordering::SeqCst);
}
