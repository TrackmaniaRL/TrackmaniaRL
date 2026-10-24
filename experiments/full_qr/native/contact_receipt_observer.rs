//! Bind delivered contact counters to actual held reader boundaries.
//! Source ordering evidence, not motor-negative-coverage declarations.
use super::*;
static PID: AtomicUsize=AtomicUsize::new(0);
static TID: AtomicUsize=AtomicUsize::new(0);
static FD: AtomicI32=AtomicI32::new(-1);
static LAST: AtomicUsize=AtomicUsize::new(0);
static SEQ: AtomicUsize=AtomicUsize::new(0);
static CONTACTS: AtomicUsize=AtomicUsize::new(0);

unsafe fn output(pid:usize)->i32 {
    let n=DIR_LEN.load(Ordering::SeqCst);
    if n==0 || n>240 {fail(980)}
    let mut path=[0u8;320];let dir=&*core::ptr::addr_of!(DIR);
    path[..n].copy_from_slice(&dir[..n]);let prefix=b"/contact-receipt-";
    path[n..n+prefix.len()].copy_from_slice(prefix);let start=n+prefix.len();
    let mut digits=[0u8;24];let mut count=0;let mut value=pid;
    loop {digits[count]=b'0'+(value%10) as u8;count+=1;value/=10;if value==0 {break}}
    for i in 0..count {path[start+i]=digits[count-i-1];}
    let end=start+count;path[end..end+4].copy_from_slice(b".bin");
    let fd=sc(257,(-100isize) as usize,path.as_ptr() as usize,0x800c1,0o600,0,0);
    if fd<0 {fail(981)}fd as i32
}

pub unsafe fn at_reader(event:&[u8;256]) {
    let sim=VALIDATOR_SIM.load(Ordering::SeqCst);
    let base=MODULE_BASE.load(Ordering::SeqCst);
    let clock=need(dword(sim+0x48),982) as usize;
    if clock<2190 {return}
    let pid=sc(39,0,0,0,0,0,0) as usize;
    let tid=sc(186,0,0,0,0,0,0) as usize;
    if PID.load(Ordering::SeqCst)!=pid {
        let old=FD.swap(-1,Ordering::SeqCst);if old>=0 {sc(3,old as usize,0,0,0,0,0);}
        PID.store(pid,Ordering::SeqCst);TID.store(tid,Ordering::SeqCst);
        LAST.store(0,Ordering::SeqCst);SEQ.store(0,Ordering::SeqCst);
        CONTACTS.store(0,Ordering::SeqCst);FD.store(output(pid),Ordering::SeqCst);
    }
    let last=LAST.load(Ordering::SeqCst);
    if last==clock {return}
    if clock>162200 || clock%10!=0 || (last!=0 && clock!=last+10)
        // reader::handle owns the one active signal frame here.
        || TID.load(Ordering::SeqCst)!=tid || DEPTH.load(Ordering::SeqCst)!=1
        || u32::from_le_bytes(event[124..128].try_into().unwrap()) as usize!=clock
        || u32::from_le_bytes(event[148..152].try_into().unwrap())&0x1bf!=0x1bf {fail(982)}
    let arena=need(word(sim+0x18),983);let players=need(word(arena+0x660),983);
    let participant=need(word(players),983);let vehicle=need(word(participant+0x1118),983);
    if dword(arena+0x668)!=Some(1) || word(participant)!=Some(base+0x1c33938)
        || dword(participant+0x1110)!=Some(0x032e2000) || vehicle<4096 {fail(983)}
    let (contacts,callback,event_time)=super::prepared_impact_observe::delivery_snapshot();
    if contacts>100000 || contacts<CONTACTS.load(Ordering::SeqCst)
        || callback>clock || (contacts==0 && (callback!=0 || event_time!=0)) {fail(984)}
    let sequence=SEQ.fetch_add(1,Ordering::SeqCst)+1;if sequence>16200 {fail(984)}
    let mut record=[0u8;96];record[..8].copy_from_slice(b"TMCR0001");
    w32(&mut record,8,96);w32(&mut record,12,1);w32(&mut record,16,pid as u32);
    w32(&mut record,20,tid as u32);w64(&mut record,24,sequence);w32(&mut record,32,clock as u32);
    w64(&mut record,40,contacts);w32(&mut record,48,callback as u32);w32(&mut record,52,event_time as u32);
    w64(&mut record,56,participant);w64(&mut record,64,vehicle);w64(&mut record,72,sim);
    w64(&mut record,80,base);w64(&mut record,88,contacts);
    if super::prepared_impact_observe::delivery_snapshot()!=(contacts,callback,event_time)
        || dword(sim+0x48)!=Some(clock as u32) {fail(985)}
    let fd=FD.load(Ordering::SeqCst);let mut offset=0;
    while offset<record.len() {
        let n=sc(1,fd as usize,record[offset..].as_ptr() as usize,record.len()-offset,0,0,0);
        if n == -4 {continue}if n<=0 {fail(986)}offset+=n as usize;
    }
    CONTACTS.store(contacts,Ordering::SeqCst);LAST.store(clock,Ordering::SeqCst);
}
