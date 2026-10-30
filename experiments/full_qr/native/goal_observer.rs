//! Actual classifier-derived writer pairs and their reader ordering.
//! Native Goal facts only; not a declaration of client UI equivalence.
use super::*;
include!("goal_signatures.rs");
static PID:AtomicUsize=AtomicUsize::new(0);
static TID:AtomicUsize=AtomicUsize::new(0);
static FACT_FD:AtomicI32=AtomicI32::new(-1);
static READER_FD:AtomicI32=AtomicI32::new(-1);
static FACT_SEQ:AtomicUsize=AtomicUsize::new(0);
static READER_SEQ:AtomicUsize=AtomicUsize::new(0);
static READER_LAST:AtomicUsize=AtomicUsize::new(0);
static GOALS:AtomicUsize=AtomicUsize::new(0);
static LAST_GOAL:AtomicUsize=AtomicUsize::new(0);
static PENDING:AtomicUsize=AtomicUsize::new(0);
static mut ENTRY:[u8;768]=[0;768];

fn u(data:&[u8],offset:usize)->u32 {u32::from_le_bytes(data[offset..offset+4].try_into().unwrap())}
fn q(data:&[u8],offset:usize)->usize {u64::from_le_bytes(data[offset..offset+8].try_into().unwrap()) as usize}

unsafe fn output(pid:usize,prefix:&[u8])->i32 {
    let n=DIR_LEN.load(Ordering::SeqCst);if n==0 || n>240 {fail(990)}
    let mut path=[0u8;320];let dir=&*core::ptr::addr_of!(DIR);
    path[..n].copy_from_slice(&dir[..n]);path[n..n+prefix.len()].copy_from_slice(prefix);
    let start=n+prefix.len();let mut digits=[0u8;24];let mut count=0;let mut value=pid;
    loop {digits[count]=b'0'+(value%10) as u8;count+=1;value/=10;if value==0 {break}}
    for i in 0..count {path[start+i]=digits[count-i-1];}
    let end=start+count;path[end..end+4].copy_from_slice(b".bin");
    let fd=sc(257,(-100isize) as usize,path.as_ptr() as usize,0x800c1,0o600,0,0);
    if fd<0 {fail(991)}fd as i32
}
unsafe fn owner()->(usize,usize) {
    let pid=sc(39,0,0,0,0,0,0) as usize;let tid=sc(186,0,0,0,0,0,0) as usize;
    if PID.load(Ordering::SeqCst)!=pid {
        for fd in [FACT_FD.swap(-1,Ordering::SeqCst),READER_FD.swap(-1,Ordering::SeqCst)] {
            if fd>=0 {sc(3,fd as usize,0,0,0,0,0);}
        }
        PID.store(pid,Ordering::SeqCst);TID.store(tid,Ordering::SeqCst);
        FACT_SEQ.store(0,Ordering::SeqCst);READER_SEQ.store(0,Ordering::SeqCst);
        READER_LAST.store(0,Ordering::SeqCst);GOALS.store(0,Ordering::SeqCst);
        LAST_GOAL.store(0,Ordering::SeqCst);PENDING.store(0,Ordering::SeqCst);
        FACT_FD.store(output(pid,b"/goal-fact-"),Ordering::SeqCst);
        READER_FD.store(output(pid,b"/goal-receipt-"),Ordering::SeqCst);
    }
    if TID.load(Ordering::SeqCst)!=tid {fail(992)}(pid,tid)
}
unsafe fn write_record(fd:i32,record:&[u8]) {
    let mut offset=0;
    while offset<record.len() {
        let n=sc(1,fd as usize,record[offset..].as_ptr() as usize,record.len()-offset,0,0,0);
        if n == -4 {continue}if n<=0 {fail(993)}offset+=n as usize;
    }
}
pub unsafe fn writer(raw:&[u8;512]) {
    let clock=u(raw,56);if clock<2200 {return}
    let (pid,tid)=owner();let base=MODULE_BASE.load(Ordering::SeqCst);
    let sim=VALIDATOR_SIM.load(Ordering::SeqCst);let controller=VALIDATOR_CONTROLLER.load(Ordering::SeqCst);
    let kind=u(raw,32);
    if kind>1 || clock>162200 || clock%10!=0 || DEPTH.load(Ordering::SeqCst)!=1
        || q(raw,40)!=sim || q(raw,48)!=controller || u(raw,16)!=pid as u32
        || u(raw,20)!=tid as u32 {fail(994)}
    for (rva,signature) in GOAL_SIGNATURES {
        let mut actual=[0u8;20];
        if !read(base+rva,&mut actual) || actual!=signature {fail(995)}
    }
    let sequence=FACT_SEQ.fetch_add(1,Ordering::SeqCst)+1;
    if sequence>10000 || (sequence-1)%2!=kind as usize {fail(996)}
    let entry=&mut *core::ptr::addr_of_mut!(ENTRY);
    if kind==0 {
        if PENDING.load(Ordering::SeqCst)!=0 || u(raw,60)!=1231
            || q(raw,368)!=base+0x11fba27 || q(raw,400)!=base+0x120ac95
            || q(raw,248)!=q(raw,256) || q(raw,64+9*8)!=q(raw,256) {fail(997)}
        let manager=q(raw,64+8*8);let id=q(raw,64);
        let endpoint=q(raw,64+8);let event=q(raw,64+12*8);let dispatch=q(raw,64+14*8);
        // The classifier result3 is converted to R9 by the pinned caller.
        // Event payload+24 is NOT that result (actual377 contains0327d000).
        if manager<4096 || id>65535 || endpoint>1 || !(2200..=162200).contains(&event)
            || dispatch>u32::MAX as usize || u(raw,416)!=0
            || u(raw,416+4)!=dispatch as u32 || u(raw,416+40)!=event as u32 {fail(998)}
        let table_owner=need(word(manager+0x18),999);
        let table=need(word(table_owner+0x68),999);let map=need(word(table_owner+0x18),999);
        let selected=need(table.checked_add(id*0x78),999);
        entry.fill(0);entry[..8].copy_from_slice(b"TMGF0001");
        w32(entry,8,768);w32(entry,12,1);w32(entry,16,pid as u32);w32(entry,20,tid as u32);
        w32(entry,36,clock);w64(entry,40,manager);w64(entry,48,table_owner);
        w64(entry,56,table);w64(entry,64,map);w32(entry,72,id as u32);w32(entry,76,endpoint as u32);
        w32(entry,80,event as u32);w32(entry,84,dispatch as u32);
        w64(entry,88,q(raw,256));w64(entry,96,sim);w64(entry,104,controller);w64(entry,112,selected);
        if !read(selected,&mut entry[120..240]) || (endpoint!=0 && entry[120+0x51]&1==0) {fail(1000)}
        w64(entry,752,base);PENDING.store(1,Ordering::SeqCst);
    } else {
        if PENDING.load(Ordering::SeqCst)!=1 || u(raw,60)!=799 || u(entry,36)!=clock
            || q(raw,248)!=q(entry,88) || u(raw,288)!=u(entry,72)
            || u(raw,292)!=u(entry,80) || u(raw,296)!=u(entry,76) {fail(1001)}
        let mut selected=[0u8;120];
        if !read(q(entry,112),&mut selected) || selected!=entry[120..240] {fail(1002)}
        if u(entry,76)==1 {
            let old=LAST_GOAL.load(Ordering::SeqCst);let actual=u(entry,80) as usize;
            if old!=0 && actual<old {fail(1003)}
            GOALS.fetch_add(1,Ordering::SeqCst);LAST_GOAL.store(actual,Ordering::SeqCst);
        }
        PENDING.store(0,Ordering::SeqCst);
    }
    w64(entry,24,sequence);w32(entry,32,kind);entry[240..752].copy_from_slice(raw);
    write_record(FACT_FD.load(Ordering::SeqCst),entry);
}

pub unsafe fn at_reader(raw:&[u8;256]) {
    let sim=VALIDATOR_SIM.load(Ordering::SeqCst);let clock=need(dword(sim+0x48),1004) as usize;
    if clock<2190 {return}
    let (pid,tid)=owner();let base=MODULE_BASE.load(Ordering::SeqCst);
    let last=READER_LAST.load(Ordering::SeqCst);if last==clock {return}
    if clock>162200 || clock%10!=0 || (last!=0 && last+10!=clock)
        || DEPTH.load(Ordering::SeqCst)!=1 || PENDING.load(Ordering::SeqCst)!=0
        || u(raw,124) as usize!=clock || u(raw,148)&0x1bf!=0x1bf {fail(1005)}
    let participant=q(raw,192);let vehicle=need(word(participant+0x1118),1006);
    if word(participant)!=Some(base+0x1c33938) || dword(participant+0x1110)!=Some(0x032e2000) {fail(1006)}
    let sequence=READER_SEQ.fetch_add(1,Ordering::SeqCst)+1;if sequence>16200 {fail(1007)}
    let facts=FACT_SEQ.load(Ordering::SeqCst);if facts%2!=0 {fail(1007)}
    let mut record=[0u8;96];record[..8].copy_from_slice(b"TMGR0001");
    w32(&mut record,8,96);w32(&mut record,12,1);w32(&mut record,16,pid as u32);
    w32(&mut record,20,tid as u32);w64(&mut record,24,sequence);w32(&mut record,32,clock as u32);
    w64(&mut record,40,facts/2);w32(&mut record,48,LAST_GOAL.load(Ordering::SeqCst) as u32);
    w32(&mut record,52,GOALS.load(Ordering::SeqCst) as u32);
    w64(&mut record,56,participant);w64(&mut record,64,vehicle);w64(&mut record,72,sim);
    w64(&mut record,80,base);w32(&mut record,88,0);
    write_record(READER_FD.load(Ordering::SeqCst),&record);READER_LAST.store(clock,Ordering::SeqCst);
}
