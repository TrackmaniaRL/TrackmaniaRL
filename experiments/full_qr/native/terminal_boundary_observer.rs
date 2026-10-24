//! Actual result-collector boundary, separate from a decoded-input reader.
//! No engine-state writes and no manufactured reader event or clock.
use super::*;

pub(super) enum ClockWitness<'a> {
    Reader(&'a [u8;256]),
    Collector(&'a [u8;768]),
}

fn u(raw:&[u8],at:usize)->u32 {u32::from_le_bytes(raw[at..at+4].try_into().unwrap())}
fn q(raw:&[u8],at:usize)->usize {u64::from_le_bytes(raw[at..at+8].try_into().unwrap())as usize}

impl ClockWitness<'_> {
    pub(super) fn participant(&self)->usize {
        match self {
            Self::Reader(raw)=>q(*raw,192),
            Self::Collector(raw)=>q(*raw,80),
        }
    }
    pub(super) unsafe fn valid(&self,clock:usize)->bool {
        let pid=sc(39,0,0,0,0,0,0)as u32;
        let tid=sc(186,0,0,0,0,0,0)as u32;
        let sim=VALIDATOR_SIM.load(Ordering::SeqCst);
        let controller=VALIDATOR_CONTROLLER.load(Ordering::SeqCst);
        if DEPTH.load(Ordering::SeqCst)!=1 || dword(sim+0x48)!=Some(clock as u32)
            || word(controller+0x1a70)!=Some(sim) {return false}
        match self {
            Self::Reader(raw)=>same(&raw[..8],b"TMREAD01")
                && u(*raw,8)==256 && u(*raw,12)==1
                && u(*raw,16)==pid && u(*raw,20)==tid
                && q(*raw,40)==controller && q(*raw,48)==sim
                && u(*raw,124)as usize==clock && u(*raw,148)&0x1bf==0x1bf
                && u(*raw,208)==1,
            Self::Collector(raw)=>same(&raw[..8],b"TMRESULT")
                && u(*raw,8)==768 && u(*raw,12)==1
                && u(*raw,16)==pid && u(*raw,20)==tid
                && q(*raw,40)==controller && q(*raw,48)==sim
                && u(*raw,128)as usize==clock && u(*raw,148)&0x47ff==0x47ff
                && u(*raw,240)==13 && u(*raw,248)==1
                && q(*raw,232)==MODULE_BASE.load(Ordering::SeqCst)+0x11834f3,
        }
    }
}

static PID:AtomicUsize=AtomicUsize::new(0);
static FD:AtomicI32=AtomicI32::new(-1);
static SEQ:AtomicUsize=AtomicUsize::new(0);
static MOTOR_FD:AtomicI32=AtomicI32::new(-1);

unsafe fn named_output(pid:usize,prefix:&[u8])->i32 {
    let n=DIR_LEN.load(Ordering::SeqCst);
    if n==0 || n>240 {fail(920)}
    let mut path=[0u8;320];let dir=&*core::ptr::addr_of!(DIR);
    path[..n].copy_from_slice(&dir[..n]);path[n..n+prefix.len()].copy_from_slice(prefix);
    let start=n+prefix.len();let mut digits=[0u8;24];let mut count=0;let mut value=pid;
    loop {digits[count]=b'0'+(value%10)as u8;count+=1;value/=10;if value==0 {break}}
    for i in 0..count {path[start+i]=digits[count-i-1];}
    let end=start+count;path[end..end+4].copy_from_slice(b".bin");
    let fd=sc(257,(-100isize)as usize,path.as_ptr()as usize,0x800c1,0o600,0,0);
    if fd<0 {fail(921)}fd as i32
}

unsafe fn output(pid:usize)->i32 {
    if PID.load(Ordering::SeqCst)!=pid {
        let old=FD.swap(-1,Ordering::SeqCst);
        if old>=0 {sc(3,old as usize,0,0,0,0,0);}
        let old=MOTOR_FD.swap(-1,Ordering::SeqCst);
        if old>=0 {sc(3,old as usize,0,0,0,0,0);}
        let fd=named_output(pid,b"/final-boundary-");
        MOTOR_FD.store(named_output(pid,b"/final-material-"),Ordering::SeqCst);
        PID.store(pid,Ordering::SeqCst);FD.store(fd,Ordering::SeqCst);
        SEQ.store(0,Ordering::SeqCst);
    }
    FD.load(Ordering::SeqCst)
}

unsafe fn class_is(pointer:usize,expected:u32)->bool {
    if pointer<4096 {return false}
    let Some(vtable)=word(pointer) else {return false};
    let Some(getter)=word(vtable+32) else {return false};
    let mut code=[0u8;11];if !read(getter,&mut code) {return false}
    same(&code[..5],b"\x55\x48\x89\xe5\xb8") && u(&code,5)==expected
        && same(&code[9..],b"\x5d\xc3")
}

unsafe fn palette(pointer:usize,raw:&mut[u8])->bool {
    let mut again=[0u8;144];
    class_is(pointer,0x090f1000) && read(pointer,raw) && read(pointer,&mut again)
        && same(raw,&again)
}

unsafe fn material_source(clock:usize,pid:usize,sequence:usize,participant:usize,vehicle:usize) {
    let descriptor=need(word(vehicle),929);let owner=need(word(vehicle+8),929);
    if !class_is(descriptor,0x090ea000) || dword(vehicle+0x380)!=Some(4) {fail(930)}
    let mut record=[0u8;1536];record[..8].copy_from_slice(b"TMFM0001");
    w32(&mut record,8,1536);w32(&mut record,12,1);w32(&mut record,16,pid as u32);
    w32(&mut record,20,sc(186,0,0,0,0,0,0)as u32);w64(&mut record,24,sequence);
    w32(&mut record,32,clock as u32);w32(&mut record,36,0);
    for (at,value) in [(40,VALIDATOR_SIM.load(Ordering::SeqCst)),
        (48,VALIDATOR_CONTROLLER.load(Ordering::SeqCst)),(56,participant),(64,vehicle),
        (72,descriptor),(80,owner),(88,MODULE_BASE.load(Ordering::SeqCst))] {
        w64(&mut record,at,value);
    }
    let enabled=need(dword(vehicle+0x1388),931);
    w32(&mut record,96,enabled);w32(&mut record,100,4);w32(&mut record,108,0x11834f2);
    if !read(owner,&mut record[128..160]) || !read(owner+0x1b0,&mut record[160..176])
        || !read(vehicle+0x848,&mut record[640..1504]) {fail(932)}
    let mut contacts=[0usize;2];let mut selected=[80u32;2];let mut fallback=[0u32;2];
    for wheel in 0..4 {
        let pointer=need(word(vehicle+0x388+wheel*8),933);
        w64(&mut record,176+wheel*8,pointer);
        if !read(pointer,&mut record[208+wheel*32..240+wheel*32]) {fail(934)}
        let longitudinal=f32::from_le_bytes(record[232+wheel*32..236+wheel*32].try_into().unwrap());
        if !longitudinal.is_finite() {fail(935)}
        let group=if longitudinal>0.0 {1}else{0};
        let material=record[640+0xb8+wheel*0x2c]as u32;
        let flags=u(&record,640+0xd0+wheel*0x2c);
        if material>80 {fail(936)}
        if flags&4!=0 {contacts[group]+=1;if selected[group]!=13 {selected[group]=material}}
        else {fallback[group]=material}
        if flags&2!=0 {selected[group]=13}
    }
    for group in 0..2 {if selected[group]>=80 {selected[group]=fallback[group]}}
    let material=selected[if contacts[0]>contacts[1] {0}else{1}];w32(&mut record,104,material);
    if material!=80 {
        let pointer=need(word(descriptor+0x6b8+material as usize*8),937);
        w64(&mut record,336,pointer);
        if !palette(pointer,&mut record[344..488])
            || word(descriptor+0x6b8+material as usize*8)!=Some(pointer) {fail(938)}
    }
    if material==14 {
        let pointer=need(word(descriptor+0x6b8),939);w64(&mut record,488,pointer);
        if !palette(pointer,&mut record[496..640]) || word(descriptor+0x6b8)!=Some(pointer) {fail(940)}
    }
    let mut visual=[0u8;864];let mut header=[0u8;32];let mut thresholds=[0u8;16];
    if !read(vehicle+0x848,&mut visual) || !same(&visual,&record[640..1504])
        || !read(owner,&mut header) || !same(&header,&record[128..160])
        || !read(owner+0x1b0,&mut thresholds) || !same(&thresholds,&record[160..176])
        || word(vehicle)!=Some(descriptor) || word(vehicle+8)!=Some(owner)
        || dword(vehicle+0x1388)!=Some(enabled) {fail(941)}
    for wheel in 0..4 {
        let pointer=q(&record,176+wheel*8);let mut model=[0u8;32];
        if word(vehicle+0x388+wheel*8)!=Some(pointer) || !read(pointer,&mut model)
            || !same(&model,&record[208+wheel*32..240+wheel*32]) {fail(942)}
    }
    let fd=MOTOR_FD.load(Ordering::SeqCst);let mut offset=0;
    while offset<record.len() {
        let n=sc(1,fd as usize,record[offset..].as_ptr()as usize,record.len()-offset,0,0,0);
        if n == -4 {continue}if n<=0 {fail(943)}offset+=n as usize;
    }
}

pub(super) unsafe fn at_collector(raw:&[u8;768]) {
    let (pairs,goals,event,pending)=super::goal_observe::delivery_snapshot();
    if goals==0 {return}
    let sim=VALIDATOR_SIM.load(Ordering::SeqCst);
    let clock=need(dword(sim+0x48),922)as usize;
    let previous=super::vehicle_observe::latest_clock();
    if previous==clock {return}
    if previous<2200 || clock!=previous+10 || clock>152200 || pending!=0
        || pairs<goals || goals!=1 || !ClockWitness::Collector(raw).valid(clock) {fail(923)}
    let controller=VALIDATOR_CONTROLLER.load(Ordering::SeqCst);
    let base=MODULE_BASE.load(Ordering::SeqCst);
    let arena=need(word(sim+0x18),924);
    let players=need(word(arena+0x660),924);
    let participant=need(word(players),924);
    let vehicle=need(word(participant+0x1118),924);
    if q(raw,80)!=participant || dword(arena+0x668)!=Some(1)
        || word(participant)!=Some(base+0x1c33938) {fail(925)}
    let pid=sc(39,0,0,0,0,0,0)as usize;let fd=output(pid);
    let next_sequence=SEQ.load(Ordering::SeqCst)+1;
    // These are independently checked collector entry points. They never
    // turn this result record into a fabricated TMREAD01 event.
    super::vehicle_observe::at_collector(raw);
    super::adherence_observe::at_collector(raw);
    super::contact_receipt_observe::at_collector(raw);
    super::goal_observe::at_collector(raw);
    material_source(clock,pid,next_sequence,participant,vehicle);
    if dword(sim+0x48)!=Some(clock as u32)
        || super::vehicle_observe::latest_clock()!=clock
        || super::goal_observe::delivery_snapshot()!=(pairs,goals,event,0)
        || word(participant+0x1118)!=Some(vehicle) {fail(926)}
    let seq=SEQ.fetch_add(1,Ordering::SeqCst)+1;if seq>5000 {fail(927)}
    if seq!=next_sequence {fail(944)}
    let mut record=[0u8;896];record[..8].copy_from_slice(b"TMFN0001");
    w32(&mut record,8,896);w32(&mut record,12,1);
    w32(&mut record,16,pid as u32);w32(&mut record,20,sc(186,0,0,0,0,0,0)as u32);
    w64(&mut record,24,seq);w32(&mut record,32,clock as u32);
    w32(&mut record,36,previous as u32);
    for (at,value) in [(40,sim),(48,controller),(56,participant),(64,vehicle),
        (80,q(raw,24)),(88,base),(96,base+0x11834f2)] {w64(&mut record,at,value)}
    w32(&mut record,72,goals as u32);w32(&mut record,76,event as u32);
    w32(&mut record,104,u(raw,148));w32(&mut record,108,1);
    record[128..].copy_from_slice(raw);
    let mut offset=0;
    while offset<record.len() {
        let n=sc(1,fd as usize,record[offset..].as_ptr()as usize,record.len()-offset,0,0,0);
        if n == -4 {continue}if n<=0 {fail(928)}offset+=n as usize;
    }
}
