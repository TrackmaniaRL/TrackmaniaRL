//! Experimental reachability/data provenance only, never a finish signal.
//! Three pinned instruction sites in the native waypoint writer. Read-only.
use super::*;

// Actual 20-byte signatures are supplied by the isolated build's pinned ELF witness.
include!("waypoint_signatures.rs");
static ENABLED: AtomicUsize = AtomicUsize::new(0);
static PID: AtomicUsize = AtomicUsize::new(0);
static TID: AtomicUsize = AtomicUsize::new(0);
static FD: AtomicI32 = AtomicI32::new(-1);
static SEQ: AtomicUsize = AtomicUsize::new(0);

pub unsafe fn configure(base: usize) {
    if DIR_LEN.load(Ordering::SeqCst) == 0 { fail(801) }
    for (off, sig) in SITES {
        let mut bytes = [0;20];
        if !read(base+off,&mut bytes) || !same(&bytes,&sig) { fail(802) }
    }
    ENABLED.store(base,Ordering::SeqCst);
}
pub unsafe fn activate() {
    let base = ENABLED.load(Ordering::SeqCst);
    if base == 0 { return }
    for (off,sig) in SITES {
        let at=base+off;let mut bytes=[0;20];
        if !read(at,&mut bytes) || !same(&bytes,&sig) { fail(803) }
        let page=at&!4095usize;
        if sc(10,page,4096,7,0,0,0)!=0 { fail(804) }
        core::ptr::write_volatile(at as *mut u8,0xcc);
        if sc(10,page,4096,5,0,0,0)!=0 { fail(805) }
    }
}
unsafe fn output(pid:usize,tid:usize)->i32 {
    if PID.load(Ordering::SeqCst)!=pid {
        let old=FD.swap(-1,Ordering::SeqCst);
        if old>=0 { sc(3,old as usize,0,0,0,0,0); }
        let mut path=[0u8;320];let n=DIR_LEN.load(Ordering::SeqCst);
        if n==0 || n>240 { fail(806) }
        copy_volatile(&(&*core::ptr::addr_of!(DIR))[..n],&mut path[..n]);
        copy_volatile(b"/waypoint-",&mut path[n..n+10]);
        let mut digits=[0u8;24];let mut x=pid;let mut count=0;
        loop { digits[count]=b'0'+(x%10)as u8;count+=1;x/=10;if x==0 { break } }
        for i in 0..count { path[n+10+i]=digits[count-1-i]; }
        copy_volatile(b".bin",&mut path[n+10+count..n+14+count]);
        let fd=sc(257,(-100isize)as usize,path.as_ptr()as usize,0x800c1,0o600,0,0);
        if fd<0 { fail(807) }
        FD.store(fd as i32,Ordering::SeqCst);PID.store(pid,Ordering::SeqCst);
        TID.store(tid,Ordering::SeqCst);SEQ.store(0,Ordering::SeqCst);
    }
    if TID.load(Ordering::SeqCst)!=tid { fail(808) }
    FD.load(Ordering::SeqCst)
}
pub unsafe fn handle(info:*const u8,ctx:*mut c_void)->bool {
    let base=ENABLED.load(Ordering::SeqCst);
    if base==0 || ctx.is_null() { return false }
    let g=(ctx as *mut usize).add(5);let rip=*g.add(16);
    let Some((kind,(off,sig)))=SITES.iter().copied().enumerate().find(|(_, (off,_))|rip==base+off+1) else { return false };
    let at=base+off;
    if info.is_null() { fail(809) }
    let code=*(info.add(8)as *const i32);
    if code!=1 && code!=128 { fail(810) }
    if DEPTH.fetch_add(1,Ordering::SeqCst)!=0 || *g.add(17)&0x10100!=0 { fail(811) }
    let mut bytes=[0;20];
    if !read(at,&mut bytes) || bytes[0]!=0xcc || !same(&bytes[1..],&sig[1..]) { fail(812) }
    // Exactly one source read also supplies the emulated instruction result.
    let loaded=match kind {
        0=>*g.add(15), // 48 89 e5 MOV RBP,RSP
        1=>need(dword((*g.add(5)).wrapping_add(0x1c8)),813)as usize, // MOV EBX,[R13+1c8]
        2=>need(dword((*g.add(5)).wrapping_add(0x1dc)),814)as usize, // MOV EAX,[R13+1dc]
        _=>fail(815),
    };
    let pid=sc(39,0,0,0,0,0,0)as usize;let tid=sc(186,0,0,0,0,0,0)as usize;
    let fd=output(pid,tid);let seq=SEQ.fetch_add(1,Ordering::SeqCst)+1;
    if seq>10000 { fail(816) }
    let mut data=[0u8;512];copy_volatile(b"TMWP0001",&mut data[..8]);
    w32(&mut data,8,512);w32(&mut data,12,1);w32(&mut data,16,pid as u32);w32(&mut data,20,tid as u32);
    w64(&mut data,24,seq);w32(&mut data,32,kind as u32);w32(&mut data,36,off as u32);
    let sim=VALIDATOR_SIM.load(Ordering::SeqCst);
    w64(&mut data,40,sim);w64(&mut data,48,VALIDATOR_CONTROLLER.load(Ordering::SeqCst));
    let mut mask=0u32;
    if let Some(clock)=dword(sim.wrapping_add(0x48)) { w32(&mut data,56,clock);mask|=1; }
    for index in 0..23 { w64(&mut data,64+index*8,*g.add(index)); }
    let context=if kind==0 { *g.add(9) }else{ *g.add(5) };
    w64(&mut data,248,context);
    if let Some(pg)=word(sim.wrapping_add(0x18)) {
        if dword(pg.wrapping_add(0x668))==Some(1) {
            if let Some(players)=word(pg.wrapping_add(0x660)) {
                if let Some(player)=word(players) {
                    w64(&mut data,256,player);
                    if player==context && word(player)==Some(base+0x1c33938) { mask|=2; }
                }
            }
        }
    }
    if read(context.wrapping_add(0x368),&mut data[264..272]) { mask|=4; }
    if read(context.wrapping_add(0x218),&mut data[272..280]) { mask|=8; }
    if kind!=0 {
        let bp=*g.add(10);
        if bp>=0x1000 && read(bp-0x50,&mut data[280..320]) { mask|=16; }
        if kind==2 && read((*g.add(6)).wrapping_add(0x28),&mut data[320..352]) { mask|=32; }
    }
    // Preserve call provenance while the stack belongs to this stopped thread.
    if kind==0 {
        if read(*g.add(15),&mut data[360..392]) { mask|=64; }
        if read(*g.add(10),&mut data[392..408]) { mask|=128; }
        // Only the statically witnessed event-dispatch -> classifier -> writer path.
        // Other callers (including the scripting API) never acquire this mask.
        if word((*g.add(15)).wrapping_add(8))==Some(base+0x11fba27)
            && word((*g.add(10)).wrapping_add(8))==Some(base+0x120ac95) {
            if let Some(parent_bp)=word(*g.add(10)) {
                if parent_bp>=0x1000 {
                    if let Some(event)=word(parent_bp-0x40) {
                        if read(event,&mut data[416..464]) {
                            mask|=1024;w64(&mut data,464,event);
                            w64(&mut data,472,*g.add(10));w64(&mut data,480,0x120ac95);
                            w64(&mut data,488,parent_bp);
                        }
                    }
                }
            }
        }
    } else if kind==1 {
        // Actual two CP-block destinations from the immediately preceding writes.
        if read(*g.add(6),&mut data[416..448]) { mask|=256; }
        if read(*g.add(11),&mut data[448..480]) { mask|=512; }
    }
    w32(&mut data,60,mask);w64(&mut data,352,loaded);
    let mut offset=0;
    while offset<data.len() {
        let n=sc(1,fd as usize,data[offset..].as_ptr()as usize,data.len()-offset,0,0,0);
        if n == -4 { continue }if n<=0 { fail(817) }offset+=n as usize;
    }
    match kind { 0=>{*g.add(10)=loaded;*g.add(16)=at+3;},1=>{*g.add(11)=loaded;*g.add(16)=at+7;},2=>{*g.add(13)=loaded;*g.add(16)=at+7;},_=>fail(818) }
    DEPTH.store(0,Ordering::SeqCst);true
}
