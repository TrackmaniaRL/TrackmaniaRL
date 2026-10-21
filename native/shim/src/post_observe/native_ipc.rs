//! Experimental V6: pinned4152 long archive; V5 ops, max native target50000.
//! Fixed frames, explicit phases, bounded original replay; not reset/full RL.
use super::*;
const PROFILE_INPUT_COUNT:usize=4152;
const MAX_NATIVE:usize=50000;
static START:AtomicUsize=AtomicUsize::new(0);
static NEXT:AtomicUsize=AtomicUsize::new(0);
static ANCHOR:AtomicUsize=AtomicUsize::new(0);
static ARMED:AtomicUsize=AtomicUsize::new(0);
static PID:AtomicUsize=AtomicUsize::new(0);
static TID:AtomicUsize=AtomicUsize::new(0);
static FD:AtomicI32=AtomicI32::new(-1);
static SEQ:AtomicUsize=AtomicUsize::new(0);
static EPOCH:AtomicU64=AtomicU64::new(0);
static STEPS:AtomicUsize=AtomicUsize::new(0);
static DONE:AtomicUsize=AtomicUsize::new(0);
static SOCKET_LEN:AtomicUsize=AtomicUsize::new(0);
static mut SOCKET_ADDR:[u8;110]=[0;110];
static PREP_PID:AtomicUsize=AtomicUsize::new(0);
static PREP_BASE:AtomicUsize=AtomicUsize::new(0);
static PREP_COUNT:AtomicUsize=AtomicUsize::new(0);
static PREP_SHADOW:AtomicUsize=AtomicUsize::new(0);
static INPUT_END:AtomicUsize=AtomicUsize::new(0);
static OBS_PID:AtomicUsize=AtomicUsize::new(0);
static MAX_INDEX:AtomicUsize=AtomicUsize::new(usize::MAX);
fn u32at(b:&[u8],o:usize)->u32{u32::from_le_bytes(b[o..o+4].try_into().unwrap())}
fn u64at(b:&[u8],o:usize)->u64{u64::from_le_bytes(b[o..o+8].try_into().unwrap())}
pub unsafe fn configure(){
 let Some(path)=std::env::var_os("FKSHIM_INNER_SOCKET")else{return};let bytes=path.as_os_str().as_bytes();
 let start=std::env::var("FKSHIM_INNER_START").ok().and_then(|v|v.parse::<usize>().ok()).unwrap_or(0);
 if bytes.is_empty()||bytes.len()>107||bytes.first()!=Some(&b'/')||bytes.contains(&0)||start<2000||start>18000||start%10!=0{fail(301)}
 let addr=&mut*core::ptr::addr_of_mut!(SOCKET_ADDR);addr[0]=1;addr[2..2+bytes.len()].copy_from_slice(bytes);
 SOCKET_LEN.store(bytes.len(),Ordering::SeqCst);START.store(start,Ordering::SeqCst);NEXT.store(start,Ordering::SeqCst);
}
/// G command path only, outside any signal handler. All allocations happen here.
pub unsafe fn prepare(){
 if START.load(Ordering::SeqCst)==0{return}
 let b=CACHED_BASE.load(Ordering::SeqCst);let n=EXPECT_N.load(Ordering::SeqCst);let shadow=EXPECT.load(Ordering::SeqCst);
 if n!=PROFILE_INPUT_COUNT{fail(356)}
 if b<4096||n==0||n>200000||shadow<4096||SEG_N.load(Ordering::SeqCst)!=2||base_still_holds(b).is_some(){fail(302)}
 // Exact NativeSimState52 provenance. This is NOT the old Gclock layout.
 let sim=VALIDATOR_SIM.load(Ordering::SeqCst);let controller=VALIDATOR_CONTROLLER.load(Ordering::SeqCst);let module=MODULE_BASE.load(Ordering::SeqCst);
 let pg=need(word(sim.wrapping_add(0x18)),346);let players=need(word(pg.wrapping_add(0x660)),346);let participant=need(word(players),346);let vehicle=need(word(participant.wrapping_add(0x1118)),346);
 if sim<4096||controller<4096||word(controller.wrapping_add(0x1a70))!=Some(sim)||dword(pg.wrapping_add(0x668))!=Some(1)||word(participant)!=Some(module+0x1c33938)||dword(participant.wrapping_add(0x1110))!=Some(0x032e2000)||vehicle<4096||SEG_LEN[0].load(Ordering::SeqCst)!=4||SEG_LEN[1].load(Ordering::SeqCst)!=40||SEG_ADDR[0].load(Ordering::SeqCst)!=sim+0x48||SEG_ADDR[1].load(Ordering::SeqCst)!=vehicle+0x12e0{fail(346)}
 let end=need(b.checked_sub(4).and_then(|x|x.checked_add(n*32)),303);let se=need(shadow.checked_add(n*4),304);
 let maps=std::fs::read_to_string("/proc/self/maps").unwrap_or_default();
 let writable=|low:usize,high:usize|maps.lines().any(|line|{let mut parts=line.split_whitespace();let Some(range)=parts.next()else{return false};let Some(perms)=parts.next()else{return false};let Some((a,z))=range.split_once('-')else{return false};match(usize::from_str_radix(a,16),usize::from_str_radix(z,16)){(Ok(a),Ok(z))=>perms.starts_with("rw")&&a<=low&&high<=z,_=>false}});
 if !writable(b-4,end)||!writable(shadow,se){fail(305)}
 PREP_BASE.store(b,Ordering::SeqCst);PREP_COUNT.store(n,Ordering::SeqCst);PREP_SHADOW.store(shadow,Ordering::SeqCst);INPUT_END.store(end,Ordering::SeqCst);PREP_PID.store(sc(39,0,0,0,0,0,0)as usize,Ordering::SeqCst);
}
unsafe fn send(fd:i32,b:&[u8]){let mut off=0;while off<b.len(){let n=sc(44,fd as usize,b[off..].as_ptr()as usize,b.len()-off,0x4000,0,0);if n==-4{continue}if n<=0{fail(306)}off+=n as usize}}
unsafe fn receive(fd:i32,b:&mut[u8]){let mut off=0;while off<b.len(){let n=sc(45,fd as usize,b[off..].as_mut_ptr()as usize,b.len()-off,0,0,0);if n==-4{continue}if n<=0{fail(307)}off+=n as usize}}
unsafe fn connect_socket()->i32{
 let fd=sc(41,1,0x80001,0,0,0,0)as i32;if fd<0{fail(308)}let timeout=[30i64,0];
 for option in[20usize,21]{if sc(54,fd as usize,1,option,timeout.as_ptr()as usize,16,0)!=0{fail(309)}}
 if sc(42,fd as usize,core::ptr::addr_of!(SOCKET_ADDR)as usize,SOCKET_LEN.load(Ordering::SeqCst)+3,0,0,0)!=0{fail(310)}fd
}
unsafe fn owner(){if PID.load(Ordering::SeqCst)!=sc(39,0,0,0,0,0,0)as usize||TID.load(Ordering::SeqCst)!=sc(186,0,0,0,0,0,0)as usize{fail(311)}}
unsafe fn respond(kind:u8,event:&[u8;256],receipt:&[u8;160],flags:u32){
 let mut h=[0u8;32];h[..4].copy_from_slice(b"TMR6");h[4]=6;h[5]=kind;w32(&mut h,8,SEQ.load(Ordering::SeqCst)as u32);w32(&mut h,12,STEPS.load(Ordering::SeqCst)as u32);w64(&mut h,16,EPOCH.load(Ordering::SeqCst)as usize);w32(&mut h,24,416);w32(&mut h,28,flags);
 let fd=FD.load(Ordering::SeqCst);send(fd,&h);send(fd,event);send(fd,receipt);
}
unsafe fn next_epoch(){let old=EPOCH.load(Ordering::SeqCst);let Some(new)=old.checked_add(1)else{fail(312)};EPOCH.store(new,Ordering::SeqCst)}
fn zero(b:&[u8])->bool{b.iter().all(|v|*v==0)}
unsafe fn target(anchor:usize,n:u32)->usize{if PREP_COUNT.load(Ordering::SeqCst)!=PROFILE_INPUT_COUNT{fail(356)}if n==0||n>1000{fail(313)}let x=need(anchor.checked_add(n as usize*10),314);if x>MAX_NATIVE{fail(315)}x}
unsafe fn write_memory(address:usize,data:&[u8])->bool{
 let local=Iovec{base:data.as_ptr()as*mut c_void,len:data.len()};let remote=Iovec{base:address as*mut c_void,len:data.len()};
 sc(311,sc(39,0,0,0,0,0,0)as usize,&local as*const _ as usize,1,&remote as*const _ as usize,1,0)==data.len()as isize
}
unsafe fn audit(status:u32,p:u32,k:u32,n:u32,seq:u32,kbefore:&[u8;32],kafter:&[u8;32],pbefore:&[u8;32],pafter:&[u8;32],sbefore:&[u8;4],safter:&[u8;4],length:u32,ordinal:u32){
 let pid=sc(39,0,0,0,0,0,0)as usize;let tid=sc(186,0,0,0,0,0,0)as usize;let fd=process_log(pid,tid);let hit=POST_HITS.fetch_add(1,Ordering::SeqCst)+1;if hit>50000{fail(316)}
 let mut d=[0u8;256];d[..8].copy_from_slice(b"TMAUD002");w32(&mut d,8,256);w32(&mut d,12,2);w32(&mut d,16,pid as u32);w32(&mut d,20,tid as u32);w64(&mut d,24,hit as usize);w64(&mut d,32,N_LROUNDF.load(Ordering::SeqCst)as usize);
 d[40..72].copy_from_slice(kbefore);d[72..104].copy_from_slice(kafter);d[104..136].copy_from_slice(pbefore);d[136..168].copy_from_slice(pafter);d[168..172].copy_from_slice(sbefore);d[172..176].copy_from_slice(safter);
 for(o,v)in[(176,p),(180,k),(184,n),(188,ANCHOR.load(Ordering::SeqCst)as u32),(200,seq),(204,status),(208,MAX_INDEX.load(Ordering::SeqCst)as u32),(212,12)]{w32(&mut d,o,v)}w64(&mut d,192,EPOCH.load(Ordering::SeqCst)as usize);
 let b=PREP_BASE.load(Ordering::SeqCst);w64(&mut d,216,b);w64(&mut d,224,PREP_SHADOW.load(Ordering::SeqCst));w64(&mut d,232,b-4+k as usize*32);w64(&mut d,240,b-4+p as usize*32);
 w32(&mut d,248,length);w32(&mut d,252,ordinal);
 let mut off=0;while off<d.len(){let x=sc(1,fd as usize,d[off..].as_ptr()as usize,d.len()-off,0,0,0);if x==-4{continue}if x<=0{fail(317)}off+=x as usize}
}
unsafe fn apply(request:&[u8;192],event:&[u8;256],n:u32,seq:u32,receipt:&mut[u8;160]){
 let p=u32at(event,112);let count=u32at(event,116);let k=u32at(request,28);let length=u32at(request,32);
 let b=PREP_BASE.load(Ordering::SeqCst);let shadow=PREP_SHADOW.load(Ordering::SeqCst);let sn=PREP_COUNT.load(Ordering::SeqCst);
 if length==0||length>5{fail(344)}
 if PREP_PID.load(Ordering::SeqCst)!=PID.load(Ordering::SeqCst)||u64at(event,88)as usize!=b||CACHED_BASE.load(Ordering::SeqCst)!=b||EXPECT.load(Ordering::SeqCst)!=shadow||EXPECT_N.load(Ordering::SeqCst)!=sn||count as usize!=sn||u32at(request,24)!=p||k.checked_add(length).map(|x|x<=count)!=Some(true){fail(318)}
 let ka=need(b.checked_sub(4).and_then(|x|x.checked_add(k as usize*32)),319);let pa=need(b.checked_sub(4).and_then(|x|x.checked_add(p as usize*32)),320);let sa=need(shadow.checked_add(k as usize*4),321);
 if ka.checked_add(length as usize*32).map(|x|x<=INPUT_END.load(Ordering::SeqCst))!=Some(true)||pa.checked_add(32).map(|x|x<=INPUT_END.load(Ordering::SeqCst))!=Some(true){fail(322)}
 let mut kb=[[0u8;32];5];let mut sb=[[0u8;4];5];let mut pb=[0;32];if !read(pa,&mut pb){fail(323)}
 for j in 0..length as usize{if !read(ka+j*32,&mut kb[j])||!read(sa+j*4,&mut sb[j]){fail(323)}}
 // Revalidate the captured reader authorization after the blocking IPC wait.
 let cbp=u64at(event,64)as usize;let rbp=u64at(event,56)as usize;
 let sim=u64at(event,48)as usize;let controller=u64at(event,40)as usize;
 let participant=u64at(event,192)as usize;let context=u64at(event,200)as usize;
 let module=MODULE_BASE.load(Ordering::SeqCst);let anchor=ANCHOR.load(Ordering::SeqCst)as u32;
 if !same(&pb,&event[152..184]){audit(342,p,k,n,seq,&kb[0],&kb[0],&pb,&pb,&sb[0],&sb[0],length,0);fail(342)}
 if sim!=VALIDATOR_SIM.load(Ordering::SeqCst)||controller!=VALIDATOR_CONTROLLER.load(Ordering::SeqCst)||word(controller.wrapping_add(0x1a70))!=Some(sim)||dword(sim.wrapping_add(0x48))!=Some(anchor)||dword(cbp.wrapping_sub(0x30))!=Some(anchor+10)||word(cbp.wrapping_sub(0x48))!=Some(sim)||word(rbp.wrapping_add(8))!=Some(module+0x1219f8e)||word(rbp)!=Some(cbp)||word(context.wrapping_add(0x20d8))!=Some(b-4)||dword(context.wrapping_add(0x20e0))!=Some(count)||word(participant)!=Some(module+0x1c33938)||dword(participant.wrapping_add(0x1110))!=Some(0x032e2000){audit(343,p,k,n,seq,&kb[0],&kb[0],&pb,&pb,&sb[0],&sb[0],length,0);fail(343)}

 // Authorize every member before the first write; failure never resumes.
 for j in 0..length as usize{
  let off=40+j*24;let mut reject=0;
  if k<=p||MAX_INDEX.load(Ordering::SeqCst)!=p as usize||k as usize<=MAX_INDEX.load(Ordering::SeqCst){reject=324}
  else if !same(&kb[j][4..16],&request[off..off+12])||!same(&sb[j],&kb[j][4..8]){reject=325}
  else{let steer=f32::from_le_bytes(request[off+12..off+16].try_into().unwrap());let gas=u32at(request,off+16);let brake=u32at(request,off+20);if !steer.is_finite()||!(-1.0..=1.0).contains(&steer)||!(gas==0||gas==0x3f800000)||!(brake==0||brake==0x3f800000){reject=326}}
  if reject!=0{for q in 0..length as usize{audit(reject,p,k+q as u32,n,seq,&kb[q],&kb[q],&pb,&pb,&sb[q],&sb[q],length,q as u32)}fail(reject)}
 }
 w32(receipt,0,p);w32(receipt,4,k);w32(receipt,8,length);
 for j in 0..length as usize{
  let off=40+j*24;let desired=&request[off+12..off+24];
  // Multiple writes are not an atomic transaction. A partial error terminates.
  if !write_memory(ka+j*32+4,desired){fail(327)}if !write_memory(sa+j*4,&desired[..4]){fail(328)}
 }
 // Verify and audit the whole range after all writes, including current P.
 for j in 0..length as usize{
  let off=40+j*24;let desired=&request[off+12..off+24];let mut after=[0;32];let mut pafter=[0;32];let mut safter=[0;4];
  if !read(ka+j*32,&mut after)||!read(pa,&mut pafter)||!read(sa+j*4,&mut safter){fail(329)}
  if !same(&after[..4],&kb[j][..4])||!same(&after[16..],&kb[j][16..])||!same(&after[4..16],desired)||!same(&pb,&pafter)||!same(&safter,&desired[..4]){audit(330,p,k+j as u32,n,seq,&kb[j],&after,&pb,&pafter,&sb[j],&safter,length,j as u32);fail(330)}
  audit(1,p,k+j as u32,n,seq,&kb[j],&after,&pb,&pafter,&sb[j],&safter,length,j as u32);
  let ro=16+j*24;receipt[ro..ro+12].copy_from_slice(&kb[j][4..16]);receipt[ro+12..ro+24].copy_from_slice(desired);
 }
}
// V5-only no-write last-reader release. This runs inside the owned reader trap.
// ACK proves authorization/readback, not consumption or a physical boundary.
#[inline(never)]
unsafe fn zero_release(bytes:&[u8])->bool{
 let mut at=0;while at<bytes.len(){if core::ptr::read_volatile(bytes.as_ptr().add(at))!=0{return false}at+=1}true
}
unsafe fn release_current(request:&[u8;192],event:&[u8;256],receipt:&mut[u8;160])->usize{
 owner();
 if DEPTH.load(Ordering::SeqCst)!=1||ARMED.load(Ordering::SeqCst)!=1||DONE.load(Ordering::SeqCst)!=0||FD.load(Ordering::SeqCst)<0||NEXT.load(Ordering::SeqCst)!=usize::MAX{fail(355)}
 if u32at(request,12)!=1||!zero_release(&request[28..]){fail(347)}
 let pid=PID.load(Ordering::SeqCst);let tid=TID.load(Ordering::SeqCst);
 if !same(&event[..8],b"TMREAD01")||u32at(event,8)!=256||u32at(event,12)!=1||u32at(event,16)as usize!=pid||u32at(event,20)as usize!=tid||u32at(event,148)!=447||u32at(event,208)!=1||u32at(event,212)!=11{fail(352)}
 let p=u32at(event,112);let count=u32at(event,116);
 if count==0||count>200000||p.checked_add(1)!=Some(count)||u32at(request,24)!=p{fail(348)}
 let b=PREP_BASE.load(Ordering::SeqCst);let shadow=PREP_SHADOW.load(Ordering::SeqCst);let sn=PREP_COUNT.load(Ordering::SeqCst);
 if PREP_PID.load(Ordering::SeqCst)!=pid||OBS_PID.load(Ordering::SeqCst)!=pid||MAX_INDEX.load(Ordering::SeqCst)!=p as usize||CACHED_BASE.load(Ordering::SeqCst)!=b||EXPECT.load(Ordering::SeqCst)!=shadow||EXPECT_N.load(Ordering::SeqCst)!=sn||sn!=count as usize||u64at(event,88)as usize!=b||u64at(event,104)as usize!=sn||b<4096||shadow<4096{fail(353)}
 let pa=need(b.checked_sub(4).and_then(|x|x.checked_add(p as usize*32)),350);
 let sa=need(shadow.checked_add(p as usize*4),350);
 if pa.checked_add(32).map(|end|end<=INPUT_END.load(Ordering::SeqCst))!=Some(true){fail(350)}
 let anchor=ANCHOR.load(Ordering::SeqCst);let t=target(anchor,1);
 let cbp=u64at(event,64)as usize;let rbp=u64at(event,56)as usize;
 let sim=u64at(event,48)as usize;let controller=u64at(event,40)as usize;
 let participant=u64at(event,192)as usize;let context=u64at(event,200)as usize;let module=MODULE_BASE.load(Ordering::SeqCst);
 // The original reader remains the sole code writer and instruction emulator.
 let patched:[u8;20]=[0xcc,0xf1,0x48,0xc1,0xe1,0x05,0x0f,0x10,0x04,0x0a,0x0f,0x10,0x4c,0x0a,0x10,0xeb,0xba,0x48,0x8d,0x15];
 let mut sig=[0u8;20];let reader_at=need(module.checked_add(0x119f163),349);
 if !read(reader_at,&mut sig)||!same(&sig,&patched)||word(rbp.wrapping_add(8))!=Some(module+0x1219f8e)||word(rbp)!=Some(cbp)||cbp.checked_sub(rbp)!=Some(0x1a0)||u64at(event,72)as usize!=module+0x1219f8e{fail(349)}
 if sim!=VALIDATOR_SIM.load(Ordering::SeqCst)||controller!=VALIDATOR_CONTROLLER.load(Ordering::SeqCst)||word(controller.wrapping_add(0x1a70))!=Some(sim)||dword(sim.wrapping_add(0x48))!=Some(anchor as u32)||dword(cbp.wrapping_sub(0x30))!=Some(t as u32)||word(cbp.wrapping_sub(0x48))!=Some(sim)||u32at(event,120)as usize!=t||u32at(event,124)as usize!=anchor||word(context.wrapping_add(0x20d8))!=Some(b-4)||dword(context.wrapping_add(0x20e0))!=Some(count)||word(participant)!=Some(module+0x1c33938)||dword(participant.wrapping_add(0x1110))!=Some(0x032e2000){fail(349)}
 let mut before=[0u8;32];let mut s_before=[0u8;4];
 if !read(pa,&mut before)||!read(sa,&mut s_before)||!same(&before,&event[152..184]){fail(350)}
 if !same(&s_before,&before[4..8])||!same(&s_before,&event[184..188]){fail(351)}
 // No write_memory, process_vm_writev, input/shadow store or new GPR/FP writes.
 let mut after=[0u8;32];let mut s_after=[0u8;4];
 if !read(pa,&mut after)||!read(sa,&mut s_after)||!same(&before,&after)||!same(&s_before,&s_after){fail(354)}
 owner();
 if u64at(request,16)!=EPOCH.load(Ordering::SeqCst)||MAX_INDEX.load(Ordering::SeqCst)!=p as usize||ARMED.load(Ordering::SeqCst)!=1||dword(sim.wrapping_add(0x48))!=Some(anchor as u32){fail(354)}
 w32(receipt,0,p);w32(receipt,4,count);w32(receipt,8,1);w32(receipt,12,1);
 copy_volatile(&before,&mut receipt[16..48]);copy_volatile(&s_before,&mut receipt[48..52]);
 t
}

unsafe fn serve(event:&[u8;256],reader:bool){
 owner();next_epoch();respond(if reader{3}else{1},event,&[0;160],0);
 let mut request=[0u8;192];receive(FD.load(Ordering::SeqCst),&mut request);
 let seq=SEQ.load(Ordering::SeqCst)as u32;let newseq=u32at(&request,8);let n=u32at(&request,12);
 if &request[..4]!=b"TMA6"||request[4]!=6||request[6]!=0||request[7]!=0||seq==u32::MAX||newseq!=seq+1||u64at(&request,16)!=EPOCH.load(Ordering::SeqCst)||!zero(&request[36..40])||!zero(&request[160..]){fail(331)}
 if request[5]!=4&&request[5]!=5&&!zero(&request[24..160]){fail(332)}
 if request[5]==4{let l=u32at(&request,32);if l==0||l>5{fail(344)}if !zero(&request[40+l as usize*24..160]){fail(345)}}
 match(request[5],reader){
  (1,false)=>{let t=target(u32at(event,112)as usize,n);NEXT.store(t,Ordering::SeqCst);SEQ.store(newseq as usize,Ordering::SeqCst);STEPS.store(n as usize,Ordering::SeqCst);},
  (2,false)=>{if n!=0{fail(333)}SEQ.store(newseq as usize,Ordering::SeqCst);STEPS.store(0,Ordering::SeqCst);respond(2,event,&[0;160],0);DONE.store(1,Ordering::SeqCst);let fd=FD.swap(-1,Ordering::SeqCst);sc(3,fd as usize,0,0,0,0,0);},
  (3,false)=>{if n!=0{fail(334)}ANCHOR.store(u32at(event,112)as usize,Ordering::SeqCst);ARMED.store(1,Ordering::SeqCst);NEXT.store(usize::MAX,Ordering::SeqCst);SEQ.store(newseq as usize,Ordering::SeqCst);STEPS.store(0,Ordering::SeqCst);},
  (4,true)=>{let t=target(ANCHOR.load(Ordering::SeqCst),n);let mut receipt=[0;160];apply(&request,event,n,newseq,&mut receipt);SEQ.store(newseq as usize,Ordering::SeqCst);STEPS.store(n as usize,Ordering::SeqCst);respond(4,event,&receipt,1);NEXT.store(t,Ordering::SeqCst);ARMED.store(0,Ordering::SeqCst);},
  (5,true)=>{let mut receipt=[0u8;160];let t=release_current(&request,event,&mut receipt);SEQ.store(newseq as usize,Ordering::SeqCst);STEPS.store(1,Ordering::SeqCst);respond(5,event,&receipt,2);NEXT.store(t,Ordering::SeqCst);ARMED.store(0,Ordering::SeqCst);},
  _=>fail(335)
 }
}
pub unsafe fn at_boundary(event:&[u8;256]){
 if START.load(Ordering::SeqCst)==0||DONE.load(Ordering::SeqCst)!=0{return}
 let current=u32at(event,112)as usize;if PID.load(Ordering::SeqCst)==0{if SEG_N.load(Ordering::SeqCst)!=2{return}}else{owner()}
 if u32at(event,148)!=75||SEG_N.load(Ordering::SeqCst)!=2||SEG_LEN[0].load(Ordering::SeqCst)!=4||SEG_LEN[1].load(Ordering::SeqCst)!=40||SEG_ADDR[0].load(Ordering::SeqCst)!=u64at(event,48)as usize+0x48||SEG_ADDR[1].load(Ordering::SeqCst)!=u64at(event,80)as usize+0x12e0||u32at(event,112)!=u32at(event,116){fail(336)}if ARMED.load(Ordering::SeqCst)!=0{fail(337)}
 let target=NEXT.load(Ordering::SeqCst);if current<target{return}if current!=target{fail(338)}
 if FD.load(Ordering::SeqCst)<0{
  let pid=sc(39,0,0,0,0,0,0)as usize;if pid!=PREP_PID.load(Ordering::SeqCst){fail(339)}
  if SEG_LEN[0].load(Ordering::SeqCst)!=4||SEG_LEN[1].load(Ordering::SeqCst)!=40{fail(340)}
  PID.store(pid,Ordering::SeqCst);TID.store(sc(186,0,0,0,0,0,0)as usize,Ordering::SeqCst);FD.store(connect_socket(),Ordering::SeqCst);
 }
 serve(event,false)
}
pub unsafe fn at_reader(event:&[u8;256]){
 if START.load(Ordering::SeqCst)==0||DONE.load(Ordering::SeqCst)!=0{return}
 let pid=sc(39,0,0,0,0,0,0)as usize;
 if OBS_PID.load(Ordering::SeqCst)!=pid{OBS_PID.store(pid,Ordering::SeqCst);MAX_INDEX.store(usize::MAX,Ordering::SeqCst)}
 let p=u32at(event,112)as usize;
 if u64at(event,88)>4096&&u64at(event,80).checked_add(4)==Some(u64at(event,88))&&u32at(event,116)as u64==u64at(event,104)&&p<(u32at(event,116)as usize){let old=MAX_INDEX.load(Ordering::SeqCst);if old==usize::MAX||p>old{MAX_INDEX.store(p,Ordering::SeqCst)}}
 if ARMED.load(Ordering::SeqCst)==0{return}owner();
 if u32at(event,208)!=1||u32at(event,148)&0x1bf!=0x1bf||MAX_INDEX.load(Ordering::SeqCst)!=p||u32at(event,120)as usize!=ANCHOR.load(Ordering::SeqCst)+10||u32at(event,124)as usize!=ANCHOR.load(Ordering::SeqCst){fail(341)}
 serve(event,true)
}
