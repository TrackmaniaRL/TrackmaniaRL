//! Diagnostic contact88 -> waypoint32 -> event48 provenance. No terminal authority.
use super::*;
include!("event_signatures.rs");
static ENABLED:AtomicUsize=AtomicUsize::new(0);
static PID:AtomicUsize=AtomicUsize::new(0);
static TID:AtomicUsize=AtomicUsize::new(0);
static FD:AtomicI32=AtomicI32::new(-1);
static SEQUENCE:AtomicUsize=AtomicUsize::new(0);
pub unsafe fn configure(base:usize){
 for(off,sig)in EVENT_SITES {let mut bytes=[0;20];if !read(base+off,&mut bytes)||!same(&bytes,&sig){fail(851)}}
 ENABLED.store(base,Ordering::SeqCst);
}
pub unsafe fn activate(){
 let base=ENABLED.load(Ordering::SeqCst);if base==0{return}
 for(off,sig)in EVENT_SITES {let at=base+off;let mut bytes=[0;20];if !read(at,&mut bytes)||!same(&bytes,&sig){fail(852)}
  let page=at&!4095usize;if sc(10,page,4096,7,0,0,0)!=0{fail(853)}
  core::ptr::write_volatile(at as*mut u8,0xcc);if sc(10,page,4096,5,0,0,0)!=0{fail(854)}
 }
}
unsafe fn output(pid:usize,tid:usize)->i32{
 if PID.load(Ordering::SeqCst)!=pid {
  let old=FD.swap(-1,Ordering::SeqCst);if old>=0{sc(3,old as usize,0,0,0,0,0);}
  let mut path=[0u8;320];let n=DIR_LEN.load(Ordering::SeqCst);if n==0||n>240{fail(855)}
  copy_volatile(&(&*core::ptr::addr_of!(DIR))[..n],&mut path[..n]);copy_volatile(b"/event-",&mut path[n..n+7]);
  let mut digits=[0u8;24];let mut x=pid;let mut count=0;
  loop{digits[count]=b'0'+(x%10)as u8;count+=1;x/=10;if x==0{break}}
  for i in 0..count{path[n+7+i]=digits[count-i-1];}copy_volatile(b".bin",&mut path[n+7+count..n+11+count]);
  let fd=sc(257,(-100isize)as usize,path.as_ptr()as usize,0x800c1,0o600,0,0);if fd<0{fail(856)}
  PID.store(pid,Ordering::SeqCst);TID.store(tid,Ordering::SeqCst);FD.store(fd as i32,Ordering::SeqCst);SEQUENCE.store(0,Ordering::SeqCst);
 }
 if TID.load(Ordering::SeqCst)!=tid{fail(857)}FD.load(Ordering::SeqCst)
}
pub unsafe fn handle(info:*const u8,ctx:*mut c_void)->bool{
 let base=ENABLED.load(Ordering::SeqCst);if base==0||ctx.is_null(){return false}
 let g=(ctx as*mut usize).add(5);let rip=*g.add(16);
 let Some((kind,(off,sig)))=EVENT_SITES.iter().copied().enumerate().find(|(_, (off,_))|rip==base+off+1)else{return false};
 if info.is_null(){fail(858)}let code=*(info.add(8)as*const i32);if code!=1&&code!=128{fail(859)}
 if DEPTH.fetch_add(1,Ordering::SeqCst)!=0||*g.add(17)&0x10100!=0{fail(860)}
 let at=base+off;let mut bytes=[0;20];if !read(at,&mut bytes)||bytes[0]!=0xcc||!same(&bytes[1..],&sig[1..]){fail(861)}
 // CBC6BC: MOV R14D,[RBX+4c]. 120B15F: one-byte NOP after all event stores.
 let loaded=if kind==0{need(dword((*g.add(11)).wrapping_add(0x4c)),862)}else{0};
 let sim=VALIDATOR_SIM.load(Ordering::SeqCst);let controller=VALIDATOR_CONTROLLER.load(Ordering::SeqCst);
 if let Some(clock)=dword(sim.wrapping_add(0x48)) {
  if clock>=2190&&clock<=162200&&clock%10==0 {
   let pg=need(word(sim.wrapping_add(0x18)),863);let players=need(word(pg.wrapping_add(0x660)),863);let player=need(word(players),863);
   let vehicle=need(word(player.wrapping_add(0x1118)),863);
   if dword(pg.wrapping_add(0x668))!=Some(1)||word(player)!=Some(base+0x1c33938)||dword(player.wrapping_add(0x1110))!=Some(0x032e2000)||word(controller.wrapping_add(0x1a70))!=Some(sim){fail(864)}
   let pid=sc(39,0,0,0,0,0,0)as usize;let tid=sc(186,0,0,0,0,0,0)as usize;let fd=output(pid,tid);
   let sequence=SEQUENCE.fetch_add(1,Ordering::SeqCst)+1;if sequence>10000{fail(865)}
   let mut data=[0u8;1024];copy_volatile(b"TMEV0001",&mut data[..8]);w32(&mut data,8,1024);w32(&mut data,12,1);w32(&mut data,16,pid as u32);w32(&mut data,20,tid as u32);
   w64(&mut data,24,sequence);w32(&mut data,32,kind as u32);w32(&mut data,36,off as u32);w32(&mut data,40,clock);w32(&mut data,44,loaded);
   w64(&mut data,48,sim);w64(&mut data,56,controller);w64(&mut data,64,player);w64(&mut data,72,vehicle);
   for i in 0..23{w64(&mut data,80+i*8,*g.add(i));}
   let mut mask=0u32;
   if read(vehicle.wrapping_add(0x12e0),&mut data[272..312]){mask|=1;}
   if let Some(stamp)=dword(vehicle.wrapping_add(0x854)){w32(&mut data,312,stamp);mask|=2;}
   if kind==0 {
    let contact=*g.add(11);w64(&mut data,320,contact);
    if read(contact,&mut data[336..424]) {
     mask|=4;
     let body_a=need(word(contact),866);let body_b=need(word(contact.wrapping_add(8)),866);
     let ma=need(word(body_a.wrapping_add(0x38)),867);let mb=need(word(body_b.wrapping_add(0x38)),867);
     w64(&mut data,424,ma);w64(&mut data,432,mb);
     if read(ma,&mut data[448..464]){mask|=8;}
     if read(mb,&mut data[464..480]){mask|=16;}
     if let Some(trigger)=word(mb.wrapping_add(8)){w64(&mut data,480,trigger);if read(trigger,&mut data[496..544]){mask|=32;}}
     if let Some(index)=dword(base+0x1d23364){w32(&mut data,544,index);mask|=64;}
     // Same-thread native read, checked again before publication.
     let mut again=[0u8;88];if !read(contact,&mut again)||!same(&again,&data[336..424]){fail(868)}
    }
   }else{
    let source=*g.add(4);let dest=need((*g.add(7)).checked_add(*g.add(11)),869);
    w64(&mut data,320,source);w64(&mut data,328,dest);
    if read(source,&mut data[336..368]){mask|=4;}
    if read(dest,&mut data[368..416]){mask|=8;}
    if read(*g.add(6),&mut data[496..544]){mask|=32;}
    let mut again=[0u8;32];if !read(source,&mut again)||!same(&again,&data[336..368]){fail(870)}
   }
   w32(&mut data,268,mask);
   if dword(sim.wrapping_add(0x48))!=Some(clock)||word(player.wrapping_add(0x1118))!=Some(vehicle){fail(871)}
   let mut offset=0;while offset<data.len(){let n=sc(1,fd as usize,data[offset..].as_ptr()as usize,data.len()-offset,0,0,0);if n == -4{continue}if n<=0{fail(872)}offset+=n as usize;}
  }
 }
 if kind==0{*g.add(6)=loaded as usize;*g.add(16)=at+4;}else{*g.add(16)=at+1;}
 DEPTH.store(0,Ordering::SeqCst);true
}
