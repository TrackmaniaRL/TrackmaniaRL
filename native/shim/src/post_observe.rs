//! Versioned logging-only persistent POST trap. It emulates exactly one guarded MOV.
//! No TF/rearm window, no input writes, no step/reset claim. Linux x86-64 only.
use super::*;
use std::os::unix::ffi::OsStrExt;
const POST_OFF: usize=0x118d839;
const SIGNATURE:[u8;20]=[0x44,0x8b,0xbb,0x30,0x0e,0,0,0x45,0x39,0xef,0x0f,0x95,0xc0,0x8b,0x4d,0x80,0x20,0xc8,0x3c,0x01];
const INNER_OFF:usize=0x1219774;
const INNER_SIG:[u8;20]=[0x89,0xd8,0x44,0x1,0xf0,0x3b,0x45,0xc0,0x48,0x8b,0x5d,0x98,0xf,0x87,0xda,0x9,0x0,0x0,0x89,0x45];
const ENTRIES:[(usize,u32,[u8;20]);4]=[(0x1218db1,8,[0x48,0x89,0xe5,0x41,0x57,0x41,0x56,0x41,0x55,0x41,0x54,0x53,0x48,0x81,0xec,0x68,0x1,0x0,0x0,0x89]),(0x118c371,7,[0x48,0x89,0xe5,0x41,0x57,0x41,0x56,0x41,0x55,0x41,0x54,0x53,0x48,0x81,0xec,0x88,0x2,0x0,0x0,0x49]),(0x118d8d1,24,[0x48,0x89,0xe5,0x41,0x57,0x41,0x56,0x41,0x55,0x41,0x54,0x53,0x50,0x48,0x89,0xfb,0xbf,0xb0,0x0,0x0]),(0x11b9071,3,[0x48,0x89,0xe5,0x41,0x57,0x41,0x56,0x41,0x55,0x41,0x54,0x53,0x48,0x81,0xec,0xa8,0x1,0x0,0x0,0x89])];
static POST_AT:AtomicUsize=AtomicUsize::new(0);
static OWNER_PID:AtomicUsize=AtomicUsize::new(0);
static OWNER_TID:AtomicUsize=AtomicUsize::new(0);
static DEPTH:AtomicUsize=AtomicUsize::new(0);
static POST_HITS:AtomicU64=AtomicU64::new(0);
static POST_FD:AtomicI32=AtomicI32::new(-1);
static DIR_LEN:AtomicUsize=AtomicUsize::new(0);
static mut DIR:[u8;256]=[0;256];
#[repr(C)]
struct Iovec { base:*mut c_void,len:usize }
unsafe fn sc(n:usize,a:usize,b:usize,c:usize,d:usize,e:usize,f:usize)->isize {
 let result:isize;
 std::arch::asm!("syscall",inlateout("rax") n as isize=>result,in("rdi")a,in("rsi")b,in("rdx")c,in("r10")d,in("r8")e,in("r9")f,lateout("rcx")_,lateout("r11")_,options(nostack));result
}
unsafe fn read(a:usize,b:&mut[u8])->bool {
 if a<4096||a.checked_add(b.len()).is_none(){return false}
 let l=Iovec{base:b.as_mut_ptr()as*mut c_void,len:b.len()};let r=Iovec{base:a as*mut c_void,len:b.len()};
 sc(310,sc(39,0,0,0,0,0,0)as usize,&l as*const _ as usize,1,&r as*const _ as usize,1,0)==b.len()as isize
}

/// Volatile byte reads keep the compiler from lowering guarded comparisons to libc.
#[inline(never)]
unsafe fn same(a:&[u8],b:&[u8])->bool{
 if a.len()!=b.len(){return false}let mut i=0;
 while i<a.len(){if core::ptr::read_volatile(a.as_ptr().add(i))!=core::ptr::read_volatile(b.as_ptr().add(i)){return false}i+=1}true
}

#[inline(never)]
unsafe fn copy_volatile(source:&[u8],dest:&mut[u8]){
 if source.len()!=dest.len(){fail(117)}let mut i=0;
 while i<source.len(){core::ptr::write_volatile(dest.as_mut_ptr().add(i),core::ptr::read_volatile(source.as_ptr().add(i)));i+=1}
}
unsafe fn word(a:usize)->Option<usize>{let mut b=[0;8];if read(a,&mut b){Some(u64::from_le_bytes(b)as usize)}else{None}}
unsafe fn dword(a:usize)->Option<u32>{let mut b=[0;4];if read(a,&mut b){Some(u32::from_le_bytes(b))}else{None}}
unsafe fn fail(code:u32)->!{let mut b=*b"FKPOST reject 000\n";b[14]=b'0'+((code/100)%10)as u8;b[15]=b'0'+((code/10)%10)as u8;b[16]=b'0'+(code%10)as u8;sc(1,2,b.as_ptr()as usize,b.len(),0,0,0);sc(231,95,0,0,0,0,0);core::hint::unreachable_unchecked()}
unsafe fn need<T>(x:Option<T>,code:u32)->T{match x{Some(v)=>v,None=>fail(code)}}
fn w32(b:&mut[u8],o:usize,x:u32){b[o..o+4].copy_from_slice(&x.to_le_bytes())}
fn w64(b:&mut[u8],o:usize,x:usize){b[o..o+8].copy_from_slice(&(x as u64).to_le_bytes())}
/// Initialization only: allocations/environment are forbidden in the signal path.
mod native_ipc;
mod reader;
mod result_observe;
pub unsafe fn configure(base:usize){
 native_ipc::configure();
 reader::configure(base);
 let Some(dir)=std::env::var_os("FKSHIM_POST_DIR")else{return};let bytes=dir.as_os_str().as_bytes();
 if getpagesize()!=4096{fail(34)}
 if base==0||bytes.len()>240||bytes.first()!=Some(&b'/')||bytes.contains(&0)||!std::path::Path::new(&dir).is_dir(){fail(1)}
 let at=base+POST_OFF;let mut got=[0;20];if !read(at,&mut got)||got!=SIGNATURE{fail(2)}
 let mut inner=[0;20];if !read(base+INNER_OFF,&mut inner)||inner!=INNER_SIG{fail(66)}
 for (off,_,sig) in ENTRIES {let mut bytes=[0;20];if !read(base+off,&mut bytes)||bytes!=sig{fail(36)}}
 (&mut*core::ptr::addr_of_mut!(DIR))[..bytes.len()].copy_from_slice(bytes);DIR_LEN.store(bytes.len(),Ordering::SeqCst);POST_AT.store(at,Ordering::SeqCst);
 result_observe::configure(base);
}
/// Called by the single existing binding owner, after restoring its original byte.
/// True means the integrated router stays installed; no second sigaction restorer.
pub unsafe fn activate()->bool{
 let at=POST_AT.load(Ordering::SeqCst);if at==0{return false}
 let mut got=[0;20];if !read(at,&mut got)||got!=SIGNATURE{fail(3)}
 let page=at&!4095usize;if sc(10,page,4096,7,0,0,0)!=0{fail(4)}
 core::ptr::write_volatile(at as*mut u8,0xcc);if sc(10,page,4096,5,0,0,0)!=0{fail(5)}
 let base=MODULE_BASE.load(Ordering::SeqCst);
 for(off,_,sig)in ENTRIES{let point=base+off;let mut b=[0;20];if !read(point,&mut b)||b!=sig{fail(37)}let page=point&!4095usize;if sc(10,page,4096,7,0,0,0)!=0{fail(38)}core::ptr::write_volatile(point as*mut u8,0xcc);if sc(10,page,4096,5,0,0,0)!=0{fail(39)}}
 let point=base+INNER_OFF;let mut b=[0;20];if !read(point,&mut b)||b!=INNER_SIG{fail(67)}let page=point&!4095usize;if sc(10,page,4096,7,0,0,0)!=0{fail(68)}core::ptr::write_volatile(point as*mut u8,0xcc);if sc(10,page,4096,5,0,0,0)!=0{fail(69)}reader::activate();result_observe::activate();true
}
unsafe fn process_log(pid:usize,tid:usize)->i32{
 if OWNER_PID.load(Ordering::SeqCst)!=pid{
  let old=POST_FD.swap(-1,Ordering::SeqCst);if old>=0{sc(3,old as usize,0,0,0,0,0);}
  let n=DIR_LEN.load(Ordering::SeqCst);let mut raw=core::mem::MaybeUninit::<[u8;288]>::uninit();let bytes=raw.as_mut_ptr()as*mut u8;for i in 0..288{core::ptr::write_volatile(bytes.add(i),0)}let path=raw.assume_init_mut();copy_volatile(&(&*core::ptr::addr_of!(DIR))[..n],&mut path[..n]);
  path[n..n+6].copy_from_slice(b"/post-");let mut digits=[0u8;20];let mut p=pid;let mut len=0;
  loop{digits[len]=b'0'+(p%10)as u8;len+=1;p/=10;if p==0{break}}
  for j in 0..len{path[n+6+j]=digits[len-1-j];}path[n+6+len..n+10+len].copy_from_slice(b".bin");
  let fd=sc(257,(-100isize)as usize,path.as_ptr()as usize,0x80000|0x80|0x40|1,0o600,0,0);if fd<0{fail(6)}
  POST_HITS.store(0,Ordering::SeqCst);OWNER_TID.store(tid,Ordering::SeqCst);POST_FD.store(fd as i32,Ordering::SeqCst);OWNER_PID.store(pid,Ordering::SeqCst);
 }
 if OWNER_TID.load(Ordering::SeqCst)!=tid{fail(7)}POST_FD.load(Ordering::SeqCst)
}
/// Handle only our exact persistent POST breakpoint. Everything else remains owned
/// by the existing binding trap handler, including its explicit foreign-trap failure.
pub unsafe fn handle(info:*const u8,ctx:*mut c_void)->bool{
 if result_observe::handle(info,ctx){return true}
 if reader::handle(info,ctx){return true}
 if inner_handle(info,ctx){return true}
 if entry_handle(info,ctx){return true}
 let at=POST_AT.load(Ordering::SeqCst);if at==0||ctx.is_null(){return false}
 let g=(ctx as*mut usize).add(5);if *g.add(16)!=at+1{return false}
 if info.is_null(){fail(8)}let code=*(info.add(8)as*const i32);if code!=1&&code!=128{fail(9)}
 if DEPTH.fetch_add(1,Ordering::SeqCst)!=0{fail(10)}
 if *g.add(17)&0x10100!=0{fail(35)}
 let mut signature=[0;20];if !read(at,&mut signature)||signature[0]!=0xcc||!same(&signature[1..],&SIGNATURE[1..]){fail(11)}
 let pid=sc(39,0,0,0,0,0,0)as usize;let tid=sc(186,0,0,0,0,0,0)as usize;let fd=process_log(pid,tid);
 let context=*g.add(11);let bp=*g.add(10);if bp<0x1000{fail(12)}
 // This exact 4-byte read supplies BOTH the recorded P and emulated MOV result.
 let processed=need(dword(need(context.checked_add(0xe30),13)),14);
 let controller=need(word(bp-0x40),15);let sim=VALIDATOR_SIM.load(Ordering::SeqCst);
 if controller!=VALIDATOR_CONTROLLER.load(Ordering::SeqCst)||sim<4096||need(word(controller+0x1a70),16)!=sim{fail(17)}
 let count=need(dword(controller+0x1a80),18);if count!=1||VALIDATOR_PARTICIPANT_COUNT_ARG.load(Ordering::SeqCst)!=1{fail(19)}
 let vector=need(word(bp-0x50),20);let cb=need(word(bp-0x30),21);let index=*g.add(4);
 if cb!=8||index!=0||vector!=need(word(controller+0x1a78),22)||need(word(vector),23)!=context{fail(24)}
 let requested=*g.add(5)as u32;let frame_time=need(dword(bp-0x58),25);if requested!=frame_time{fail(26)}
 let status=need(dword(context+0xe34),27);let flags=need(dword(context+0xef8),28);
 let special=need(dword(bp-0x80),29);let mode=*g.add(6)as u32;
 let sim_clock=need(dword(sim+0x48),30);let global=need(dword(MODULE_BASE.load(Ordering::SeqCst)+0x1e445b0),31);
 let mut data=[0u8;256];data[..8].copy_from_slice(b"TMPOST02");w32(&mut data,8,256);w32(&mut data,12,2);w32(&mut data,16,pid as u32);w32(&mut data,20,tid as u32);
 let hit=POST_HITS.fetch_add(1,Ordering::SeqCst)+1;if hit>50000{fail(32)}w64(&mut data,24,hit as usize);w64(&mut data,32,N_LROUNDF.load(Ordering::SeqCst)as usize);
 for(o,v)in[(40,controller),(48,sim),(56,context),(88,vector),(96,cb),(104,index),(216,*g.add(7)),(224,*g.add(17)),(232,bp),(240,VALIDATOR_PARTICIPANT_COUNT_ARG.load(Ordering::SeqCst))]{w64(&mut data,o,v)}
 for(o,v)in[(112,requested),(116,processed),(120,status),(124,flags),(128,sim_clock),(132,global),(136,mode),(140,special),(144,frame_time)]{w32(&mut data,o,v)}
 let mut validity=8u32;if processed==requested{validity|=4}if special==0&&mode==u32::MAX{validity|=16}
 // Early callbacks may precede vehicle allocation. Unavailable bytes are explicitly
 // masked and decoded as null, never exported as filled model observations.
 if let Some(playground)=word(sim+0x18){
  w64(&mut data,64,playground);w32(&mut data,248,if context==playground{1}else{0});if context==playground{validity|=32}
  if let(Some(1),Some(players))=(dword(playground+0x668),word(playground+0x660)){
   if let Some(participant)=word(players){w64(&mut data,72,participant);
    if let(Some(0x032e2000),Some(vehicle))=(dword(participant+0x1110),word(participant+0x1118)){
     let base=MODULE_BASE.load(Ordering::SeqCst);if word(participant)==Some(base+0x1c33938)&&vehicle>=4096{
      let mut physical=[0;40];let mut rpm=[0;8];let mut mirror=[0;12];
      if read(vehicle+0x12e0,&mut physical)&&read(vehicle+0x15d8,&mut rpm)&&read(vehicle+0x1608,&mut mirror){
       w64(&mut data,80,vehicle);data[152..192].copy_from_slice(&physical);data[192..200].copy_from_slice(&rpm);data[200..212].copy_from_slice(&mirror);validity|=1|64;
       let mut again=[0;40];let mut r2=[0;8];let mut m2=[0;12];
       if read(vehicle+0x12e0,&mut again)&&read(vehicle+0x15d8,&mut r2)&&read(vehicle+0x1608,&mut m2)&&same(&physical,&again)&&same(&rpm,&r2)&&same(&mirror,&m2){validity|=2}
      }
     }
    }
   }
  }
 }
 w32(&mut data,212,1);w32(&mut data,148,validity);w32(&mut data,252,if requested>10&&requested%10==0{1}else{0});
 let mut offset=0;while offset<data.len(){let n=sc(1,fd as usize,data[offset..].as_ptr()as usize,data.len()-offset,0,0,0);if n==-4{continue}if n<=0{fail(33)}offset+=n as usize;}
 // Exact semantics of MOV r15d,[rbx+e30]: zero-extend u32, flags/FP unchanged.
 *g.add(7)=processed as usize;*g.add(16)=at+7;DEPTH.store(0,Ordering::SeqCst);true
}

/// Reachability-only entries: emulate MOV RBP,RSP after the original PUSH.
unsafe fn entry_handle(info:*const u8,ctx:*mut c_void)->bool{
 if ctx.is_null()||POST_AT.load(Ordering::SeqCst)==0{return false}
 let base=MODULE_BASE.load(Ordering::SeqCst);let g=(ctx as*mut usize).add(5);let rip=*g.add(16);
 let Some((off,kind,sig))=ENTRIES.iter().find(|(off,_,_)|rip==base+*off+1).copied()else{return false};let at=base+off;
 if info.is_null(){fail(40)}let code=*(info.add(8)as*const i32);if code!=1&&code!=128{fail(41)}
 if DEPTH.fetch_add(1,Ordering::SeqCst)!=0{fail(42)}if *g.add(17)&0x10100!=0{fail(43)}
 let mut got=[0;20];if !read(at,&mut got)||got[0]!=0xcc||!same(&got[1..],&sig[1..]){fail(44)}
 let pid=sc(39,0,0,0,0,0,0)as usize;let tid=sc(186,0,0,0,0,0,0)as usize;let fd=process_log(pid,tid);
 let hit=POST_HITS.fetch_add(1,Ordering::SeqCst)+1;if hit>50000{fail(45)}
 let mut data=[0u8;256];data[..8].copy_from_slice(b"TMPOST02");w32(&mut data,8,256);w32(&mut data,12,2);w32(&mut data,16,pid as u32);w32(&mut data,20,tid as u32);
 w64(&mut data,24,hit as usize);w64(&mut data,32,N_LROUNDF.load(Ordering::SeqCst)as usize);w64(&mut data,40,VALIDATOR_CONTROLLER.load(Ordering::SeqCst));w64(&mut data,48,VALIDATOR_SIM.load(Ordering::SeqCst));w64(&mut data,56,*g.add(8));
 // Kind-specific raw arguments. Validity0 excludes physical-state interpretation.
 for(o,index)in[(152,8usize),(160,9),(168,12),(176,14),(184,0),(192,1)]{w64(&mut data,o,*g.add(index))}
 w32(&mut data,132,need(dword(base+0x1e445b0),46));w32(&mut data,212,kind);w64(&mut data,216,*g.add(15));w64(&mut data,224,*g.add(17));w64(&mut data,232,*g.add(10));
 let mut offset=0;while offset<data.len(){let n=sc(1,fd as usize,data[offset..].as_ptr()as usize,data.len()-offset,0,0,0);if n==-4{continue}if n<=0{fail(47)}offset+=n as usize;}
 *g.add(10)=*g.add(15);*g.add(16)=at+3;DEPTH.store(0,Ordering::SeqCst);true
}

/// Native inner-loop local completion point. Its relationship to physics is measured.
unsafe fn inner_handle(info:*const u8,ctx:*mut c_void)->bool{
 if ctx.is_null()||POST_AT.load(Ordering::SeqCst)==0{return false}
 let base=MODULE_BASE.load(Ordering::SeqCst);let at=base+INNER_OFF;let g=(ctx as*mut usize).add(5);if *g.add(16)!=at+1{return false}
 if info.is_null(){fail(50)}let code=*(info.add(8)as*const i32);if code!=1&&code!=128{fail(51)}
 if DEPTH.fetch_add(1,Ordering::SeqCst)!=0{fail(52)}if *g.add(17)&0x10100!=0{fail(53)}
 let mut got=[0;20];if !read(at,&mut got)||got[0]!=0xcc||!same(&got[1..],&INNER_SIG[1..]){fail(54)}
 let pid=sc(39,0,0,0,0,0,0)as usize;let tid=sc(186,0,0,0,0,0,0)as usize;let fd=process_log(pid,tid);
 let sim=*g.add(7);let bp=*g.add(10);let dt=*g.add(6)as u32;let completed=*g.add(11)as u32;
 let controller=VALIDATOR_CONTROLLER.load(Ordering::SeqCst);
 if sim<4096||sim!=VALIDATOR_SIM.load(Ordering::SeqCst)||need(word(controller+0x1a70),55)!=sim||bp<4096{fail(56)}
 let simclock=need(dword(sim+0x48),57);let current=need(dword(bp-0x30),58);let limit=need(dword(bp-0x40),59);let frame_dt=need(dword(bp-0x3c),60);
 if need(word(bp-0x48),61)!=sim||dt!=10||frame_dt!=dt||current!=completed||simclock!=completed{fail(62)}
 let mut data=[0u8;256];data[..8].copy_from_slice(b"TMPOST03");w32(&mut data,8,256);w32(&mut data,12,3);w32(&mut data,16,pid as u32);w32(&mut data,20,tid as u32);
 let hit=POST_HITS.fetch_add(1,Ordering::SeqCst)+1;if hit>50000{fail(63)}w64(&mut data,24,hit as usize);w64(&mut data,32,N_LROUNDF.load(Ordering::SeqCst)as usize);
 for(o,v)in[(40,controller),(48,sim),(56,sim),(216,*g.add(15)),(224,*g.add(17)),(232,bp)]{w64(&mut data,o,v)}
 for(o,v)in[(112,completed),(116,simclock),(120,limit),(132,need(dword(base+0x1e445b0),64)),(136,dt),(140,frame_dt),(144,current),(212,10)]{w32(&mut data,o,v)}
 let mut validity=8u32;
 if let Some(playground)=word(sim+0x18){w64(&mut data,64,playground);
  if let(Some(1),Some(players))=(dword(playground+0x668),word(playground+0x660)){
   if let Some(participant)=word(players){w64(&mut data,72,participant);
    if let(Some(0x032e2000),Some(vehicle))=(dword(participant+0x1110),word(participant+0x1118)){
     if vehicle>=4096&&word(participant)==Some(base+0x1c33938){
      let mut physical=[0;40];let mut rpm=[0;8];let mut mirror=[0;12];let mut inputs=[0;16];
      if read(vehicle+0x12e0,&mut physical)&&read(vehicle+0x15d8,&mut rpm)&&read(vehicle+0x1608,&mut mirror)&&read(vehicle+0x98,&mut inputs[..12])&&read(vehicle+0x1430,&mut inputs[12..]){
       w64(&mut data,80,vehicle);data[152..192].copy_from_slice(&physical);data[192..200].copy_from_slice(&rpm);data[200..212].copy_from_slice(&mirror);data[88..104].copy_from_slice(&inputs);validity|=1|64;
       let mut again=[0;40];let mut r2=[0;8];let mut m2=[0;12];let mut i2=[0;16];
       if read(vehicle+0x12e0,&mut again)&&read(vehicle+0x15d8,&mut r2)&&read(vehicle+0x1608,&mut m2)&&read(vehicle+0x98,&mut i2[..12])&&read(vehicle+0x1430,&mut i2[12..])&&same(&again,&physical)&&same(&r2,&rpm)&&same(&m2,&mirror)&&same(&i2,&inputs){validity|=2}
       // Read a G clock only for the exact owned6-segment observation layout.
       if SEG_N.load(Ordering::SeqCst)==6&&SEG_LEN[0].load(Ordering::SeqCst)==4&&SEG_LEN[1].load(Ordering::SeqCst)==40&&SEG_ADDR[1].load(Ordering::SeqCst)==vehicle+0x12e0&&SEG_ADDR[2].load(Ordering::SeqCst)==vehicle+0x15d8&&SEG_ADDR[3].load(Ordering::SeqCst)==vehicle+0x1608&&SEG_ADDR[4].load(Ordering::SeqCst)==vehicle+0x98&&SEG_ADDR[5].load(Ordering::SeqCst)==vehicle+0x1430{
        let address=SEG_ADDR[0].load(Ordering::SeqCst);if let Some(clock)=dword(address){w64(&mut data,104,address);w32(&mut data,128,clock);validity|=128}
       }
      }
     }
    }
   }
  }
 }
 w32(&mut data,148,validity);
 let mut offset=0;while offset<data.len(){let n=sc(1,fd as usize,data[offset..].as_ptr()as usize,data.len()-offset,0,0,0);if n==-4{continue}if n<=0{fail(65)}offset+=n as usize;}
 native_ipc::at_boundary(&data);
 // Original 89 d8: MOV EAX,EBX, zero-extends the low32, leaves flags/FP alone.
 *g.add(13)=completed as usize;*g.add(16)=at+2;DEPTH.store(0,Ordering::SeqCst);true
}

/// Called only by G before returning to native simulation, outside a signal.
pub unsafe fn prepare_action(){native_ipc::prepare()}
