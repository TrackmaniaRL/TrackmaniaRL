//! Read-only binding of the active waypoint contact registry at its native consumer.
use super::*;
include!("registry_signature.rs");
static BASE:AtomicUsize=AtomicUsize::new(0);
static PID:AtomicUsize=AtomicUsize::new(0);
static TID:AtomicUsize=AtomicUsize::new(0);
static FD:AtomicI32=AtomicI32::new(-1);
static SEQ:AtomicUsize=AtomicUsize::new(0);
pub unsafe fn configure(base:usize){let mut b=[0u8;20];if !read(base+REGISTRY_RVA,&mut b)||!same(&b,&REGISTRY_SIGNATURE){fail(910)}BASE.store(base,Ordering::SeqCst);}
pub unsafe fn activate(){let base=BASE.load(Ordering::SeqCst);if base==0{return}let at=base+REGISTRY_RVA;let mut b=[0u8;20];if !read(at,&mut b)||!same(&b,&REGISTRY_SIGNATURE){fail(911)}let page=at&!4095usize;if sc(10,page,4096,7,0,0,0)!=0{fail(912)}core::ptr::write_volatile(at as*mut u8,0xcc);if sc(10,page,4096,5,0,0,0)!=0{fail(913)}}
unsafe fn output(pid:usize,tid:usize)->i32{
 if PID.load(Ordering::SeqCst)!=pid{
  let old=FD.swap(-1,Ordering::SeqCst);if old>=0{sc(3,old as usize,0,0,0,0,0);}
  let n=DIR_LEN.load(Ordering::SeqCst);if n==0||n>240{fail(914)}let mut path=[0u8;320];copy_volatile(&(&*core::ptr::addr_of!(DIR))[..n],&mut path[..n]);copy_volatile(b"/registry-",&mut path[n..n+10]);let mut digits=[0u8;24];let mut x=pid;let mut count=0;loop{digits[count]=b'0'+(x%10)as u8;count+=1;x/=10;if x==0{break}}for i in 0..count{path[n+10+i]=digits[count-i-1];}copy_volatile(b".bin",&mut path[n+10+count..n+14+count]);let fd=sc(257,(-100isize)as usize,path.as_ptr()as usize,0x800c1,0o600,0,0);if fd<0{fail(915)}PID.store(pid,Ordering::SeqCst);TID.store(tid,Ordering::SeqCst);FD.store(fd as i32,Ordering::SeqCst);SEQ.store(0,Ordering::SeqCst);
 }if TID.load(Ordering::SeqCst)!=tid{fail(916)}FD.load(Ordering::SeqCst)
}
pub unsafe fn handle(info:*const u8,ctx:*mut c_void)->bool{
 let base=BASE.load(Ordering::SeqCst);if base==0||ctx.is_null(){return false}let g=(ctx as*mut usize).add(5);if *g.add(16)!=base+REGISTRY_RVA+1{return false}
 if info.is_null(){fail(917)}let code=*(info.add(8)as*const i32);if code!=1&&code!=128{fail(917)}if DEPTH.fetch_add(1,Ordering::SeqCst)!=0||*g.add(17)&0x10100!=0{fail(918)}let at=base+REGISTRY_RVA;let mut b=[0u8;20];if !read(at,&mut b)||b[0]!=0xcc||!same(&b[1..],&REGISTRY_SIGNATURE[1..]){fail(919)}
 let sim=VALIDATOR_SIM.load(Ordering::SeqCst);let manager=*g.add(8);
 if let Some(clock)=dword(sim.wrapping_add(0x48)){
  if clock>=2190&&clock<=162200&&clock%10==0{
   let pid=sc(39,0,0,0,0,0,0)as usize;let tid=sc(186,0,0,0,0,0,0)as usize;let fd=output(pid,tid);let sequence=SEQ.fetch_add(1,Ordering::SeqCst)+1;
   let mut record=[0u8;128];copy_volatile(b"TMCR0001",&mut record[..8]);w32(&mut record,8,128);w32(&mut record,12,1);w32(&mut record,16,pid as u32);w32(&mut record,20,tid as u32);w64(&mut record,24,sequence as usize);w32(&mut record,32,clock);w32(&mut record,36,REGISTRY_RVA as u32);w64(&mut record,40,sim as usize);w64(&mut record,48,manager as usize);
   if !read(manager.wrapping_add(0xa8),&mut record[56..88]){fail(920)}w64(&mut record,88,base as usize);w64(&mut record,96,*g.add(15)as usize);w64(&mut record,104,at as usize);
   if sc(1,fd as usize,record.as_ptr()as usize,record.len(),0,0,0)!=record.len()as isize{fail(921)}
  }
 }
 // Exact LEA RAX,[RDI+0xa8]; no memory write or flag change.
 *g.add(13)=manager.wrapping_add(0xa8);*g.add(16)=at+7;DEPTH.fetch_sub(1,Ordering::SeqCst);true
}
