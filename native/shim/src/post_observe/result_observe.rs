//! Read-only result collector return observer, Linux x86-64 and pinned ELF only.
//! TMRESULT records have a separate FD/counter; existing POST256 and V4 are unchanged.
//! Missing groups are masked. No manager==capturedSIM or physical finish assumption.
use super::*;
const OFF:usize=0x11834f2;
const SIG:[u8;20]=[0x41,0x8b,0x85,0xdc,0x01,0,0,0x42,0x89,0x44,0x33,0x14,0x49,0x8b,0x07,0x48,0x8b,0x78,0x28,0x4c];
const CALL:[u8;5]=[0xe8,0x4e,0x2d,0x09,0];
const BYTES:usize=768;
const CP_CAP:usize=64; // Probe cap, not an inferred game limit.
const VECTOR_CAP:usize=4096; // Bounds only this observer's accepted provenance domain.
static AT:AtomicUsize=AtomicUsize::new(0);
static LOG_PID:AtomicUsize=AtomicUsize::new(0);
static LOG_TID:AtomicUsize=AtomicUsize::new(0);
static LOG_FD:AtomicI32=AtomicI32::new(-1);
static HITS:AtomicU64=AtomicU64::new(0);
const CODE:u32=1;
const FRAME:u32=2;
const VECTOR:u32=4;
const CONTEXT:u32=8;
const OWNER:u32=16;
const MANAGER:u32=32;
const OUT:u32=64;
const OUT_STABLE:u32=128;
const CP_BOUNDS:u32=256;
const CP_READ:u32=512;
const CP_STABLE:u32=1024;
const SIM_CLOCK:u32=2048;
const GLOBAL_CLOCK:u32=4096;
const SHAPE_STABLE:u32=16384;
const CORE:u32=CODE|FRAME|VECTOR|CONTEXT|OWNER|MANAGER;
const COMPLETE:u32=CORE|OUT|OUT_STABLE|CP_BOUNDS|CP_READ|CP_STABLE|SHAPE_STABLE;

fn pointer(a:usize)->bool{a>=4096&&a<0x0000_8000_0000_0000&&a&7==0}
unsafe fn pw(a:usize,o:usize)->Option<usize>{if !pointer(a){return None}word(a.checked_add(o)?)}
unsafe fn pd(a:usize,o:usize)->Option<u32>{if !pointer(a){return None}dword(a.checked_add(o)?)}
unsafe fn clear(p:*mut u8,n:usize){let mut i=0;while i<n{core::ptr::write_volatile(p.add(i),0);i+=1}}
fn u32at(b:&[u8],o:usize)->u32{u32::from_le_bytes(b[o..o+4].try_into().unwrap())}
fn u64at(b:&[u8],o:usize)->usize{u64::from_le_bytes(b[o..o+8].try_into().unwrap())as usize}

/// Outside signal handlers, after the base POST directory has been validated.
pub unsafe fn configure(base:usize){
 if std::env::var("FKSHIM_RESULT").ok().as_deref()!=Some("1"){return}
 if base==0||DIR_LEN.load(Ordering::SeqCst)==0{fail(401)}
 let at=need(base.checked_add(OFF),402);let mut b=[0;20];let mut call=[0;5];
 if !read(at,&mut b)||!same(&b,&SIG)||!read(at-5,&mut call)||!same(&call,&CALL){fail(403)}
 AT.store(at,Ordering::SeqCst);
}
/// Integrated with the existing binding owner; no independent sigaction restorer.
pub unsafe fn activate(){
 let at=AT.load(Ordering::SeqCst);if at==0{return}let mut b=[0;20];let mut call=[0;5];
 if !read(at,&mut b)||!same(&b,&SIG)||!read(at-5,&mut call)||!same(&call,&CALL){fail(404)}
 let page=at&!4095usize;if sc(10,page,4096,7,0,0,0)!=0{fail(405)}
 core::ptr::write_volatile(at as*mut u8,0xcc);if sc(10,page,4096,5,0,0,0)!=0{fail(406)}
}
unsafe fn result_log(pid:usize,tid:usize)->i32{
 if LOG_PID.load(Ordering::SeqCst)!=pid{
  let old=LOG_FD.swap(-1,Ordering::SeqCst);if old>=0{sc(3,old as usize,0,0,0,0,0);}
  let n=DIR_LEN.load(Ordering::SeqCst);if n==0||n>240{fail(407)}
  let mut storage=core::mem::MaybeUninit::<[u8;288]>::uninit();clear(storage.as_mut_ptr()as*mut u8,288);let path=storage.assume_init_mut();
  copy_volatile(&(&*core::ptr::addr_of!(DIR))[..n],&mut path[..n]);copy_volatile(b"/result-",&mut path[n..n+8]);
  let mut digits=[0u8;20];let mut x=pid;let mut len=0;
  loop{if len==20{fail(408)}digits[len]=b'0'+(x%10)as u8;len+=1;x/=10;if x==0{break}}
  let mut i=0;while i<len{path[n+8+i]=digits[len-1-i];i+=1}copy_volatile(b".bin",&mut path[n+8+len..n+12+len]);
  let fd=sc(257,(-100isize)as usize,path.as_ptr()as usize,0x80000|0x80|0x40|1,0o600,0,0);if fd<0{fail(409)}
  HITS.store(0,Ordering::SeqCst);LOG_TID.store(tid,Ordering::SeqCst);LOG_FD.store(fd as i32,Ordering::SeqCst);LOG_PID.store(pid,Ordering::SeqCst);
 }
 if LOG_TID.load(Ordering::SeqCst)!=tid{fail(410)}LOG_FD.load(Ordering::SeqCst)
}

pub unsafe fn handle(info:*const u8,ctx:*mut c_void)->bool{
 let at=AT.load(Ordering::SeqCst);if at==0||ctx.is_null(){return false}
 let g=(ctx as*mut usize).add(5);if *g.add(16)!=at+1{return false}
 if info.is_null(){fail(411)}let code=*(info.add(8)as*const i32);if code!=1&&code!=128{fail(412)}
 if DEPTH.fetch_add(1,Ordering::SeqCst)!=0{fail(413)}if *g.add(17)&0x10100!=0{fail(414)}
 let mut b=[0;20];let mut call=[0;5];if !read(at,&mut b)||b[0]!=0xcc||!same(&b[1..],&SIG[1..])||!read(at-5,&mut call)||!same(&call,&CALL){fail(415)}
 let context=*g.add(5); // R13; original instruction's source, not a guessed participant.
 let loaded=need(pd(context,0x1dc),416); // One exact read supplies logged value AND MOV emulation.
 let pid=sc(39,0,0,0,0,0,0)as usize;let tid=sc(186,0,0,0,0,0,0)as usize;let fd=result_log(pid,tid);
 let hit=HITS.fetch_add(1,Ordering::SeqCst)+1;if hit>50000{fail(417)}
 // Never a by-value 768B array, memset, heap allocation or returned large struct.
 let mut storage=core::mem::MaybeUninit::<[u8;BYTES]>::uninit();clear(storage.as_mut_ptr()as*mut u8,BYTES);let d=storage.assume_init_mut();
 copy_volatile(b"TMRESULT",&mut d[..8]);w32(d,8,BYTES as u32);w32(d,12,1);w32(d,16,pid as u32);w32(d,20,tid as u32);w64(d,24,hit as usize);w64(d,32,N_LROUNDF.load(Ordering::SeqCst)as usize);
 let controller=VALIDATOR_CONTROLLER.load(Ordering::SeqCst);let sim=VALIDATOR_SIM.load(Ordering::SeqCst);let bp=*g.add(10);let stateptr=*g.add(7);let vector=*g.add(11);let index=*g.add(6);
 for(o,v)in[(40,controller),(48,sim),(64,stateptr),(80,context),(96,bp),(104,vector),(112,index),(208,*g.add(13)),(216,*g.add(15)),(224,*g.add(17)),(232,*g.add(16))]{w64(d,o,v)}
 w32(d,196,CP_CAP as u32);w32(d,200,loaded);w32(d,240,13);
 let mut mask=CODE;
 let state=pw(stateptr,0).filter(|v|pointer(*v));
 if let Some(state)=state{
  w64(d,56,state);
  if pointer(bp)&&pw(bp.checked_sub(0x38).unwrap_or(0),0)==Some(stateptr)&&pw(bp.checked_sub(0x48).unwrap_or(0),0)==vector.checked_add(8){mask|=FRAME}
  if pointer(controller)&&pointer(sim)&&pw(bp.checked_sub(0x60).unwrap_or(0),0)==Some(controller)&&pw(controller,0x1a70)==Some(sim){mask|=OWNER}
  if let Some(manager)=pw(state,0x28).filter(|v|pointer(*v)){w64(d,72,manager);mask|=MANAGER;if mask&OWNER!=0{w32(d,244,if manager==sim{1}else{0})}}
  if let Some(count)=pd(state,0x60){w32(d,144,count);
   if count>0&&count as usize<=VECTOR_CAP&&pointer(vector)&&pw(state,0x58)==Some(vector)&&index%72==0&&index/72<count as usize{
    if let Some(entry)=vector.checked_add(index){if let Some(out)=entry.checked_add(8){w64(d,88,out);mask|=VECTOR;let mut context_table=0usize;let mut context_count_before=0u32;
     if let(Some(context_index),Some(context_count),Some(contexts))=(pd(entry,0),pd(state,0x50),pw(state,0x48)){
      w32(d,136,context_index);
      if context_count>0&&context_count as usize<=VECTOR_CAP&&context_index<context_count&&pointer(contexts){
       if let Some(slot)=contexts.checked_add(context_index as usize*8){if pw(slot,0)==Some(context){if let Some(selector)=pd(context,0x368){w32(d,140,selector);if selector!=u32::MAX{mask|=CONTEXT;context_table=contexts;context_count_before=context_count}}}}
      }
     }
     if mask&CORE==CORE&&read(out,&mut d[152..184]){
      mask|=OUT;let pairs=u64at(d,168);let count=u32at(d,176);let capacity=u32at(d,180);w64(d,120,pairs);w32(d,184,count);w32(d,188,capacity);
      if count as usize<=CP_CAP&&count<=capacity&&(count==0||(pointer(pairs)&&pairs.checked_add(count as usize*8).is_some())){
       mask|=CP_BOUNDS;let length=count as usize*8;
       if length==0||read(pairs,&mut d[256..256+length]){
        mask|=CP_READ;w32(d,192,length as u32);let mut stable=true;let mut j=0;
        while j<count as usize{let mut pair=[0;8];if !read(pairs+j*8,&mut pair)||!same(&pair,&d[256+j*8..264+j*8]){stable=false;break}j+=1}
        if stable{mask|=CP_STABLE}
       }
      }
      let mut again=[0;32];if read(out,&mut again)&&same(&again,&d[152..184]){mask|=OUT_STABLE}else{mask&=!CP_STABLE}
      if pw(stateptr,0)==Some(state)&&pw(state,0x28)==Some(u64at(d,72))&&pw(state,0x58)==Some(vector)&&pd(state,0x60)==Some(u32at(d,144))&&pd(entry,0)==Some(u32at(d,136))&&pw(state,0x48)==Some(context_table)&&pd(state,0x50)==Some(context_count_before)&&context_table.checked_add(u32at(d,136)as usize*8).and_then(|a|pw(a,0))==Some(context)&&pw(bp.checked_sub(0x38).unwrap_or(0),0)==Some(stateptr)&&pw(bp.checked_sub(0x48).unwrap_or(0),0)==vector.checked_add(8)&&pw(bp.checked_sub(0x60).unwrap_or(0),0)==Some(controller)&&pw(controller,0x1a70)==Some(sim)&&pd(context,0x368)==Some(u32at(d,140)){mask|=SHAPE_STABLE}
     }
    }}
   }
  }
 }
 if mask&OWNER!=0{if let Some(clock)=pd(sim,0x48){w32(d,128,clock);mask|=SIM_CLOCK}}
 if let Some(address)=MODULE_BASE.load(Ordering::SeqCst).checked_add(0x1e445b0){if let Some(clock)=dword(address){w32(d,132,clock);mask|=GLOBAL_CLOCK}}
 let mut missing=0u32;for(bit,why)in[(FRAME,1),(VECTOR,2),(CONTEXT,4),(OWNER,8),(MANAGER,16),(OUT,32),(CP_BOUNDS,64),(CP_READ,128),(OUT_STABLE,256),(CP_STABLE,512),(SIM_CLOCK,1024),(GLOBAL_CLOCK,2048),(SHAPE_STABLE,4096)]{if mask&bit==0{missing|=why}}
 w32(d,148,mask);w32(d,204,missing);w32(d,248,if mask&COMPLETE==COMPLETE{1}else{0});
 let mut off=0;while off<BYTES{let n=sc(1,fd as usize,d[off..].as_ptr()as usize,BYTES-off,0,0,0);if n==-4{continue}if n<=0{fail(418)}off+=n as usize}
 // Exact 41 8b 85 dc 01 00 00: MOV EAX,[R13+1dc], zero-extend; no flag/FP/stack writes.
 *g.add(13)=loaded as usize;*g.add(16)=at+7;DEPTH.store(0,Ordering::SeqCst);true
}
