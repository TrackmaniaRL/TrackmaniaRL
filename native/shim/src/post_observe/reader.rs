//! Read-only native decoded-record reader observer. No action or pause here.
use super::*;
const OFF:usize=0x119f163;
const SIG:[u8;20]=[0x89,0xf1,0x48,0xc1,0xe1,0x05,0x0f,0x10,0x04,0x0a,0x0f,0x10,0x4c,0x0a,0x10,0xeb,0xba,0x48,0x8d,0x15];
static AT:AtomicUsize=AtomicUsize::new(0);
pub unsafe fn configure(base:usize){
 if std::env::var("FKSHIM_READER").ok().as_deref()!=Some("1"){return}
 let mut b=[0;20];let mut prefix=[0;4];let mut call=[0;5];
 if !read(base+OFF,&mut b)||b!=SIG||!read(base+0x119f0f0,&mut prefix)||prefix!=[0x55,0x48,0x89,0xe5]||!read(base+0x1219f89,&mut call)||call!=[0xe8,0x62,0x51,0xf8,0xff]{fail(201)}
 AT.store(base+OFF,Ordering::SeqCst);
}
pub unsafe fn activate(){
 let at=AT.load(Ordering::SeqCst);if at==0{return}let mut b=[0;20];if !read(at,&mut b)||b!=SIG{fail(202)}
 let page=at&!4095usize;if sc(10,page,4096,7,0,0,0)!=0{fail(203)}core::ptr::write_volatile(at as*mut u8,0xcc);if sc(10,page,4096,5,0,0,0)!=0{fail(204)}
}
pub unsafe fn handle(info:*const u8,ctx:*mut c_void)->bool{
 let at=AT.load(Ordering::SeqCst);if at==0||ctx.is_null(){return false}let g=(ctx as*mut usize).add(5);if *g.add(16)!=at+1{return false}
 if info.is_null(){fail(205)}let code=*(info.add(8)as*const i32);if code!=1&&code!=128{fail(206)}
 if DEPTH.fetch_add(1,Ordering::SeqCst)!=0{fail(207)}if *g.add(17)&0x10100!=0{fail(208)}
 let mut signature=[0;20];if !read(at,&mut signature)||signature[0]!=0xcc||!same(&signature[1..],&SIG[1..]){fail(209)}
 let pid=sc(39,0,0,0,0,0,0)as usize;let tid=sc(186,0,0,0,0,0,0)as usize;let fd=process_log(pid,tid);
 let hit=POST_HITS.fetch_add(1,Ordering::SeqCst)+1;if hit>50000{fail(210)}
 let base=MODULE_BASE.load(Ordering::SeqCst);let bp=*g.add(10);let caller=word(bp).unwrap_or(0);let ret=word(bp.wrapping_add(8)).unwrap_or(0);
 let index=*g.add(9)as u32;let count=*g.add(14)as u32;let header=*g.add(12);let cached=CACHED_BASE.load(Ordering::SeqCst);let expected_n=EXPECT_N.load(Ordering::SeqCst);
 let controller=VALIDATOR_CONTROLLER.load(Ordering::SeqCst);let sim=VALIDATOR_SIM.load(Ordering::SeqCst);let participant=*g.add(5);let context=*g.add(6);
 let mut data=[0u8;256];data[..8].copy_from_slice(b"TMREAD01");w32(&mut data,8,256);w32(&mut data,12,1);w32(&mut data,16,pid as u32);w32(&mut data,20,tid as u32);
 w64(&mut data,24,hit as usize);w64(&mut data,32,N_LROUNDF.load(Ordering::SeqCst)as usize);
 for(o,v)in[(40,controller),(48,sim),(56,bp),(64,caller),(72,ret),(80,header),(88,cached),(104,expected_n),(192,participant),(200,context),(216,*g.add(15)),(224,*g.add(17)),(232,*g.add(13))]{w64(&mut data,o,v)}
 for(o,v)in[(112,index),(116,count),(120,dword(caller.wrapping_sub(0x30)).unwrap_or(0)),(124,dword(sim.wrapping_add(0x48)).unwrap_or(0)),(132,dword(base+0x1e445b0).unwrap_or(0)),(140,SEG_N.load(Ordering::SeqCst)as u32),(144,*g.add(0)as u32),(188,dword(context.wrapping_add(0x20e0)).unwrap_or(0)),(212,11),(248,dword(context.wrapping_add(0x20e8)).unwrap_or(0)),(252,*g.add(8)as u32)]{w32(&mut data,o,v)}
 let mut mask=0u32;
 if ret==base+0x1219f8e&&caller>=4096&&caller.checked_sub(bp)==Some(0x1a0)&&*g.add(15)==bp&&word(bp+0x10)==Some(*g.add(13))&&dword(caller-0x30)==Some(*g.add(11)as u32)&&*g.add(1)==0{mask|=1}
 let record=header.checked_add(index as usize*32);
 if let Some(address)=record{
  let mut raw=[0;32];if index<count&&count<=200000&&read(address,&mut raw){data[152..184].copy_from_slice(&raw);w32(&mut data,128,u32::from_le_bytes(raw[..4].try_into().unwrap()));mask|=128;
   if let Some(channels)=address.checked_add(4){w64(&mut data,96,channels);}
   let shadow=EXPECT.load(Ordering::SeqCst);
   if (index as usize)<expected_n&&shadow>=4096{let mut expected=[0;4];if read(shadow+index as usize*4,&mut expected){data[184..188].copy_from_slice(&expected);if same(&expected,&raw[4..8]){mask|=32}}}
  }
 }
 if cached>=4096&&header.checked_add(4)==Some(cached)&&index<count{mask|=2}
 if sim>=4096&&controller>=4096&&word(controller+0x1a70)==Some(sim)&&word(caller.wrapping_sub(0x48))==Some(sim){mask|=4}
 let mut vehicle=0usize;
 if let Some(pg)=word(sim.wrapping_add(0x18)){
  if let(Some(1),Some(players))=(dword(pg.wrapping_add(0x668)),word(pg.wrapping_add(0x660))){
   if word(players)==Some(participant)&&word(participant)==Some(base+0x1c33938)&&dword(participant.wrapping_add(0x1110))==Some(0x032e2000){
    vehicle=word(participant+0x1118).unwrap_or(0);if vehicle>=4096&&*g.add(13)==participant+0x188{mask|=8}
   }
  }
 }
 if let Some(manager)=word(caller.wrapping_sub(0x50)){
  let mut small=[0;1];if read(participant.wrapping_add(0x1d8),&mut small)&&word(manager.wrapping_add(0x28+small[0]as usize*8))==Some(context)&&word(context.wrapping_add(0x20d8))==Some(header)&&dword(context.wrapping_add(0x20e0))==Some(count)&&dword(caller.wrapping_sub(0x88))==Some(count)&&dword(context.wrapping_add(0x20e8)).map(|v|v.wrapping_add(10))==Some(*g.add(0)as u32){mask|=16}
 }
 w64(&mut data,240,word(context.wrapping_add(0x20d8)).unwrap_or(0));
 if expected_n>0&&expected_n==count as usize{mask|=256}
 if SEG_N.load(Ordering::SeqCst)==6&&vehicle>=4096&&SEG_LEN[0].load(Ordering::SeqCst)==4&&SEG_LEN[1].load(Ordering::SeqCst)==40&&SEG_ADDR[1].load(Ordering::SeqCst)==vehicle+0x12e0{mask|=64}
 let role=if mask&0x1bf==0x1bf{1}else{0};w32(&mut data,208,role);w32(&mut data,148,mask);
 let mut offset=0;while offset<data.len(){let n=sc(1,fd as usize,data[offset..].as_ptr()as usize,data.len()-offset,0,0,0);if n==-4{continue}if n<=0{fail(211)}offset+=n as usize;}
 native_ipc::at_reader(&data);
 // Exact original 89 f1. Saved flags and FP/SIMD state are untouched.
 *g.add(14)=index as usize;*g.add(16)=at+2;DEPTH.store(0,Ordering::SeqCst);true
}
