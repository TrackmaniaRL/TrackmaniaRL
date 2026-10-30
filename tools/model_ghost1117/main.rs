//! Offline export of recorded model poses and applied controls, no engine run.
use gbx::container::{Container,write_gbx};
use gbx::record::{CLASS_CSCENEVEHICLEVIS,neutralise,NEUTRALISE};
use gbx::recwrite::{rewrite_ghost,write_transform,Xform};
use gbx::tape::{Tape,Archive,Packet,StateEnc,Encoding,literal_for};
use serde_json::{Value,json};
use std::path::Path;

fn numbers(v:&Value,key:&str)->Vec<f64>{v[key].as_array().unwrap().iter().map(|x|x.as_f64().unwrap()).collect()}
struct ValidationReader<'a>{bytes:&'a [u8],at:usize}
impl<'a> ValidationReader<'a>{
 fn take(&mut self,n:usize)->Result<&'a [u8],String>{
  let end=self.at.checked_add(n).ok_or("metadata size overflow")?;
  let out=self.bytes.get(self.at..end).ok_or("truncated validation metadata")?;
  self.at=end;Ok(out)
 }
 fn u32(&mut self)->Result<u32,String>{Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))}
 fn string(&mut self)->Result<String,String>{
  let n=self.u32()? as usize;
  if n>1024*1024{return Err("metadata string limit".into())}
  String::from_utf8(self.take(n)?.to_vec()).map_err(|e|e.to_string())
 }
}
fn wrap(a:&[String])->Result<(),String>{
 if a.len()!=5{return Err("wrap MODEL.Ghost.Gbx NATIVE_TEMPLATE.Replay.Gbx FRESH.Replay.Gbx".into())}
 if Path::new(&a[4]).exists(){return Err("refuse output overwrite".into())}
 let model=Container::load(&a[2])?;let native=Container::load(&a[3])?;
 if native.gbx.class_id!=0x03093000 || !native.uids().iter().any(|(_,s)|s=="oqIJ5rQDRrNwLPTh9H2p_W4tLof"){
  return Err("wrong native replay container/map".into())
 }
 let (_,start,end,index,old_at)=ghost::unwrap::unwrap_body(native.body())?;
 let mut body=model.body().to_vec();let pattern=[0,240,17,9,0,240,17,9];
 let hits:Vec<usize>=body.windows(8).enumerate().filter(|(_,w)|*w==pattern).map(|(i,_)|i).collect();
 println!("native ghost index={index}, declared nodes={}",native.gbx.num_nodes);
 if hits.len()!=1 || index==0 || native.gbx.num_nodes<3 || native.gbx.num_nodes>64{return Err("unsupported node topology".into())}
 let at=hits[0]-4;body[at..at+4].copy_from_slice(&native.body()[start+old_at..start+old_at+4]);
 let mut replay=native.body()[..start].to_vec();replay.extend_from_slice(&body);replay.extend_from_slice(&native.body()[end..]);
 ghost::trim::set_all_declared(&mut replay,17712);
 let (back,_,_,_,_)=ghost::unwrap::unwrap_body(&replay)?;
 if back!=model.body(){return Err("wrapped model ghost changed beyond node rebasing".into())}
 let mut gbx=native.gbx.clone();
 if let Some(header)=ghost::hdr::rewrite(&native,true,Some("TrackmaniaRL Model (headless)"),Some(17712)){
  gbx.user_data=header.user_data;
 }
 write_gbx(&gbx,replay,&a[4])?;
 let out=Container::load(&a[4])?;
 if ghost::unwrap::unwrap_body(out.body())?.0!=model.body(){return Err("replay disk readback changed model".into())}
 println!("PASS_REPLAY_WRAP_EXACT_MODEL_NODE {}",a[4]);Ok(())
}
fn main()->Result<(),String>{
 let a:Vec<String>=std::env::args().collect();
 if a.get(1).map(|s|s.as_str())==Some("diagnostic-claim-control"){
  if a.len()!=4 || Path::new(&a[3]).exists(){return Err("diagnostic-claim-control VALID_NATIVE_GHOST FRESH_NEGATIVE_CONTROL (not a record)".into())}
  let c=Container::load(&a[2])?;
  if c.gbx.class_id!=0x03092000 || c.result().ok_or("native result")?.race_ms!=17712{
   return Err("this known native17.712 diagnostic source only".into())
  }
  let before=Tape::from_file(&a[2])?;
  let mut b=c.body().to_vec();ghost::trim::set_all_declared(&mut b,17713);
  write_gbx(&c.gbx,b,&a[3])?;
  let after=Tape::from_file(&a[3])?;
  if format!("{before:?}")!=format!("{after:?}"){return Err("diagnostic changed actual input tape".into())}
  println!("NEGATIVE_CONTROL_ONLY: deliberately incorrect declared17.713; actual inputs unchanged; never a claimed result");return Ok(())
 }
 if a.get(1).map(|s|s.as_str())==Some("unwrap-native"){
  if a.len()!=4 || Path::new(&a[3]).exists(){return Err("unwrap-native NATIVE.Replay.Gbx FRESH.Ghost.Gbx".into())}
  ghost::unwrap::cmd(&a[2..4]);return Ok(())
 }
 if a.get(1).map(|s|s.as_str())==Some("inspect-native"){
  if a.len()!=3{return Err("inspect-native SAVED_FILE (read-only)".into())}
  let c=Container::load(&a[2])?;
  let r=c.result().ok_or("native result missing")?;
  let t=Tape::from_file(&a[2])?;
  println!("{}",json!({"status":"READ_ONLY_NATIVE_FILE","class_id":c.gbx.class_id,
   "race_ms":r.race_ms,"checkpoint_count":r.entries.len(),
   "declared_times":c.declared_times(),"map_uids":c.uids(),
   "archives":t.archives.iter().map(|x|json!({"start_offset_ms":x.start_offset_ms,
    "packets":x.packets.len()})).collect::<Vec<_>>()}));return Ok(())
 }
 if a.get(1).map(|s|s.as_str())==Some("inspect-validation"){
  if a.len()!=3{return Err("inspect-validation GHOST (read-only)".into())}
  let c=Container::load(&a[2])?;
  let chunks:Vec<_>=c.chunks().into_iter().filter(|x|x.0==0x0309202d).collect();
  if chunks.len()!=1{return Err("expected one validation metadata chunk".into())}
  let (_,_,at,size)=chunks[0];
  let bytes=c.body().get(at..at+size).ok_or("validation chunk range")?;
  let mut r=ValidationReader{bytes,at:0};
  let version=r.u32()?;
  if version!=0{return Err(format!("unsupported validation metadata layout {version}"))}
  let exe=r.string()?;let checksum=r.u32()?;let os=r.u32()?;let cpu=r.u32()?;
  let start=r.u32()?;let end=r.u32()?;let title=r.string()?;
  r.take(32)?;
  let u02=r.u32()?;let u03=r.u32()?;let seed=r.u32()?;let u04=r.u32()?;
  let settings=r.string()?;
  if r.at!=bytes.len(){return Err("unparsed validation metadata bytes".into())}
  println!("{}",json!({"status":"READ_ONLY_VALIDATION_METADATA","exe_version":exe,
   "exe_checksum":checksum,"os":os,"cpu":cpu,"wall_start":start,"wall_end":end,
   "title":title,"u02":u02,"u03":u03,"validation_seed":seed,"u04":u04,
   "race_settings_length":settings.len(),"metadata_bytes":size}));return Ok(())
 }
 if a.get(1).map(|s|s.as_str())==Some("observed-clock-control"){
  if a.len()!=5 || Path::new(&a[4]).exists(){return Err("observed-clock-control VERIFIED_JSON COUNTDOWN_BOUND_PREVIEW FRESH_OUTPUT".into())}
  let data:Value=serde_json::from_slice(&std::fs::read(&a[2]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
  let frames=data["frames"].as_array().ok_or("frames")?;
  if frames.len()!=1776 || frames[0]["t"]!=0 || data["goal_ms"]!=17712{return Err("bound observed source".into())}
  let own=Container::load(&a[3])?;let mut tape=Tape::from_file(&a[3])?;
  if tape.archives.len()!=1 || tape.archives[0].start_offset_ms!=-1510{return Err("bound countdown required".into())}
  let mut packets=tape.archives[0].packets[..151].to_vec();
  for f in &frames[..1773]{
   let s=numbers(f,"sgb");
   if ![-1.,0.,1.].contains(&s[0]) || ![0.,1.].contains(&s[1]) || ![0.,1.].contains(&s[2]){return Err("exact digital input required".into())}
   packets.push(Packet{word0:2,flags:0,mode:2,state:StateEnc::Lit(literal_for(2,0)),mouse:None,vsame:false,
    steer:((s[0]*127.) as i8) as u8 as u32,accel:s[1] as u32,brake:s[2] as u32,tri:None});
  }
  tape.archives[0].packets=packets;tape.archives[0].tail.clear();
  write_gbx(&own.gbx,tape.splice_into(own.body(),Encoding::Explicit)?,&a[4])?;
  println!("OBSERVED_CLOCK_HYPOTHESIS_REQUIRES_PLAIN_VALIDATOR {}",a[4]);return Ok(())
 }
 if a.get(1).map(|s|s.as_str())==Some("tape-control"){
  if a.len()!=5 || Path::new(&a[4]).exists(){return Err("tape-control TAPE_SOURCE BODY_CONTROL FRESH_OUTPUT (diagnostic only)".into())}
  let tape=Tape::from_file(&a[2])?;let body=Container::load(&a[3])?;
  write_gbx(&body.gbx,tape.splice_into(body.body(),Encoding::Explicit)?,&a[4])?;
  println!("DIAGNOSTIC_ONLY_NOT_MODEL_VISUAL_REPLAY {}",a[4]);return Ok(())
 }
 if a.get(1).map(|s|s.as_str())==Some("format-controls"){
  if a.len()!=4 || Path::new(&a[3]).exists(){return Err("format-controls HUMAN_CONTROL FRESH_DIRECTORY".into())}
  let root=Path::new(&a[3]);std::fs::create_dir(root).map_err(|e|e.to_string())?;
  let c=Container::load(&a[2])?;
  write_gbx(&c.gbx,c.body().to_vec(),root.join("container-roundtrip.Ghost.Gbx").to_str().unwrap())?;
  let t=Tape::from_file(&a[2])?;
  write_gbx(&c.gbx,t.splice_into(c.body(),Encoding::Explicit)?,root.join("explicit-tape.Ghost.Gbx").to_str().unwrap())?;
  ghost::ident::cmd(&vec!["set".into(),a[2].clone(),root.join("identity-only.Ghost.Gbx").to_string_lossy().into(),"--anonymise".into(),"--no-oracle".into(),"--name".into(),"FORMAT CONTROL ONLY".into()]);
  return Ok(())
 }
 if a.get(1).map(|s|s.as_str())==Some("bind-countdown"){
  if a.len()!=5 || Path::new(&a[4]).exists(){return Err("bind-countdown MODEL_PREVIEW COUNTDOWN_FORMAT FRESH_OUTPUT".into())}
  let model=Container::load(&a[2])?;
  let native=Tape::from_file(&a[3])?;let own=Tape::from_file(&a[2])?;
  if native.archives.len()!=1 || own.archives.len()!=1{return Err("one archive required".into())}
  let original=&native.archives[0];let mine=&own.archives[0];
  if original.start_offset_ms!=-1510 || mine.start_offset_ms!=0 || mine.packets.len()!=1772{return Err("known pre-start clock required".into())}
  let count=151usize;
  if original.packets[0].word0!=15 || original.packets[0].mode!=15{return Err("native initialization marker required".into())}
  for p in &original.packets[1..count]{
   if p.word0!=2 || p.steer_i8()!=0 || p.accel!=1 || p.brake!=0{return Err("only straight frozen-countdown scaffold permitted".into())}
  }
  let mut archive=original.clone();
  archive.packets=original.packets[..count].to_vec();
  archive.packets.extend_from_slice(&mine.packets);
  archive.tail.clear();archive.orig_bitstream.clear();archive.orig_bitstream_len=0;archive.orig_bits_used=0;
  let new=Tape{chunk_version:own.chunk_version,archives:vec![archive]};
  write_gbx(&model.gbx,new.splice_into(model.body(),Encoding::Explicit)?,&a[4])?;
  let back=Tape::from_file(&a[4])?;
  for (actual,wanted) in back.archives[0].packets[count..].iter().zip(&mine.packets){
   if (actual.word0,actual.flags,actual.mode,actual.steer,actual.accel,actual.brake)!=(wanted.word0,wanted.flags,wanted.mode,wanted.steer,wanted.accel,wanted.brake){return Err("race input changed".into())}
  }
  println!("PASS_COUNTDOWN_SCAFFOLD_ONLY_NO_TEACHER_RACING_CONTROLS {}",a[4]);return Ok(())
 }
 if a.get(1).map(|s|s.as_str())==Some("inspect-tape"){
  for file in &a[2..]{
   let t=Tape::from_file(file)?;
   println!("TAPE {file} version {} archives {}",t.chunk_version,t.archives.len());
   for ar in &t.archives{
    println!("format {} field0 {} start {} packets {} tail {}",ar.format_version,ar.field0,ar.start_offset_ms,ar.packets.len(),ar.tail.len());
    for (i,p) in ar.packets.iter().enumerate(){
     if i<4 || (155..165).contains(&i) || (215..225).contains(&i) || i>ar.packets.len().saturating_sub(4){
      println!("{i}: word{} flags{} mode{} state{:?} s{} g{} b{}",p.word0,p.flags,p.mode,p.state,p.steer_i8(),p.accel,p.brake);
     }
    }
   }
  }return Ok(())
 }
 if a.get(1).map(|s|s.as_str())==Some("wrap"){return wrap(&a)}
 if a.len()!=4{return Err("writer VERIFIED_MODEL.json FORMAT_TEMPLATE.Ghost.Gbx NEW_DIRECTORY".into())}
 let data:Value=serde_json::from_slice(&std::fs::read(&a[1]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
 let frames=data["frames"].as_array().ok_or("actual frames required")?;
 if data["goal_ms"]!=17712 || frames.len()!=1776{return Err("bound source run required".into())}
 let root=Path::new(&a[3]);std::fs::create_dir(root).map_err(|e|e.to_string())?;
 let container=Container::load(&a[2])?;
 let mut packets=Vec::new();
 // Input interval [t,t+10) is the applied command recorded at its ending POST.
 for f in &frames[1..1773]{
  let sgb=numbers(f,"sgb");
  if ![-1.,0.,1.].contains(&sgb[0]) || ![0.,1.].contains(&sgb[1]) || ![0.,1.].contains(&sgb[2]){
   return Err("non-digital source input: refuse to quantize".into())
  }
  packets.push(Packet{word0:2,flags:0,mode:2,state:StateEnc::Lit(literal_for(2,0)),
   mouse:None,vsame:false,steer:((sgb[0]*127.) as i8) as u8 as u32,
   accel:sgb[1] as u32,brake:sgb[2] as u32,tri:None});
 }
 let tape=Tape{chunk_version:4,archives:vec![Archive{format_version:12,field0:387790,
  start_offset_ms:0,packets,orig_bitstream_len:0,orig_bits_used:0,tail:Vec::new(),orig_bitstream:Vec::new()}]};
 let tape_file=root.join("input-owned.Ghost.Gbx");
 // splice_into returns a body; inject_into returns a complete GBX file.
 write_gbx(&container.gbx,tape.splice_into(container.body(),Encoding::Explicit)?,tape_file.to_str().unwrap())?;
 let record_file=root.join("record-owned.Ghost.Gbx");
 let mut unknown_constants=Vec::new();
 rewrite_ghost(tape_file.to_str().unwrap(),record_file.to_str().unwrap(),|rd|{
  let vehicles:Vec<usize>=rd.ents.iter().enumerate().filter(|(_,e)|e.type_>=0
    && rd.descs[e.type_ as usize].class_id==CLASS_CSCENEVEHICLEVIS).map(|(i,_)|i).collect();
  if vehicles.len()!=1{return Err("exactly one vehicle format required".into())}
  let vi=vehicles[0];let ss=rd.ents[vi].sample_size;
  if ss!=116{return Err(format!("expected known116-byte visual sample, got{ss}"))}
  let mut format=rd.ents[vi].raw[..ss].to_vec();
  // Retain only bytes that the format source proves invariant on every sample.
  for k in 0..ss{
   if NEUTRALISE.contains(&k) || (47..69).contains(&k) || [14,15,18].contains(&k){continue}
   if rd.ents[vi].raw.chunks_exact(ss).any(|s|s[k]!=format[k]){
    return Err(format!("unclassified changing byte{k}: refuse donor animation"))
   }
   unknown_constants.push(k);
  }
  neutralise(&mut format);
  let mut raw=Vec::new();let mut times=Vec::new();
  // Use the scaffold's standard20Hz visual format for compatibility. Keep
  // the original10ms states in the audit; export actual every-fifth states.
  // No interpolated17.712-position is invented: the last stored pose is17.700.
  for f in frames.iter().step_by(5).take(355){
   let t=f["t"].as_i64().unwrap() as i32;
   let p=numbers(f,"p");let q=numbers(f,"q");let v=numbers(f,"v");let sgb=numbers(f,"sgb");
   let mut sample=format.clone();
   write_transform(&mut sample,47,&Xform{pos:[p[0] as f32,p[1] as f32,p[2] as f32],
     quat:[q[1],q[2],q[3],q[0]],vel:[v[0],v[1],v[2]]});
   sample[14]=((sgb[0]+1.)*127.).floor() as u8;
   sample[15]=(sgb[1]*255.) as u8;sample[18]=(sgb[2]*255.) as u8;
   times.push(t);raw.extend_from_slice(&sample);
  }
  rd.ents[vi].times=times;rd.ents[vi].raw=raw;
  rd.ents[vi].u02=0;rd.ents[vi].u03=17700;
  rd.ents[vi].deltas2.clear(); // no donor skid/sound/vehicle events
  for (i,e) in rd.ents.iter_mut().enumerate(){
   if i==vi{continue}
   // Keep scene descriptors/order (including the client-required empty CP
   // placeholder), but no other entity's sampled motion/state.
   e.times.clear();e.raw.clear();e.sample_size=0;
   e.deltas2.clear();e.u02=0;e.u03=17710;
  }
  rd.start_ms=0;rd.end_ms=17712;rd.bulk_notices.clear();rd.custom_modules.clear();
  Ok(())
 })?;
 // Donor timing is format scaffolding only. Bind every declared duration/split
 // to the independently decoded Goal of this new neural-policy run.
 let timed_file=root.join("timed-owned.Ghost.Gbx");
 let recorded=Container::load(record_file.to_str().unwrap())?;
 let mut timed_body=recorded.body().to_vec();
 ghost::trim::set_all_declared(&mut timed_body,17712);
 let source_result=gbx::container::read_result(&timed_body)
     .ok_or("same-map single-Goal result layout required")?;
 if source_result.entries.len()!=1{return Err("unexpected intermediate checkpoint scaffold".into())}
 let timed_body=ghost::trim::rewrite_result(&timed_body,|result|{
     result.race_ms=17712;
     result.entries[0].0=17712; // one independently decoded actual Goal, no invented CPs
 })?;
 write_gbx(&recorded.gbx,timed_body,timed_file.to_str().unwrap())?;
 let final_file=root.join("TrackmaniaRL-Model-17.712.Ghost.Gbx");
 ghost::ident::cmd(&vec!["set".into(),timed_file.to_string_lossy().into(),
   final_file.to_string_lossy().into(),"--anonymise".into(),"--no-oracle".into(),
   "--name".into(),"TrackmaniaRL Model (headless)".into(),"--trigram".into(),"TRL".into(),
   "--zone".into(),"World|Other".into(),"--skin".into(),"Skins\\Models\\CarSport\\Stadium.zip".into()]);
 let decoded=gbx::record::decode_ghost(final_file.to_str().unwrap())?;
 let final_container=Container::load(final_file.to_str().unwrap())?;
 if !final_container.body().starts_with(&0x0303F006u32.to_le_bytes()){
  return Err("invalid ghost root chunk stream (possibly nested GBX header)".into())
 }
 if final_container.splits()!=vec![17712]{return Err(format!("unmeasured intermediate splits {:?}",final_container.splits()))}
 if decoded.samples.len()!=355 || decoded.sample_period_ms!=Some(50){return Err("standard native20Hz record required".into())}
 let mut max_error=0f64;
 for (s,f) in decoded.samples.iter().zip(frames.iter().step_by(5)){
  let p=numbers(f,"p");
  if s.time_ms!=f["t"].as_i64().unwrap() as i32{return Err("sample clock changed".into())}
  let err=((s.x as f64-p[0] as f32 as f64).powi(2)+(s.y as f64-p[1] as f32 as f64).powi(2)+(s.z as f64-p[2] as f32 as f64).powi(2)).sqrt();
  max_error=max_error.max(err);
 }
 if max_error!=0.0{return Err(format!("exported positions differ{max_error}"))}
 let check=Tape::from_file(final_file.to_str().unwrap())?;
 for (p,f) in check.archives[0].packets.iter().zip(&frames[1..1773]){
  let sgb=numbers(f,"sgb");
  if p.steer_i8() as f64/127.!=sgb[0] || p.accel as f64!=sgb[1] || p.brake as f64!=sgb[2]{
   return Err("input readback differs from model".into())
  }
 }
 let report=json!({"status":"PASS_OFFLINE_MODEL_GHOST_EXPORT_CLIENT_CHECK_PENDING",
  "source":data["provenance"],"samples":355,"visual_sample_period_ms":50,"source_physics_sample_period_ms":10,"input_intervals":1772,"native_goal_ms":17712,
  "position_readback_max_error_m":max_error,"other_entity_samples_removed":true,
  "donor_vehicle_events_removed":true,"only_invariant_format_bytes_retained":unknown_constants,
  "unmeasured_visual_channels":"neutral defaults; not measured engine animation",
  "game_client_verified":false,"public_record_claimed":false,"file":final_file});
 std::fs::write(root.join("offline-audit.json"),serde_json::to_vec_pretty(&report).unwrap()).map_err(|e|e.to_string())?;
 println!("{}",report);Ok(())
}
