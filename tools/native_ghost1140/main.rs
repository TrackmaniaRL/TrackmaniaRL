//! Offline comparison: measured 864-byte states against a genuine game recording.
//! No regenerated ghost is emitted until source coverage has been inspected.
#[allow(dead_code)]
mod vislayout;
use std::collections::BTreeMap;
use serde_json::{Value,json};
use vislayout::State;

fn wrap_replay(args:&[String])->Result<(),String>{
 if args.len()!=6{return Err("wrap MODEL_GHOST NATIVE_REPLAY_TEMPLATE FRESH_REPLAY FRESH_REPORT".into())}
 for output in [&args[4],&args[5]]{if std::path::Path::new(output).exists(){return Err("Never overwrite a replay/report".into())}}
 let model=gbx::container::Container::load(&args[2])?;
 let native=gbx::container::Container::load(&args[3])?;
 let goal=model.result().ok_or("Actual model result required")?.race_ms;
 if model.gbx.class_id!=0x03092000 || native.gbx.class_id!=0x03093000 || !(1..=17712).contains(&goal) ||
  !native.uids().iter().any(|(_,u)|u=="oqIJ5rQDRrNwLPTh9H2p_W4tLof") ||
  !model.uids().iter().any(|(_,u)|u=="oqIJ5rQDRrNwLPTh9H2p_W4tLof"){
  return Err("Bound map99162 model/replay format required".into())
 }
 let (_,start,end,index,old_at)=ghost::unwrap::unwrap_body(native.body())?;
 if index==0 || !(3..=64).contains(&native.gbx.num_nodes){return Err("Known replay node topology required".into())}
 let (map_at,map_size)=gbx::container::embedded_map_in(native.body()).ok_or("Actual embedded map required")?;
 let map=native.body()[map_at..map_at+map_size].to_vec();
 let mut rebased=model.body().to_vec();let pattern=[0,240,17,9,0,240,17,9];
 let hits:Vec<usize>=rebased.windows(8).enumerate().filter(|(_,w)|*w==pattern).map(|(i,_)|i).collect();
 if hits.len()!=1 || hits[0]<4{return Err("Single model record node required".into())}
 let at=hits[0]-4;
 rebased[at..at+4].copy_from_slice(&native.body()[start+old_at..start+old_at+4]);
 let mut body=native.body()[..start].to_vec();body.extend_from_slice(&rebased);body.extend_from_slice(&native.body()[end..]);
 // The model already carries its proven actual Goal. Never scan/rewrite the embedded map.
 if ghost::unwrap::unwrap_body(&body)?.0!=model.body(){return Err("Model node changed during wrapping".into())}
 let mut header=native.gbx.clone();
 if let Some(rewritten)=ghost::hdr::rewrite(&native,false,None,Some(goal as u32)){header.user_data=rewritten.user_data;}
 gbx::container::write_gbx(&header,body,&args[4])?;
 let back=gbx::container::Container::load(&args[4])?;
 if ghost::unwrap::unwrap_body(back.body())?.0!=model.body(){return Err("Replay readback changed exact model".into())}
 let (new_at,new_size)=gbx::container::embedded_map_in(back.body()).ok_or("Embedded map lost")?;
 if back.body()[new_at..new_at+new_size]!=map{return Err("Embedded map changed".into())}
 let declared=ghost::hdr::header_declared_ms(&back);
 if declared.is_empty() || declared.iter().any(|(_,t)|*t!=goal as u32){return Err("Replay header result disagreement".into())}
 let decoded=gbx::record::decode_ghost(&args[4])?;
 let source=gbx::record::decode_ghost(&args[2])?;
 if decoded.raw!=source.raw || decoded.race_time_ms!=Some(goal){return Err("Replay vehicle data changed".into())}
 let report=json!({"status":"PASS_OFFLINE_REPLAY_CONTAINER_EXACT_MODEL_CLIENT_CHECK_PENDING",
  "actual_native_goal_ms":goal,"exact_model_node_unchanged":true,"embedded_map_bytes_unchanged":true,
  "recorded_vehicle_bytes_unchanged":true,"header_declared_times_ms":declared,"samples":decoded.samples.len(),
  "map_bytes":map_size,"ghost":args[2],"replay":args[4],"game_client_opened":false,
  "native_client_saved":false,"public_submission":false,"new_simulations":0});
 std::fs::write(&args[5],serde_json::to_vec_pretty(&report).unwrap()).map_err(|e|e.to_string())?;
 println!("{}",json!({"status":report["status"],"actual_native_goal_ms":goal,"samples":decoded.samples.len()}));
 Ok(())
}

fn recorded_states(raw:&[u8],owner:u32)->Result<BTreeMap<u32,&[u8]>,String>{
 if raw.is_empty() || raw.len()%1280!=0{return Err("Incomplete capture".into())}
 let mut states=BTreeMap::new();
 for r in raw.chunks_exact(1280){
  let u=|o:usize|u32::from_le_bytes(r[o..o+4].try_into().unwrap());
  if &r[..8]!=b"TMVH0001" || (u(8),u(12),u(16),u(20))!=(1280,1,owner,owner){return Err("Capture provenance".into())}
  let state=&r[208..1072];let t=u(32);
  if u32::from_le_bytes(state[12..16].try_into().unwrap())!=t{return Err("State clock".into())}
  if let Some(old)=states.insert(t,state){if old!=state{return Err("Conflicting same-clock states".into())}}
 }
 Ok(states)
}

fn export_control(args:&[String])->Result<(),String>{
 if args.len()!=7{return Err("export-control RAW PHYSICS NATIVE_TEMPLATE FRESH_GHOST FRESH_REPORT".into())}
 let autonomous=args[1]=="export-autonomous";
 let raw=std::fs::read(&args[2]).map_err(|e|e.to_string())?;
 let physics:Value=serde_json::from_slice(&std::fs::read(&args[3]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
 let goal=physics["goal_ms"].as_i64().ok_or("Actual Goal")? as i32;
 if !(1..=17712).contains(&goal){return Err("An actual improvement on this bounded reference is required".into())}
 if autonomous{
  if physics["status"]!="VERIFIED_AUTONOMOUS_MODEL_TRACE_FOR_GHOST" ||
   physics["provenance"]["actual_goals_ms"]!=json!(vec![goal;5]) ||
   physics["provenance"]["teacher_files_used"]!=false{return Err("Verified autonomous source required".into())}
 }else if goal!=17712{return Err("Same-time baseline control only".into())}
 let origin=physics["native_origin_ms"].as_u64().unwrap_or(2200) as u32;
 let owner=physics["provenance"]["native_pid"].as_u64().ok_or("Owner")? as u32;
 let states=recorded_states(&raw,owner)?;
 let original=gbx::record::decode_ghost(&args[4])?;
 if original.race_time_ms!=Some(17712) || original.sample_size!=116{return Err("Genuine native17.712 baseline required".into())}
 let template=gbx::container::Container::load(&args[4])?;
 if template.gbx.class_id!=0x03092000 || !template.uids().iter().any(|(_,u)|u=="oqIJ5rQDRrNwLPTh9H2p_W4tLof") ||
  template.result().ok_or("Single Goal template")?.entries.len()!=1{return Err("Bound map99162 ghost topology required".into())}
 for output in [&args[5],&args[6]]{if std::path::Path::new(output).exists(){return Err("Never overwrite an export".into())}}
 let frames=physics["frames"].as_array().ok_or("Frames")?;
 let last_observed=frames.last().and_then(|f|f["t"].as_i64()).ok_or("Observed end")? as i32;
 let times:Vec<i32>=original.samples.iter().map(|s|s.time_ms).filter(|&t|t<=last_observed).collect();
 if times.last().copied().unwrap_or(-1)<goal{return Err("Observed visual recording ends before actual Goal".into())}
 let mut packed=Vec::new();
 for &t in &times{
  if t<0 || t%10!=0{return Err("Unaligned baseline visual clock".into())}
  let f=frames.get((t/10)as usize).ok_or("No actual source frame for native visual sample")?;
  if f["t"]!=t{return Err("Source frame clock".into())}
  let state=Recorded(states.get(&(t as u32+origin)).ok_or("Missing measured state")?);
  let mut bytes=vislayout::pack_checked(&state)?;
  let q=f["q"].as_array().ok_or("Actual physical orientation")?;
  if q.len()!=4 || q.iter().any(|x|x.as_f64().map_or(true,|v|!v.is_finite())){return Err("Invalid actual orientation".into())}
  if (q.iter().map(|v|v.as_f64().unwrap().powi(2)).sum::<f64>()-1.).abs()>0.001{return Err("Actual orientation norm".into())}
  for off in [0x50,0x54,0x58,0x5c,0x60,0x64,0x74,0x198,0xac,0xd8,0x104,0x130]{
   if !state.f32(off).is_finite(){return Err("Nonfinite measured visual field".into())}
  }
  let p=[state.f32(0x50),state.f32(0x54),state.f32(0x58)];
  let v=[state.f32(0x5c)as f64,state.f32(0x60)as f64,state.f32(0x64)as f64];
  gbx::recwrite::write_transform(&mut bytes,47,&gbx::recwrite::Xform{pos:p,
   quat:[q[1].as_f64().unwrap(),q[2].as_f64().unwrap(),q[3].as_f64().unwrap(),q[0].as_f64().unwrap()],vel:v});
  // The archiver has an explicit unset sentinel. No client/donor countdown is copied.
  let start=state.u32(0x340);
  let countdown=if start==u32::MAX{-1}else{-2-(t as u32+origin).wrapping_sub(start).min(3000)as i32};
  bytes[108..112].copy_from_slice(&countdown.to_le_bytes());
  packed.extend_from_slice(&bytes);
 }
 let temporary=std::path::Path::new(&args[5]).with_extension("visual-stage.Gbx");
 if temporary.exists(){return Err("Fresh visual intermediate required".into())}
 gbx::recwrite::rewrite_ghost(&args[4],temporary.to_str().ok_or("Output path")?,|record|{
  let vehicles:Vec<_>=record.ents.iter().enumerate().filter(|(_,e)|e.type_>=0 &&
   record.descs[e.type_ as usize].class_id==gbx::record::CLASS_CSCENEVEHICLEVIS).map(|(i,_)|i).collect();
  if vehicles.len()!=1{return Err("Single vehicle format required".into())}
  let entity=&mut record.ents[vehicles[0]];
  if entity.sample_size!=116 || !entity.times.starts_with(&times){return Err("Native sample clock topology changed".into())}
  entity.times=times.clone();entity.u03=*times.last().unwrap();
  entity.raw=packed.clone();
  if autonomous{
   for (index,other) in record.ents.iter_mut().enumerate(){
    if index==vehicles[0]{continue}
    other.times.clear();other.raw.clear();other.sample_size=0;other.deltas2.clear();
   }
   record.bulk_notices.clear();record.custom_modules.clear();
   record.ents[vehicles[0]].deltas2.clear();
  }
  record.end_ms=last_observed;
  // Stop at the measured endpoint. No post-finish poses are fabricated to fill a client tail.
  Ok(())
 })?;
 let visual=gbx::container::Container::load(temporary.to_str().unwrap())?;
 let mut body=visual.body().to_vec();
 if autonomous{
  let mut tape=gbx::tape::Tape::from_file(&args[4])?;
  if tape.archives.len()!=1 || tape.archives[0].start_offset_ms!=0 || tape.n()!=2072{return Err("Known native input framing required".into())}
  let source=tape.archives[0].packets.clone();
  let tail=&source[1773..];
  if tail.len()!=299 || tail.iter().any(|p|p.mode!=15 || p.steer_i8()!=0 || p.accel!=0 || p.brake!=0){return Err("Known inactive native post-finish framing required".into())}
  let intervals=(goal as usize+9)/10;
  let mut packets=source[..intervals+1].to_vec();
  for i in 0..intervals{
   let values=frames.get(i+1).and_then(|f|f["sgb"].as_array()).ok_or("Actual applied command")?;
   let s=values[0].as_f64().ok_or("Steer")?;let g=values[1].as_f64().ok_or("Gas")?;let b=values[2].as_f64().ok_or("Brake")?;
   if ![-1.,0.,1.].contains(&s) || ![0.,1.].contains(&g) || ![0.,1.].contains(&b){return Err("No quantization of actual controls".into())}
   let packet=&mut packets[i+1];
   if packet.word0!=2 || packet.flags!=0 || packet.mode!=2 || packet.respawn(){return Err("Known racing packet required".into())}
   packet.steer=((s*127.)as i8)as u8 as u32;packet.accel=g as u32;packet.brake=b as u32;
  }
  packets.extend_from_slice(tail);tape.archives[0].packets=packets;
  body=tape.splice_into(&body,gbx::tape::Encoding::Explicit)?;
  ghost::trim::set_all_declared(&mut body,goal as u32);
  body=ghost::trim::rewrite_result(&body,|result|{result.race_ms=goal;result.entries[0].0=goal;})?;
 }
 let mut header=visual.gbx.clone();
 if let Some(rewritten)=ghost::hdr::rewrite(&visual,false,None,Some(goal as u32)){header.user_data=rewritten.user_data;}
 gbx::container::write_gbx(&header,body,&args[5])?;
 let back=gbx::record::decode_ghost(&args[5])?;
 if back.raw!=packed || back.samples.len()!=times.len() || back.race_time_ms!=Some(goal){return Err("Packed visual readback".into())}
 let before_tape=gbx::tape::Tape::from_file(&args[4])?;
 let after_tape=gbx::tape::Tape::from_file(&args[5])?;
 if !autonomous && format!("{before_tape:?}")!=format!("{after_tape:?}"){return Err("Actual native input tape changed".into())}
 if autonomous{
  let intervals=(goal as usize+9)/10;
  if after_tape.n()!=intervals+300{return Err("Changed applied-input clock".into())}
  for (i,packet) in after_tape.archives[0].packets[1..intervals+1].iter().enumerate(){
   let values=frames[i+1]["sgb"].as_array().unwrap();
   if packet.steer_i8() as f64/127.!=values[0].as_f64().unwrap() || packet.accel as f64!=values[1].as_f64().unwrap() || packet.brake as f64!=values[2].as_f64().unwrap(){return Err("Actual applied command readback changed".into())}
  }
 }
 let report=json!({"status":if autonomous{"GENERATED_AUTONOMOUS_GHOST_NATIVE_AND_CLIENT_CHECK_PENDING"}else{"MEASURED_SERVER_VISUAL_BASELINE_EXPORT_CLIENT_COSMETICS_UNVERIFIED"},
  "actual_native_goal_ms":goal,"samples":back.samples.len(),"sample_size":116,"input_tape_unchanged":!autonomous,
  "actual_applied_model_commands_encoded":autonomous,"native_client_saved":false,"public_submission":false,
  "validation_metadata_template_inherited_not_new_client_attestation":true,
  "new_simulations":0,"game_client_opened":false,"donor_vehicle_bytes_used":false,
  "neutralise_used":false,"source":physics["provenance"],
  "known_unpopulated_client_only_channels":vislayout::DEAD_IN_SERVER,
  "client_only_channel_status":"actual server slots; not claimed to reproduce client-only computed cosmetics",
  "visual_recording_readback_exact":true,"output":args[5]});
 std::fs::write(&args[6],serde_json::to_vec_pretty(&report).unwrap()).map_err(|e|e.to_string())?;
 println!("{}",json!({"status":report["status"],"samples":back.samples.len(),"report":args[6]}));
 Ok(())
}

#[cfg(test)]
mod tests{
 use super::*;
 fn row()->Vec<u8>{
  let mut r=vec![0;1280];r[..8].copy_from_slice(b"TMVH0001");
  for (at,v) in [(8,1280u32),(12,1),(16,41),(20,41),(32,2200),(220,2200)]{r[at..at+4].copy_from_slice(&v.to_le_bytes())}
  r
 }
 #[test]fn reject_wrong_owner_and_incomplete_state(){
  let r=row();assert!(recorded_states(&r,42).is_err());assert!(recorded_states(&r[..1279],41).is_err());
  assert!(recorded_states(&[],41).is_err());
 }
 #[test]fn reject_clock_disagreement_and_conflicting_same_clock_states(){
  let mut r=row();r[220]=0;assert!(recorded_states(&r,41).is_err());
  let mut a=row();let mut b=row();b[224]=1;a.extend_from_slice(&b);assert!(recorded_states(&a,41).is_err());
 }
 #[test]fn actual_same_clock_duplicate_is_unambiguous(){
  let mut r=row();r.extend_from_slice(&row());assert_eq!(recorded_states(&r,41).unwrap().len(),1);
 }
}

struct Recorded<'a>(&'a[u8]);
impl State for Recorded<'_>{
 fn f32(&self,o:usize)->f32{f32::from_le_bytes(self.0[o..o+4].try_into().unwrap())}
 fn u32(&self,o:usize)->u32{u32::from_le_bytes(self.0[o..o+4].try_into().unwrap())}
 fn u8(&self,o:usize)->u8{self.0[o]}
 fn covers_state(&self)->bool{self.0.len()==864}
}
fn main()->Result<(),String>{
 let args:Vec<String>=std::env::args().collect();
 if args.get(1).map(|s|s.as_str())==Some("wrap"){return wrap_replay(&args)}
 if args.get(1).map(|s|s=="export-control" || s=="export-autonomous")==Some(true){return export_control(&args)}
 if args.len()!=5{return Err("compare RAW_CAPTURE PHYSICS_JSON GENUINE_NATIVE_GHOST FRESH_REPORT".into())}
 let root=std::path::Path::new(&args[1]);
 let raw=std::fs::read(root).map_err(|e|e.to_string())?;
 let physics:Value=serde_json::from_slice(&std::fs::read(&args[2]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
 let owner=physics["provenance"]["native_pid"].as_u64().ok_or("Owner")? as u32;
 let states=recorded_states(&raw,owner)?;
 let original=gbx::record::decode_ghost(&args[3])?;
 if original.race_time_ms!=Some(17712) || original.sample_size!=116{return Err("Bound actual native17.712 control required".into())}
 let mut matches=[0usize;116];let mut compared=0usize;let mut differences=Vec::new();
 let mut front_max=0f32;let mut wheel_max=0f32;let mut speed_error=0f32;let mut rpm_error=0f32;
 let frames=physics["frames"].as_array().ok_or("Frames")?;
 for (i,sample) in original.samples.iter().enumerate(){
  let t=sample.time_ms;
  if t<0 || t>17750 || t%10!=0{continue}
  let state=Recorded(states.get(&(t as u32+2200)).ok_or("Missing exact recorded state")?);
  let predicted=vislayout::pack_checked(&state)?;
  let native=original.raw_sample(i).ok_or("Native sample")?;
  compared+=1;
  for j in 0..116{if predicted[j]==native[j]{matches[j]+=1}}
  front_max=front_max.max(state.f32(0x74).abs());wheel_max=wheel_max.max(state.f32(0xac));
  let observed_front=u16::from_le_bytes(native[0..2].try_into().unwrap()) as f32/65535.*11000.-1000.;
  speed_error=speed_error.max((observed_front-state.f32(0x74)).abs());
  let observed_rpm=u16::from_le_bytes(native[4..6].try_into().unwrap()) as f32/65535.*30000.;
  rpm_error=rpm_error.max((observed_rpm-state.f32(0x198)).abs());
  if differences.len()<7 && (t==0 || t==1000 || t==5000 || t==10000 || t==17000){
   differences.push(json!({"time_ms":t,"native_clock_tail":i32::from_le_bytes(native[108..112].try_into().unwrap()),
    "state_clock_field":state.u32(0x340),"native_front_speed":observed_front,"recorded_front_speed":state.f32(0x74),
    "native_rpm":observed_rpm,"recorded_rpm":state.f32(0x198),"native_bytes":native,"predicted_bytes":predicted.to_vec(),
    "recorded_matrix":(0..9).map(|k|state.f32(0x2c+4*k)).collect::<Vec<_>>(),"physics_quat":frames[(t/10)as usize]["q"]}));
  }
 }
 if std::path::Path::new(&args[4]).exists(){return Err("Refuse report overwrite".into())}
 let report=json!({"status":"READ_ONLY_SAME_LINE_COMPARISON_NOT_EXPORT_APPROVAL","samples":compared,
  "bytes_exact_all_samples":(0..116).filter(|&i|matches[i]==compared).collect::<Vec<_>>(),
  "matches_per_byte":matches.to_vec(),"known_unpredicted_bytes":vislayout::UNPREDICTED,
  "known_server_unpopulated_channels":vislayout::DEAD_IN_SERVER,
  "maximum_front_speed_error_m_s":speed_error,"maximum_rpm_error":rpm_error,
  "recorded_front_speed_max_m_s":front_max,"recorded_fl_rotation_max_rad":wheel_max,"examples":differences,
  "new_simulations":0,"game_client_opened":false});
 std::fs::write(&args[4],serde_json::to_vec_pretty(&report).unwrap()).map_err(|e|e.to_string())?;
 println!("{}",json!({"samples":compared,"maximum_front_speed_error_m_s":speed_error,"maximum_rpm_error":rpm_error,"report":args[4]}));
 Ok(())
}
