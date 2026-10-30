//! Offline export of recorded model poses and applied controls, no engine run.
use gbx::container::{Container,write_gbx};
use gbx::record::{CLASS_CSCENEVEHICLEVIS,neutralise,NEUTRALISE};
use gbx::recwrite::{rewrite_ghost,write_transform,Xform};
use gbx::tape::{Tape,Archive,Packet,StateEnc,Encoding,literal_for};
use serde_json::{Value,json};
use std::path::Path;

fn numbers(v:&Value,key:&str)->Vec<f64>{v[key].as_array().unwrap().iter().map(|x|x.as_f64().unwrap()).collect()}
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
 ghost::trim::set_all_declared(&mut replay,17715);
 let (back,_,_,_,_)=ghost::unwrap::unwrap_body(&replay)?;
 if back!=model.body(){return Err("wrapped model ghost changed beyond node rebasing".into())}
 let mut gbx=native.gbx.clone();
 if let Some(header)=ghost::hdr::rewrite(&native,true,Some("TrackmaniaRL Model (headless)"),Some(17715)){
  gbx.user_data=header.user_data;
 }
 write_gbx(&gbx,replay,&a[4])?;
 let out=Container::load(&a[4])?;
 if ghost::unwrap::unwrap_body(out.body())?.0!=model.body(){return Err("replay disk readback changed model".into())}
 println!("PASS_REPLAY_WRAP_EXACT_MODEL_NODE {}",a[4]);Ok(())
}
fn main()->Result<(),String>{
 let a:Vec<String>=std::env::args().collect();
 if a.get(1).map(|s|s.as_str())==Some("wrap"){return wrap(&a)}
 if a.len()!=4{return Err("writer VERIFIED_MODEL.json FORMAT_TEMPLATE.Ghost.Gbx NEW_DIRECTORY".into())}
 let data:Value=serde_json::from_slice(&std::fs::read(&a[1]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
 let frames=data["frames"].as_array().ok_or("actual frames required")?;
 if data["goal_ms"]!=17715 || frames.len()!=1776{return Err("bound source run required".into())}
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
  // No interpolated17.715-position is invented: the last stored pose is17.700.
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
  rd.start_ms=0;rd.end_ms=17715;rd.bulk_notices.clear();rd.custom_modules.clear();
  Ok(())
 })?;
 let final_file=root.join("TrackmaniaRL-Model-17.715.Ghost.Gbx");
 ghost::ident::cmd(&vec!["set".into(),record_file.to_string_lossy().into(),
   final_file.to_string_lossy().into(),"--anonymise".into(),"--no-oracle".into(),
   "--name".into(),"TrackmaniaRL Model (headless)".into(),"--trigram".into(),"TRL".into(),
   "--zone".into(),"World|Other".into(),"--skin".into(),"Skins\\Models\\CarSport\\Stadium.zip".into()]);
 let decoded=gbx::record::decode_ghost(final_file.to_str().unwrap())?;
 let final_container=Container::load(final_file.to_str().unwrap())?;
 if !final_container.body().starts_with(&0x0303F006u32.to_le_bytes()){
  return Err("invalid ghost root chunk stream (possibly nested GBX header)".into())
 }
 if final_container.splits()!=vec![17715]{return Err(format!("unmeasured intermediate splits {:?}",final_container.splits()))}
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
  "source":data["provenance"],"samples":355,"visual_sample_period_ms":50,"source_physics_sample_period_ms":10,"input_intervals":1772,"native_goal_ms":17715,
  "position_readback_max_error_m":max_error,"other_entity_samples_removed":true,
  "donor_vehicle_events_removed":true,"only_invariant_format_bytes_retained":unknown_constants,
  "unmeasured_visual_channels":"neutral defaults; not measured engine animation",
  "game_client_verified":false,"public_record_claimed":false,"file":final_file});
 std::fs::write(root.join("offline-audit.json"),serde_json::to_vec_pretty(&report).unwrap()).map_err(|e|e.to_string())?;
 println!("{}",report);Ok(())
}
