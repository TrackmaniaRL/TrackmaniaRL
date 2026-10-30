//! Private input carrier and exact, observed input export for Fall05.
//! Inherited visual samples and declared times are never treated as fitness.
use gbx::{container::{Container,write_gbx},tape::{Tape,Encoding}};
use serde_json::json;
fn main()->Result<(),String>{
 let a:Vec<String>=std::env::args().collect();
 if a.len()!=5{return Err("fall-tape FALL_REPLAY LONG_CARRIER FRESH_TAPE FRESH_INPUT_JSON".into())}
 for p in [&a[3],&a[4]]{if std::path::Path::new(p).exists(){return Err("fresh outputs required".into())}}
 let c=Container::load(&a[1])?;
 if !c.uids().iter().any(|(_,u)|u=="v96wCr7mP5xEGxNLSnZC6DAp4se"){return Err("Fall05 UID required".into())}
 let original=Tape::from_file(&a[1])?;original.verbatim_is_identity()?;
 let carrier=Tape::from_file(&a[2])?;carrier.verbatim_is_identity()?;
 if original.archives.len()!=1||carrier.archives.len()!=1||carrier.n()!=16152||carrier.archives[0].start_offset_ms != -1510{return Err("qualified long neutral-bootstrap carrier layout required".into())}
 let source=&original.archives[0];
 eprintln!("Source modes/steering examples: {:?}",source.packets.iter().filter(|p|p.steer!=0).take(10).map(|p|(p.mode,p.steer,p.steer as i32)).collect::<Vec<_>>());
 if !(-3000..=0).contains(&source.start_offset_ms)||source.start_offset_ms%10!=0{return Err(format!("unsupported source archive clock {}",source.start_offset_ms))}
 let controls:Vec<_>=source.packets.iter().map(|p|json!([p.steer_i8() as f64/127.,p.accel,p.brake])).collect();
 // Keep the nonconstant, codec-proven carrier required by the native input
 // locator. Every racing tick is subsequently replaced before its input load.
 let skip=((-1510-source.start_offset_ms)/10) as usize;
 if source.start_offset_ms > -1510 || skip>=source.packets.len() || source.packets[..skip].iter().any(|p|p.steer!=0||p.accel!=0||p.brake!=0){return Err("only measured neutral pre-countdown packets may be cropped".into())}
 let mut edited=original.clone();
 edited.archives[0].start_offset_ms = -1510;
 edited.archives[0].packets=source.packets[skip..].to_vec();
 let first=&mut edited.archives[0].packets[0];
 first.state=gbx::tape::StateEnc::Lit(gbx::tape::literal_for(first.word0,first.flags));
 let last=edited.archives[0].packets.last().ok_or("source empty")?.clone();
 edited.archives[0].packets.resize(16152,last);
 write_gbx(&c.gbx,edited.splice_into(c.body(),Encoding::Explicit)?,&a[3])?;
 let back=Tape::from_file(&a[3])?;
 if back.n()!=16152||back.archives[0].start_offset_ms != -1510||back.archives[0].packets.iter().zip(edited.archives[0].packets.iter()).any(|(p,q)|(p.word0,p.flags,p.mode,p.steer,p.accel,p.brake)!=(q.word0,q.flags,q.mode,q.steer,q.accel,q.brake)){return Err("carrier roundtrip mismatch".into())}
 let out=json!({"schema":"fall05.actual-inputs/v1","source":a[1],"source_start_offset_ms":source.start_offset_ms,"source_packets":controls.len(),"controls":controls,
  "source_declared_goal_ms":c.result().map(|r|r.race_ms),"carrier_packets":16152,"carrier_start_offset_ms":-1510,
  "carrier_is_input_only":true,"visual_samples_inherited":true,"declared_time_is_not_fitness":true,"race_command_packet_offset":(10-source.start_offset_ms)/10});
 std::fs::write(&a[4],serde_json::to_vec_pretty(&out).unwrap()).map_err(|e|e.to_string())?;
 println!("Prepared exact input carrier and {} measured-source input packets",source.packets.len());Ok(())
}
