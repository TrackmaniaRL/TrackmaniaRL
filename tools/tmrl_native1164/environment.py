"""Fall05 physical-state environment, distinct from the full-V5 motor/UI profile.

No UI-finish, LiDAR, contact penalty or rumble values are inferred. The observation
contains causal native physics only. Native Goal facts require separate replay
validation before admitting a training environment.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import socket
import time
from types import MappingProxyType
import numpy as np

from trackmaniarl.trackmania.native_headless._runtime.immediate import ImmediateSession, physical_state
from trackmaniarl.trackmania.native_headless._runtime import env as wire
from trackmaniarl.trackmania.native_headless._runtime.host import IncompleteReceiveError
from trackmaniarl.trackmania.native_headless._runtime.shapes import RequestSpec, ResponseSpec
from trackmaniarl.trackmania.native_headless._runtime.shapes import CaptureSpec
from trackmaniarl.trackmania.native_headless._runtime.capture import Capture
from trackmaniarl.trackmania.native_headless._runtime.posix_api import linux_os,linux_socket
from trackmaniarl.trackmania.native_headless.assets import VerifiedAssets
from trackmaniarl.trackmania.native_headless.config import NativeRuntimeConfig
from trackmaniarl.trackmania.native_headless.full_qr.profile import PROFILE, PROTOCOL
from trackmaniarl.trackmania.native_headless.full_qr.goal_delivery import NativeGoalContext, HeldNativeGoalReader, decode_goal_facts
from multi_goal import MultiGoalReader, decode_multi_goal_facts


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def canonical_control(value):
    if len(value)!=3 or not all(type(x) in (int,float,np.float32,np.float64) and math.isfinite(float(x)) for x in value):
        raise ValueError('Three finite S/G/B controls required')
    steer,gas,brake=map(float,value)
    if not -1<=steer<=1 or gas not in (0.,1.) or brake not in (0.,1.):
        raise ValueError('Native steering [-1,1], literal binary gas/brake required')
    return struct.unpack('<3f',struct.pack('<3f',steer,gas,brake))


class FloatSession(ImmediateSession):
    """Same guarded V7 wire, authorizing one explicit float32 S/G/B per tick."""
    def reset_bootstrap(self):
        if self.episode is not None:raise RuntimeError('Fresh source bootstrap required')
        self.provenance.update(literalResetNativeT=2190,raceOriginNativeT=2200,
            initialIntervalPolicyControlled=True,observationContract='tmrl-causal-native22-rotation/reset2200/v2')
        number=len(self.captures);spawned=self.command('spawn',episode=number)
        if spawned['kind']!='spawned':raise RuntimeError('Owned source bootstrap spawn failed')
        self.pid,self.meta,self.ep=spawned['pid'],spawned['metadata'],Path(spawned['directory'])
        self.episode=number;self.capture=Capture(CaptureSpec(self.ep,self.pid,self.tmp,self.provenance));self.captures.append(self.capture);self.capture.location()
        self.listener.settimeout(min(90,self.remaining()));self.connection,_=self.listener.accept()
        if struct.unpack('3i',self.connection.getsockopt(socket.SOL_SOCKET,linux_socket.SO_PEERCRED,12))!=(self.pid,linux_os.getuid(),linux_os.getgid()):raise RuntimeError('Owned bootstrap socket peer differs')
        self.seq,self.epoch,self.requests,self.responses=0,1,[],[]
        self.boundary,_=self.response(ResponseSpec(kind=1,seq=0,n=0,epoch=1))
        if wire.u(self.boundary,112)!=2190:raise RuntimeError('Literal native2190 bootstrap required')
        self.arm();self.post_offset,self.events,self.audits,self.hits=0,{},[],set();self._event_times=[];self.poll_posts()
        state=physical_state(self.packet())
        if state['appliedSGB']!=[0.,0.,0.]:raise RuntimeError('Actual neutral pre-input bootstrap required')
        return self.packet()
    def advance_control(self,value):
        sgb=canonical_control(value)
        before=physical_state(self.packet())
        reader=self.reader;anchor=before['nativeT'];index=wire.u(reader,112)
        if index+1>=self.meta['input_count']:raise RuntimeError('Leave an actual next reader; no tape-exhaustion terminal')
        original=self.read_mem(wire.q(reader,88)-4+index*32,32)
        prior=original[4:16];after=struct.pack('<3f',*sgb)
        epoch,sequence=self.epoch,self.seq+1
        self.poll_posts();audit_start=len(self.audits)
        self.request(RequestSpec(op=6,n=1,p=index,k=index,befores=[prior],afters=[after]))
        committed,receipt=self.response(ResponseSpec(kind=6,seq=sequence,n=1,epoch=epoch))
        if committed!=reader or receipt!=struct.pack('<4I',index,index,1,2)+prior+after+bytes(120):
            raise RuntimeError('Exact current-record commit receipt differs')
        eof=False
        try:
            self.boundary,_=self.response(ResponseSpec(kind=1,seq=sequence,n=1,epoch=epoch+1))
            if wire.u(self.boundary,112)!=anchor+10:raise RuntimeError('Native ACK clock differs')
            self.arm()
        except IncompleteReceiveError as error:
            evidence=error.evidence
            if not evidence['reason'].startswith('EOFError:') or evidence['currentResponseBytes']!=0 or evidence['currentResponseHex']!='':raise
            eof=True
        self.poll_posts()
        posts=self._decision_events(anchor,anchor+10)
        audits=self.audits[audit_start:]
        if len(audits)!=1 or len(posts)!=1:raise RuntimeError('Exactly one actual channel-write audit and physical POST required')
        audit=audits[0];post=posts[0]
        wire.qualify_post(post,self.pid,self.meta)
        if (audit[:8]!=b'TMAUD003' or wire.u(audit,16)!=self.pid or wire.u(audit,20)!=self.pid
            or audit[40:72]!=original or audit[72:76]!=original[:4] or audit[88:104]!=original[16:]
            or audit[76:88]!=after or wire.u(audit,180)!=index or wire.u(audit,200)!=sequence
            or wire.q(audit,192)!=epoch or wire.q(audit,216)!=self.meta['input_base']
            or wire.q(audit,232)!=self.meta['input_base']-4+index*32
            or wire.u(post,112)!=anchor+10):
            raise RuntimeError('Owned input audit/address/sequence or physical clock mismatch')
        gas,brake,steer=struct.unpack_from('<3f',post,88)
        if struct.pack('<3f',steer,gas,brake)!=after:raise RuntimeError('Actual applied POST controls differ from model command')
        self.boundary=posts[-1]
        if not eof and wire.u(self.reader,112)!=index+1:raise RuntimeError('Next decoded input index mismatch')
        if eof:self.end(truncated=False)
        return self.packet(),eof,sgb


from orientation import rotation_matrix

def observation(state):
    physical=state['physical']
    # Absolute position and time are measured sensors, not a reference lookup.
    values=[(state['nativeT']-2200)/30000.,*rotation_matrix(physical['quaternionWXYZ']),
        *(x/1000 for x in physical['physicalPosition']),*(x/100 for x in physical['physicalVelocity']),
        physical['rpm']/10000,physical['gearRaw']/10,physical['gas'],physical['brake'],physical['steer'],physical['filteredSteer']]
    result=np.asarray(values,dtype=np.float32)
    if result.shape!=(22,) or not np.isfinite(result).all():raise RuntimeError('Causal physical observation shape/finiteness')
    return result


class FallEnvironment:
    def __init__(self,config,folder):
        self.config=config;self.folder=Path(folder);self.session=None;self.closed=False;self.commands=[];self.frames=[]
        root=Path(config['runtime_root']);manifest=Path(config['manifest']);raw=json.loads(manifest.read_text())
        if sha(manifest)!=config['manifest_sha256']:raise RuntimeError('Fall asset manifest changed')
        paths={role:root/name for role,name in raw['roles'].items()}
        for role,path in paths.items():
            item=raw['assets'][raw['roles'][role]]
            if path.resolve()!=path or not path.is_file() or sha(path)!=item['sha256']:raise RuntimeError(f'Fall asset pin {role}')
        if sha(paths['map'])!='ed61a14dcfed9d4b3e3e961b1f1655deef6ac3f5ad05b6875f7722c1440004cc':raise RuntimeError('Fall05 map identity')
        settings=NativeRuntimeConfig(runtime_root=root,trace_dir=Path(config['private'])/'traces'/self.folder.name,work_dir=root/'build/server-probe'/self.folder.name,
            asset_manifest=manifest,timeout_s=150,duration_limit=150000,allow_native=True,reset_profile='neutral-race-start/v7',protocol_profile=PROTOCOL,max_trace_bytes=256*2**20)
        assets=VerifiedAssets(PROFILE,root,MappingProxyType(paths),config['manifest_sha256'])
        self.settings,self.assets=settings,assets

    def reset(self):
        if self.session is not None:raise RuntimeError('Fresh Fall environment per episode required')
        self.folder.mkdir(exist_ok=False,parents=True)
        self.session=FloatSession(self.settings,self.assets)
        packet=self.session.reset_bootstrap()
        packet,eof,applied=self.session.advance_control((0.,0.,0.))
        if eof or applied!=(0.,0.,0.):raise RuntimeError('Measured neutral reset interval required')
        self.session.provenance.update(literalResetNativeT=2200,initialIntervalPolicyControlled=False,administrativeNeutralReset=True)
        self.state=physical_state(packet)
        if self.state['nativeT']!=2200:raise RuntimeError('Actual native2200 first policy observation required')
        context=NativeGoalContext(self.session.pid,self.session.meta)
        self.goal_reader=MultiGoalReader(self.session.local_posts/f'goal-receipt-{self.session.pid}.bin',self.session.local_posts/f'goal-fact-{self.session.pid}.bin',context)
        self.context=context;self.initial_position=self.state['physical']['physicalPosition'];self.started=time.monotonic()
        self.frames.append(observation(self.state))
        return self.frames[-1]

    def step(self,action):
        if self.closed or self.session is None:raise RuntimeError('Active environment required')
        if time.monotonic()-self.started>140:raise RuntimeError('Bounded actual native episode wall time')
        packet,eof,applied=self.session.advance_control(action)
        state=physical_state(packet,terminal=eof);self.state=state
        race_ms=state['nativeT']-2200
        if eof:
            path=self.session.local_posts/f'goal-fact-{self.session.pid}.bin'
            facts=decode_multi_goal_facts(path.read_bytes(),self.context)
            goals=[f.event_native_t-2200 for f in facts if f.endpoint_requested and f.event_native_t<=state['nativeT']]
        else:
            boundary=self.goal_reader.read(state['nativeT'])
            goals=[boundary.last_goal_native_t-2200] if boundary.goal_active else []
        goal=min(goals) if goals else None
        terminated=goal is not None
        truncated=not terminated and (eof or race_ms>=30000)
        self.commands.append(applied);self.frames.append(observation(state))
        reward=-.01
        if terminated:reward+=100.+10.*(17.601-goal/1000.)
        if truncated:reward-=20.
        info=dict(actual_goal_ms=goal,race_ms=race_ms,finished=terminated,EOF_is_not_finish=True,
            observation_contract='causal-native22-rotation/v1',map_uid='oqIJ5rQDRrNwLPTh9H2p_W4tLof')
        return self.frames[-1],reward,terminated,truncated,info

    def close(self):
        if self.closed:return
        self.closed=True
        if self.session is not None:self.session.close()
        self.folder.mkdir(exist_ok=True,parents=True)
        np.savez_compressed(self.folder/'actual-rollout.npz',observations=np.asarray(self.frames),controls=np.asarray(self.commands,dtype=np.float32))


def qualify(config_path,output,controls_path):
    config=json.loads(Path(config_path).read_text());source=json.loads(Path(controls_path).read_text());controls=source['controls'];offset=source['race_command_packet_offset']-1
    env=FallEnvironment(config,output);result=dict(status='FAILED')
    try:
        initial=env.reset()
        # A native archive's final input remains held until Finish. Observe at
        # most five fresh ticks after its last record; never invent telemetry.
        for index in range(min(3000,len(controls)-offset+5)):
            # Native archive packet zero is initial state; racing commands start at one.
            action=controls[min(index+offset,len(controls)-1)]
            _,_,terminated,truncated,info=env.step(action)
            if terminated or truncated:
                result.update(status='NATIVE_GOAL_PENDING_INDEPENDENT_REPLAY_COMPARISON' if terminated else 'DID_NOT_FINISH',steps=index+1,info=info,initial_position=env.initial_position,source_tail_hold_ticks=max(0,index+offset-len(controls)+1))
                break
    finally:
        env.close()
        result['trace_dir']=str(env.settings.trace_dir)
        result['supervisor']=json.loads((env.settings.trace_dir/'native-supervisor.json').read_text()) if (env.settings.trace_dir/'native-supervisor.json').exists() else None
        (env.folder/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
