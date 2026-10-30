"""Multi-checkpoint append proof; legacy one-checkpoint decoder is unchanged.

The extra observer bytes capture each actual last native destination, including
its address, rather than assuming that a waypoint is always the first entry.
The Goal classifier and held-boundary decoders retain their existing bytecode;
only their explicitly bound writer-proof dependency differs in this namespace.
"""
import struct
from types import FunctionType
from trackmaniarl.trackmania.native_headless.full_qr import goal_delivery as original
from trackmaniarl.trackmania.native_headless.full_qr.waypoint_capture import decode_writer_pairs


def multi_writer_pairs(raw,*,pid,meta):
    if not raw or len(raw)%1024:raise ValueError('Complete multi-checkpoint pairs required')
    normalized=bytearray(raw)
    u=lambda b,o:struct.unpack_from('<I',b,o)[0]
    q=lambda b,o:struct.unpack_from('<Q',b,o)[0]
    for i in range(0,len(raw),1024):
        entry,store=raw[i:i+512],raw[i+512:i+1024]
        for record in (entry,store):
            if record[:8]!=b'TMWP0002' or u(record,12)!=2:raise ValueError('New measured append schema required')
        for offset,last,pointer in ((416,480,6),(448,488,11)):
            count=u(store,offset+12)
            address=q(store,496+(0 if offset==416 else 8))
            expected=q(store,64+pointer*8)+16+(count-1)*8
            if not 1<=count<=64 or address!=expected or u(store,last)!=u(store,292) or u(store,last+4)&65535!=u(store,288):
                raise ValueError('Actual appended waypoint destination/address mismatch')
            # Reuse the strict identity/caller/store decoder on a normalized
            # one-entry view, after proving the measured appended entry above.
            at=i+512+offset
            struct.pack_into('<I',normalized,at+12,1)
            normalized[at+16:at+24]=store[last:last+8]
        for at in (i,i+512):
            normalized[at:at+8]=b'TMWP0001'
            struct.pack_into('<I',normalized,at+12,1)
    return decode_writer_pairs(bytes(normalized),pid=pid,meta=meta)


def bound_function(function,**bindings):
    namespace={**function.__globals__,**bindings}
    return FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)


decode_multi_goal_facts=bound_function(original.decode_goal_facts,decode_writer_pairs=multi_writer_pairs)


class MultiGoalReader(original.HeldNativeGoalReader):
    read=bound_function(original.HeldNativeGoalReader.read,decode_goal_facts=decode_multi_goal_facts)
