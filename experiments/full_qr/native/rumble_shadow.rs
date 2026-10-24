//! Synthetic converter qualification, never physical collision evidence.
//! Calls the pinned native converter on observer-owned receiver/queue storage.
use super::*;

unsafe extern "C" fn owned_queue(receiver: *mut u8) -> *mut u8 {
    receiver.add(0x200)
}

unsafe fn output(pid: usize) -> i32 {
    let mut path = [0u8; 320];
    let n = DIR_LEN.load(Ordering::SeqCst);
    if n == 0 || n > 240 { fail(901) }
    let dir = &*core::ptr::addr_of!(DIR);
    path[..n].copy_from_slice(&dir[..n]);
    let prefix = b"/rumble-shadow-";
    path[n..n + prefix.len()].copy_from_slice(prefix);
    let start = n + prefix.len();
    let mut digits = [0u8; 24];
    let mut value = pid;
    let mut count = 0;
    loop {
        digits[count] = b'0' + (value % 10) as u8;
        count += 1;
        value /= 10;
        if value == 0 { break }
    }
    for i in 0..count { path[start + i] = digits[count - i - 1]; }
    let end = start + count;
    path[end..end + 4].copy_from_slice(b".bin");
    let fd = sc(257, (-100isize) as usize, path.as_ptr() as usize, 0x800c1, 0o600, 0, 0);
    if fd < 0 { fail(902) }
    fd as i32
}

pub unsafe fn synthetic_fixture(pid: usize, base: usize) {
    let mut signature = [0u8; 20];
    if !read(base + 0x1356080, &mut signature)
        || signature != [0x55,0x48,0x89,0xe5,0x41,0x57,0x41,0x56,0x53,0x48,
                        0x83,0xec,0x28,0x0f,0x29,0x4d,0xc0,0xf3,0x0f,0x10]
    { fail(903) }
    // The full converter calls powf and the bounded inline native queue writer.
    // Its only virtual call is replaced by our receiver's owned-queue accessor.
    type Converter = unsafe extern "C" fn(*mut u8, u32, u32, f32, f32);
    let converter: Converter = core::mem::transmute(base + 0x1356080);
    let mut vtable = [0usize; 41];
    vtable[40] = owned_queue as *const () as usize;
    let fd = output(pid);
    let mut sequence = 0;
    for setting in [0.0f32, 0.5, 1.0, 2.0] {
        for amplitude in [-0.01f32, 0.0, 0.1, 0.5, 1.0] {
            for frequency in [0.0f32, 3.5, 36.75, 70.0, 100.0] {
                let mut receiver = [0u8; 1024];
                receiver[..8].copy_from_slice(&(vtable.as_ptr() as usize as u64).to_le_bytes());
                receiver[0x188..0x18c].copy_from_slice(&setting.to_le_bytes());
                receiver[720..].fill(0xa5);
                let before = receiver;
                let clock = need(dword(base + 0x1e44568),904);
                let minimum_duration = need(dword(base + 0x1e4456c),904);
                let native_time = need(dword(VALIDATOR_SIM.load(Ordering::SeqCst) + 0x48),904);
                converter(receiver.as_mut_ptr(), 0, 80, amplitude, frequency);
                if receiver[..512] != before[..512] || receiver[720..] != before[720..]
                    || need(dword(base + 0x1e44568),904) != clock
                    || need(dword(base + 0x1e4456c),904) != minimum_duration
                    || need(dword(VALIDATOR_SIM.load(Ordering::SeqCst) + 0x48),904) != native_time
                { fail(905) }
                let count = u32::from_le_bytes(receiver[512..516].try_into().unwrap());
                if count > 1 { fail(906) }
                sequence += 1;
                let mut record = [0u8; 512];
                record[..8].copy_from_slice(b"TMRS0001");
                w32(&mut record,8,512);w32(&mut record,12,1);w32(&mut record,16,pid as u32);
                w32(&mut record,20,sequence);w32(&mut record,24,native_time);
                w32(&mut record,28,1); // explicit synthetic input marker
                record[32..36].copy_from_slice(&amplitude.to_le_bytes());
                record[36..40].copy_from_slice(&frequency.to_le_bytes());
                record[40..44].copy_from_slice(&setting.to_le_bytes());
                w32(&mut record,44,80);w32(&mut record,48,clock);w32(&mut record,52,minimum_duration);
                w64(&mut record,56,base + 0x1356080);
                record[64..272].copy_from_slice(&receiver[512..720]);
                w32(&mut record,272,need(dword(base + 0x1cb4558),907));
                w32(&mut record,276,need(dword(base + 0x1cb455c),907));
                w32(&mut record,280,1); // own receiver prefix/canary and clocks unchanged
                let mut offset = 0;
                while offset < record.len() {
                    let n = sc(1,fd as usize,record[offset..].as_ptr() as usize,record.len()-offset,0,0,0);
                    if n == -4 { continue }
                    if n <= 0 { fail(908) }
                    offset += n as usize;
                }
            }
        }
    }
    sc(3,fd as usize,0,0,0,0,0);
}
