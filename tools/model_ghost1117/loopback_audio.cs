// Bounded WASAPI render-endpoint capture. Never opens a microphone or changes settings.
using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;

namespace ModelReplayAudio {
 [ComImport, Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")] class EnumeratorClass {}
 [ComImport, Guid("A95664D2-9614-4F35-A746-DE8DB63617E6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
 interface Enumerator {
  [PreserveSig] int EnumAudioEndpoints(int flow, uint state, out DeviceCollection devices);
  [PreserveSig] int GetDefaultAudioEndpoint(int flow, int role, out Device device);
 }
 [ComImport, Guid("D666063F-1587-4E43-81F1-B948E807363F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
 interface Device {
  [PreserveSig] int Activate(ref Guid iid, uint context, IntPtr parameters, [MarshalAs(UnmanagedType.IUnknown)] out object instance);
  [PreserveSig] int OpenPropertyStore(uint access, out IntPtr properties);
  [PreserveSig] int GetId([MarshalAs(UnmanagedType.LPWStr)] out string id);
 }
 [ComImport, Guid("0BD7A1BE-7A1A-44DB-8397-CC5392387B5E"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
 interface DeviceCollection {
  [PreserveSig] int GetCount(out uint count);
  [PreserveSig] int Item(uint index, out Device device);
 }
 [ComImport, Guid("77AA99A0-1BD6-484F-8BC7-2C654C9A9B6F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
 interface SessionManager {
  [PreserveSig] int GetAudioSessionControl(IntPtr session, uint flags, out IntPtr control);
  [PreserveSig] int GetSimpleAudioVolume(IntPtr session, uint flags, out IntPtr volume);
  [PreserveSig] int GetSessionEnumerator(out SessionEnumerator sessions);
 }
 [ComImport, Guid("E2F5BB11-0570-40CA-ACDD-3AA01277DEE8"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
 interface SessionEnumerator {
  [PreserveSig] int GetCount(out int count);
  [PreserveSig] int GetSession(int index, out SessionControl session);
 }
 [ComImport, Guid("BFB7FF88-7239-4FC9-8FA2-07C950BE9C6D"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
 interface SessionControl {
  [PreserveSig] int GetState(out int state);
  [PreserveSig] int GetDisplayName([MarshalAs(UnmanagedType.LPWStr)] out string name);
  [PreserveSig] int SetDisplayName(IntPtr name, IntPtr context);
  [PreserveSig] int GetIconPath([MarshalAs(UnmanagedType.LPWStr)] out string path);
  [PreserveSig] int SetIconPath(IntPtr path, IntPtr context);
  [PreserveSig] int GetGroupingParam(out Guid grouping);
  [PreserveSig] int SetGroupingParam(IntPtr grouping, IntPtr context);
  [PreserveSig] int RegisterAudioSessionNotification(IntPtr notification);
  [PreserveSig] int UnregisterAudioSessionNotification(IntPtr notification);
  [PreserveSig] int GetSessionIdentifier([MarshalAs(UnmanagedType.LPWStr)] out string id);
  [PreserveSig] int GetSessionInstanceIdentifier([MarshalAs(UnmanagedType.LPWStr)] out string id);
  [PreserveSig] int GetProcessId(out uint pid);
 }
 [ComImport, Guid("87CE5498-68D6-44E5-9215-6DA47EF883D8"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
 interface SessionVolume {
  [PreserveSig] int SetMasterVolume(float value, IntPtr context);
  [PreserveSig] int GetMasterVolume(out float value);
  [PreserveSig] int SetMute([MarshalAs(UnmanagedType.Bool)] bool value, IntPtr context);
  [PreserveSig] int GetMute([MarshalAs(UnmanagedType.Bool)] out bool value);
 }
 [ComImport, Guid("1CB9AD4C-DBFA-4C32-B178-C2F568A703B2"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
 interface AudioClient {
  [PreserveSig] int Initialize(int mode, uint flags, long duration, long period, IntPtr format, IntPtr session);
  [PreserveSig] int GetBufferSize(out uint frames);
  [PreserveSig] int GetStreamLatency(out long latency);
  [PreserveSig] int GetCurrentPadding(out uint frames);
  [PreserveSig] int IsFormatSupported(int mode, IntPtr format, out IntPtr closest);
  [PreserveSig] int GetMixFormat(out IntPtr format);
  [PreserveSig] int GetDevicePeriod(out long normal, out long minimum);
  [PreserveSig] int Start();
  [PreserveSig] int Stop();
  [PreserveSig] int Reset();
  [PreserveSig] int SetEventHandle(IntPtr handle);
  [PreserveSig] int GetService(ref Guid iid, [MarshalAs(UnmanagedType.IUnknown)] out object service);
 }
 [ComImport, Guid("C8ADBD64-E71E-48A0-A4DE-185C395CD317"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
 interface CaptureClient {
  [PreserveSig] int GetBuffer(out IntPtr data, out uint frames, out uint flags, out ulong position, out ulong qpc);
  [PreserveSig] int ReleaseBuffer(uint frames);
  [PreserveSig] int GetNextPacketSize(out uint frames);
 }
 public sealed class Receipt {
  public string StartedUtc, FirstPacketUtc, FinishedUtc;
  public long Frames, Bytes, Packets, SilentPackets, Discontinuities, TimestampErrors;
  public int SampleRate, Channels, Bits, FormatTag;
  public double ElapsedSeconds;
  public string EndpointId;
  public uint GameProcessId;
  public int GameSessionState;
  public bool GameSessionMuted;
  public float GameSessionVolume;
 }
 public static class Record {
  static void Check(int hr) { Marshal.ThrowExceptionForHR(hr); }
  static void Four(BinaryWriter w, string s) { w.Write(System.Text.Encoding.ASCII.GetBytes(s)); }
  static Device FindGame(Enumerator enumerator, uint pid, Receipt r, bool? setMute=null) {
   DeviceCollection devices; Check(enumerator.EnumAudioEndpoints(0,1,out devices));
   try {
    uint count;Check(devices.GetCount(out count));
    for(uint i=0;i<count;i++) {
     Device d;Check(devices.Item(i,out d));object raw=null;SessionEnumerator sessions=null;bool keep=false;
     try {
      Guid iid=typeof(SessionManager).GUID;Check(d.Activate(ref iid,23,IntPtr.Zero,out raw));
      Check(((SessionManager)raw).GetSessionEnumerator(out sessions));int n;Check(sessions.GetCount(out n));
      for(int k=0;k<n;k++) {
       SessionControl s;Check(sessions.GetSession(k,out s));
       try {
        uint actual;Check(s.GetProcessId(out actual));if(actual!=pid)continue;
        Check(s.GetState(out r.GameSessionState));
        Check(((SessionVolume)s).GetMute(out r.GameSessionMuted));
        Check(((SessionVolume)s).GetMasterVolume(out r.GameSessionVolume));
        if(setMute.HasValue) {
         Check(((SessionVolume)s).SetMute(setMute.Value,IntPtr.Zero));
         Check(((SessionVolume)s).GetMute(out r.GameSessionMuted));
        }
        Check(d.GetId(out r.EndpointId));r.GameProcessId=pid;keep=true;return d;
       } finally { Marshal.ReleaseComObject(s); }
      }
     } finally {
      if(sessions!=null)Marshal.ReleaseComObject(sessions);
      if(raw!=null)Marshal.ReleaseComObject(raw);
      if(!keep)Marshal.ReleaseComObject(d);
     }
    }
   } finally { Marshal.ReleaseComObject(devices); }
   throw new InvalidOperationException("No active render endpoint contains the specified game process session");
  }
  public static Receipt SetMute(uint gamePid, bool muted) {
   object e=new EnumeratorClass();Device d=null;
   try { Receipt r=new Receipt();d=FindGame((Enumerator)e,gamePid,r,muted);return r; }
   finally { if(d!=null)Marshal.ReleaseComObject(d);Marshal.ReleaseComObject(e); }
  }
  public static Receipt Run(string output, int seconds, string ready, uint gamePid) {
   if(seconds<1 || seconds>45 || File.Exists(output) || File.Exists(ready)) throw new ArgumentException("Fresh bounded capture only");
   object enumerator=new EnumeratorClass(), device=null, rawClient=null, rawCapture=null;
   IntPtr format=IntPtr.Zero;
   bool started=false;
   AudioClient client=null;
   Receipt r=new Receipt();
   Stopwatch timer=new Stopwatch();
   try {
    Device selected=FindGame((Enumerator)enumerator,gamePid,r);device=selected;
    Guid clientIid=typeof(AudioClient).GUID; Check(selected.Activate(ref clientIid,23,IntPtr.Zero,out rawClient));
    client=(AudioClient)rawClient; Check(client.GetMixFormat(out format));
    int formatBytes=18+(ushort)Marshal.ReadInt16(format,16);
    if(formatBytes<18 || formatBytes>256) throw new InvalidDataException("Mix format length");
    byte[] fmt=new byte[formatBytes]; Marshal.Copy(format,fmt,0,fmt.Length);
    r.FormatTag=BitConverter.ToUInt16(fmt,0);r.Channels=BitConverter.ToUInt16(fmt,2);
    r.SampleRate=(int)BitConverter.ToUInt32(fmt,4);r.Bits=BitConverter.ToUInt16(fmt,14);
    int align=BitConverter.ToUInt16(fmt,12);
    if(align<1 || r.SampleRate<8000 || r.SampleRate>384000) throw new InvalidDataException("Mix format");
    Check(client.Initialize(0,0x20000,1000000,0,format,IntPtr.Zero));
    Guid captureIid=typeof(CaptureClient).GUID; Check(client.GetService(ref captureIid,out rawCapture));
    CaptureClient capture=(CaptureClient)rawCapture;
    using(FileStream fs=new FileStream(output,FileMode.CreateNew,FileAccess.ReadWrite,FileShare.Read))
    using(BinaryWriter w=new BinaryWriter(fs)) {
     Four(w,"RIFF");w.Write(0);Four(w,"WAVE");Four(w,"fmt ");w.Write(fmt.Length);w.Write(fmt);
     Four(w,"data");long lengthOffset=fs.Position;w.Write(0);long dataStart=fs.Position;
     long anchorQpc=Stopwatch.GetTimestamp(); DateTime anchorUtc=DateTime.UtcNow;
     Check(client.Start());started=true;timer.Start();r.StartedUtc=DateTime.UtcNow.ToString("o");
     File.WriteAllText(ready,r.StartedUtc);
     while(timer.Elapsed.TotalSeconds<seconds) {
      uint next;Check(capture.GetNextPacketSize(out next));
      while(next>0) {
       IntPtr data;uint frames,flags;ulong position,qpc;
       Check(capture.GetBuffer(out data,out frames,out flags,out position,out qpc));
       try {
        if(frames==0) break;
        if((flags&1)!=0)r.Discontinuities++;
        if((flags&4)!=0)r.TimestampErrors++;
        if(r.FirstPacketUtc==null && (flags&4)==0) {
         double delta=qpc/10000000.0-anchorQpc/(double)Stopwatch.Frequency-r.Frames/(double)r.SampleRate;
         r.FirstPacketUtc=anchorUtc.AddSeconds(delta).ToString("o");
        }
        byte[] bytes=new byte[checked((int)frames*align)];
        if((flags&2)==0) Marshal.Copy(data,bytes,0,bytes.Length); else r.SilentPackets++;
        w.Write(bytes);r.Frames+=frames;r.Packets++;
       } finally { Check(capture.ReleaseBuffer(frames)); }
       Check(capture.GetNextPacketSize(out next));
      }
      Thread.Sleep(3);
     }
     Check(client.Stop());started=false;r.Bytes=fs.Position-dataStart;
     fs.Position=lengthOffset;w.Write(checked((uint)r.Bytes));fs.Position=4;w.Write(checked((uint)fs.Length-8));
    }
    r.ElapsedSeconds=timer.Elapsed.TotalSeconds;r.FinishedUtc=DateTime.UtcNow.ToString("o");return r;
   } finally {
    if(started && client!=null)client.Stop();
    if(format!=IntPtr.Zero)Marshal.FreeCoTaskMem(format);
    foreach(object item in new object[]{rawCapture,rawClient,device,enumerator}) if(item!=null && Marshal.IsComObject(item))Marshal.ReleaseComObject(item);
   }
  }
 }
}
