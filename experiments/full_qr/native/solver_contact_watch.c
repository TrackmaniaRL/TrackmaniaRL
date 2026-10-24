// Diagnostic candidate producer execution sites. No engine writes.
#define _GNU_SOURCE
#include <sys/ptrace.h>
#include <sys/user.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <time.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
static uint64_t ticks(pid_t pid){char p[64],b[4096];snprintf(p,sizeof(p),"/proc/%d/stat",pid);FILE*f=fopen(p,"r");if(!f)return 0;if(!fgets(b,sizeof(b),f)){fclose(f);return 0;}fclose(f);char*s=strrchr(b,')');if(!s)return 0;char*save=NULL;char*t=strtok_r(s+2," ",&save);for(int i=0;t&&i<19;i++)t=strtok_r(NULL," ",&save);return t?strtoull(t,NULL,10):0;}
static uint64_t nanos(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (uint64_t)t.tv_sec*1000000000+t.tv_nsec;}
static long peek(pid_t pid,size_t off,int*error){errno=0;long v=ptrace(PTRACE_PEEKUSER,pid,(void*)off,0);if(errno)*error=errno;return v;}
static int memory(pid_t pid,uintptr_t p,void*out,size_t n){for(size_t i=0;i<n;i+=sizeof(long)){errno=0;long v=ptrace(PTRACE_PEEKDATA,pid,(void*)(p+i),0);if(errno)return -1;size_t k=n-i<sizeof(v)?n-i:sizeof(v);memcpy((char*)out+i,&v,k);}return 0;}
static int poke(pid_t pid,int index,unsigned long value){return ptrace(PTRACE_POKEUSER,pid,(void*)(offsetof(struct user,u_debugreg)+index*sizeof(long)),(void*)value);}
static void w32(unsigned char*b,size_t o,uint32_t v){memcpy(b+o,&v,4);}static void w64(unsigned char*b,size_t o,uint64_t v){memcpy(b+o,&v,8);}
// Returns0 only after restoring original debug registers and detaching a live owned child.
int watch_solver_owned(int pid,uint64_t birth,uint64_t entry,uint64_t leave,uint64_t allocation,uint64_t copy,uint64_t vehicle,uint64_t clockaddr,int outfd,int readyfd,int stopfd,int milliseconds){
 if(pid<=1||!birth||entry<4096||leave<4096||entry==leave||allocation<4096||copy<4096||vehicle<4096||milliseconds<=0||milliseconds>30000||ticks(pid)!=birth)return 10;
 int attached=0,stopped=0,configured=0,result=0,status=0,error=0;unsigned long old[8]={0};const int ids[6]={0,1,2,3,6,7};uint64_t deadline=nanos()+(uint64_t)milliseconds*1000000;unsigned seq=0;
 if(ptrace(PTRACE_SEIZE,pid,0,0)){return 11;}attached=1;
 if(ptrace(PTRACE_INTERRUPT,pid,0,0)||waitpid(pid,&status,__WALL)!=pid||!WIFSTOPPED(status)){result=12;goto done;}stopped=1;
 for(int i=0;i<6;i++)old[ids[i]]=(unsigned long)peek(pid,offsetof(struct user,u_debugreg)+ids[i]*sizeof(long),&error);
 if(error||old[7]||ticks(pid)!=birth){result=13;goto done;}
 configured=1;if(poke(pid,0,entry)||poke(pid,1,leave)||poke(pid,2,allocation)||poke(pid,3,copy)||poke(pid,7,0x55)){result=14;goto done;}
 if(ptrace(PTRACE_CONT,pid,0,0)){result=15;goto done;}stopped=0;
 {char ready='1';if(write(readyfd,&ready,1)!=1){result=16;goto done;}}
 while(nanos()<deadline){
  struct pollfd pfd={.fd=stopfd,.events=POLLIN};if(poll(&pfd,1,0)>0)break;
  pid_t got=waitpid(pid,&status,__WALL|WNOHANG);if(got==0){usleep(1000);continue;}if(got<0){result=17;goto done;}if(!WIFSTOPPED(status)){result=18;attached=0;goto done;}stopped=1;
  int sig=WSTOPSIG(status);siginfo_t info;memset(&info,0,sizeof(info));if(ptrace(PTRACE_GETSIGINFO,pid,0,&info)){result=19;goto done;}
  int deliver=sig;
  if(sig==SIGTRAP&&info.si_code==TRAP_HWBKPT){
   unsigned char record[4096]={0};struct user_regs_struct regs;uint32_t native=0;
   if(ptrace(PTRACE_GETREGS,pid,0,&regs)||memory(pid,clockaddr,&native,4)){result=20;goto done;}
   memcpy(record,"TMSV0001",8);w32(record,8,4096);w32(record,12,1);w32(record,16,pid);w32(record,20,gettid());w64(record,24,++seq);w64(record,32,entry);w64(record,40,leave);w64(record,48,peek(pid,offsetof(struct user,u_debugreg)+6*sizeof(long),&error));w32(record,56,info.si_code);w32(record,60,sig);w64(record,64,nanos());w32(record,72,native);w64(record,848,birth);memcpy(record+80,&regs,sizeof(regs));
   if(memory(pid,regs.rsp,record+304,256)){result=21;goto done;}
   unsigned long hits=(unsigned long)peek(pid,offsetof(struct user,u_debugreg)+6*sizeof(long),&error)&15;
   uint32_t kind=(hits&1)?0:((hits&2)?1:((hits&4)?2:3));w32(record,76,kind);
   if(!hits||seq>8192||(kind==3&&regs.rip!=copy)){result=34;goto done;}
   w64(record,576,allocation);w64(record,584,copy);w64(record,592,vehicle);
   if(memory(pid,vehicle,record+608,192)){result=35;goto done;}
   if(kind<2){
    uint64_t context=(kind==0)?regs.rbx:regs.r14;
    w64(record,896,context);
    if(memory(pid,context,record+912,0x270)){result=36;goto done;}
    const unsigned offsets[5]={0x190,0x1a0,0x1c8,0x1f0,0x248};
    for(unsigned j=0;j<5;j++){
     uint64_t pointer=0;uint32_t count=0;
     if(memory(pid,context+offsets[j],&pointer,8)||memory(pid,context+offsets[j]+8,&count,4)){result=37;goto done;}
     w32(record,1552+j*480,offsets[j]);w32(record,1556+j*480,count);w64(record,1560+j*480,pointer);
     if(count&&pointer>4096){if(memory(pid,pointer,record+1568+j*480,464)){result=38;goto done;}}
    }
   }else if(kind==2){
    w64(record,896,regs.rax);w64(record,904,regs.rdi-0x31);
    if(memory(pid,regs.rax,record+912,192)||memory(pid,regs.rdi-0x31,record+1104,56)||memory(pid,regs.r9,record+1160,64)){result=39;goto done;}
   }
   if(write(outfd,record,sizeof(record))!=sizeof(record)){result=24;goto done;}if(poke(pid,6,0)){result=25;goto done;}deliver=0;
  }
  if(ptrace(PTRACE_CONT,pid,0,(void*)(long)deliver)){result=26;goto done;}stopped=0;
 }
 if(nanos()>=deadline)result=27;
done:
 if(attached){
  if(!stopped){if(ptrace(PTRACE_INTERRUPT,pid,0,0)||waitpid(pid,&status,__WALL)!=pid||!WIFSTOPPED(status)){return result?result:28;}stopped=1;}
  if(configured){if(poke(pid,7,0))result=29;for(int i=0;i<6;i++)if(poke(pid,ids[i],old[ids[i]]))result=30;for(int i=0;i<6;i++){int e=0;if((unsigned long)peek(pid,offsetof(struct user,u_debugreg)+ids[i]*sizeof(long),&e)!=old[ids[i]]||e)result=31;}}
  if(ptrace(PTRACE_DETACH,pid,0,0))result=32;
 }
 return result;
}
