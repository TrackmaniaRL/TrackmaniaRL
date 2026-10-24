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
int watch_owned(int pid,uint64_t birth,uint64_t address,uint64_t clockaddr,uint64_t execute,int outfd,int readyfd,int stopfd,int milliseconds){
 if(pid<=1||!birth||execute<4096||address%4||milliseconds<=0||milliseconds>30000||ticks(pid)!=birth)return 10;
 int attached=0,stopped=0,configured=0,result=0,status=0,error=0;unsigned long old[8]={0};const int ids[6]={0,1,2,3,6,7};uint64_t deadline=nanos()+(uint64_t)milliseconds*1000000;unsigned seq=0;
 if(ptrace(PTRACE_SEIZE,pid,0,0)){return 11;}attached=1;
 if(ptrace(PTRACE_INTERRUPT,pid,0,0)||waitpid(pid,&status,__WALL)!=pid||!WIFSTOPPED(status)){result=12;goto done;}stopped=1;
 for(int i=0;i<6;i++)old[ids[i]]=(unsigned long)peek(pid,offsetof(struct user,u_debugreg)+ids[i]*sizeof(long),&error);
 if(error||old[7]||ticks(pid)!=birth){result=13;goto done;}
 configured=1;if(poke(pid,0,address)||poke(pid,1,execute)||poke(pid,7,0xd0005)){result=14;goto done;}
 if(ptrace(PTRACE_CONT,pid,0,0)){result=15;goto done;}stopped=0;
 {char ready='1';if(write(readyfd,&ready,1)!=1){result=16;goto done;}}
 while(nanos()<deadline){
  struct pollfd pfd={.fd=stopfd,.events=POLLIN};if(poll(&pfd,1,0)>0)break;
  pid_t got=waitpid(pid,&status,__WALL|WNOHANG);if(got==0){usleep(1000);continue;}if(got<0){result=17;goto done;}if(!WIFSTOPPED(status)){result=18;attached=0;goto done;}stopped=1;
  int sig=WSTOPSIG(status);siginfo_t info;memset(&info,0,sizeof(info));if(ptrace(PTRACE_GETSIGINFO,pid,0,&info)){result=19;goto done;}
  int deliver=sig;
  if(sig==SIGTRAP&&info.si_code==TRAP_HWBKPT){
   unsigned char record[1024]={0};struct user_regs_struct regs;uint64_t value=0;uint32_t native=0;unsigned char header[16]={0};
   if(ptrace(PTRACE_GETREGS,pid,0,&regs)||memory(pid,address,&value,4)||memory(pid,clockaddr,&native,4)){result=20;goto done;}
   memcpy(record,"TMHW0004",8);w32(record,8,1024);w32(record,12,4);w32(record,16,pid);w32(record,20,gettid());w64(record,24,++seq);w64(record,32,address);w64(record,40,value);w64(record,48,peek(pid,offsetof(struct user,u_debugreg)+6*sizeof(long),&error));w32(record,56,info.si_code);w32(record,60,sig);w64(record,64,nanos());w32(record,72,native);w64(record,768,birth);memcpy(record+80,&regs,sizeof(regs));
   if(memory(pid,regs.rsp,record+304,256)){result=21;goto done;}
   if(memory(pid,address-8,header,16)){result=22;goto done;}memcpy(record+576,header,16);uint64_t vector=0;uint32_t count=0;memcpy(&vector,header,8);memcpy(&count,header+8,4);
   if(count&&count<=10000&&vector){if(memory(pid,vector+(uint64_t)(count-1)*88,record+608,88)){result=23;goto done;}}
   struct user_fpregs_struct fp;if(ptrace(PTRACE_GETFPREGS,pid,0,&fp)){result=33;goto done;}
   w32(record,784,fp.mxcsr);w32(record,788,fp.cwd);memcpy(record+800,fp.xmm_space+13*4,16);memcpy(record+816,fp.xmm_space+3*4,16);memcpy(record+832,fp.xmm_space,16);w64(record,776,execute);
   unsigned long dr6=(unsigned long)peek(pid,offsetof(struct user,u_debugreg)+6*sizeof(long),&error);unsigned kind=(dr6&2)?2:1;w32(record,76,kind);
   uint64_t parent_bp=0,ret=0,base_ns=0,clockptr=0;uintptr_t module=execute-0x9364ed;
   if(kind==2){
    if(regs.rip!=execute||memory(pid,regs.rbp,&parent_bp,8)||memory(pid,regs.rbp+8,&ret,8)){result=35;goto done;}
    base_ns=regs.rbx;if(memory(pid,regs.r14,record+608,88)){result=36;goto done;}
   }else if(value&&regs.rip==module+0xa557d5){parent_bp=regs.rbp;if(memory(pid,parent_bp-0x68,&base_ns,8)){result=37;goto done;}}
   if(parent_bp&&(kind==1||ret==module+0xa55768)){
    if(memory(pid,parent_bp-0x80,&clockptr,8)||memory(pid,clockptr,record+848,16)){result=38;goto done;}
   }
   w64(record,592,base_ns);w64(record,600,(kind==2||value)?regs.r14:0);w64(record,752,parent_bp);w64(record,760,ret);
   uint64_t body_a=0,body_b=0;memcpy(&body_a,record+608,8);memcpy(&body_b,record+616,8);
   if(kind==2||(value&&value<=10000)){if(memory(pid,body_b,record+864,64)||memory(pid,body_a,record+928,64)){result=34;goto done;}}

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
