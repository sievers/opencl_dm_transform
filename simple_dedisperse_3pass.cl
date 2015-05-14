//    Copyright Jonathan Sievers, 2015.  All rights reserved.  This code may only be used with permission of the owner. 

#include "/home/sievers/frb/opencl/opencl_frb.h"

//#define THREADS_PER_BLOCK 128
//#pragma OPENCL EXTENSION cl_amd_printf : enable

__kernel void dedisperse_1pass(__global float *a, __global float *b, int nchan, int ndata, int cursize)
{
  
  __local float s1[2*THREADS_PER_BLOCK];
  __local float s2[2*THREADS_PER_BLOCK];

  int blockDim=get_local_size(0);
  int nchunk=ndata/blockDim;
  
  int blockIdx=get_group_id(0);
  int threadIdx=get_local_id(0);


  //printf("my indices are %d %d %d\n",threadIdx,blockDim,blockIdx);

  int curchunk=(2*blockIdx)/cursize;
  int my_local_ind=blockIdx-(curchunk*cursize/2);
  int in1=cursize*curchunk+2*my_local_ind;

  int out1=cursize*curchunk+my_local_ind;
  int out2=cursize*curchunk+my_local_ind+cursize/2;
  s1[threadIdx]=a[threadIdx+ndata*(in1)];
  s2[threadIdx]=a[threadIdx+ndata*(in1+1)];

  for (int i=1;i<nchunk;i++) {
    s1[threadIdx+blockDim]=a[threadIdx+ndata*in1+i*blockDim];
    s2[threadIdx+blockDim]=a[threadIdx+ndata*(in1+1)+i*blockDim];
    barrier(CLK_LOCAL_MEM_FENCE);
#if 1
    b[threadIdx+ndata*(out1)+(i-1)*blockDim]=s1[threadIdx]+s2[threadIdx];
    if (threadIdx+(i-1)*blockDim>=my_local_ind) //cast it this way since compiler is making stuff unsigned
      b[threadIdx+ndata*(out2)+(i-1)*blockDim-my_local_ind]=s1[threadIdx]+s2[threadIdx+1];
#endif
    barrier(CLK_LOCAL_MEM_FENCE);

    s1[threadIdx]=s1[threadIdx+blockDim];
    s2[threadIdx]=s2[threadIdx+blockDim];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  b[threadIdx+ndata*(out1)+(nchunk-1)*blockDim]=s1[threadIdx]+s2[threadIdx];
  if (threadIdx<blockDim-1)
    b[threadIdx+ndata*(out2)+(nchunk-1)*blockDim-my_local_ind]=s1[threadIdx]+s2[threadIdx+1];
  barrier(CLK_LOCAL_MEM_FENCE);
  
}



/*--------------------------------------------------------------------------------*/
__kernel  void dedisperse_kernel_3pass(__global float *a, __global float *b, int nchan, int ndata, int cursize)
{

  __local float s1[2*THREADS_PER_BLOCK];
  __local float s2[2*THREADS_PER_BLOCK];
  __local float s3[2*THREADS_PER_BLOCK];
  __local float s4[2*THREADS_PER_BLOCK];
  __local float s5[2*THREADS_PER_BLOCK];
  __local float s6[2*THREADS_PER_BLOCK];
  __local float s7[2*THREADS_PER_BLOCK];
  __local float s8[2*THREADS_PER_BLOCK];

  //__shared__ float tmp[4][THREADS_PER_BLOCK];



  int blockDim=get_local_size(0);
  int nchunk=ndata/blockDim;

  int blockIdx=get_group_id(0);
  int threadIdx=get_local_id(0);


  
  int curchunk=(8*blockIdx)/cursize;


  int my_local_ind=blockIdx-(curchunk*cursize/8);

  //printf("my indices are %d %d %d %d %d\n",threadIdx,blockDim,blockIdx,my_local_ind,curchunk);

  int in1=cursize*curchunk+8*my_local_ind;



  //int out1=cursize*curchunk+my_local_ind;
  //int out2=cursize*curchunk+my_local_ind+(1*cursize)/8;
  //int out3=cursize*curchunk+my_local_ind+(2*cursize)/8;
  //int out4=cursize*curchunk+my_local_ind+(3*cursize)/8;
  //int out5=cursize*curchunk+my_local_ind+(4*cursize)/8;
  //int out6=cursize*curchunk+my_local_ind+(5*cursize)/8;
  //int out7=cursize*curchunk+my_local_ind+(6*cursize)/8;
  //int out8=cursize*curchunk+my_local_ind+(7*cursize)/8;
  int out1=cursize*curchunk+my_local_ind;
  cursize/=8;


  if (threadIdx==0) {
    //printf("out on %d are %d %d %d %d %d %d %d %d\n",blockIdx,out1,out2,out3,out4,out5,out6,out7,out8);
  }

  


  s1[threadIdx]=a[threadIdx+ndata*(in1)];
  s2[threadIdx]=a[threadIdx+ndata*(in1+1)];
  s3[threadIdx]=a[threadIdx+ndata*(in1+2)];
  s4[threadIdx]=a[threadIdx+ndata*(in1+3)];
  s5[threadIdx]=a[threadIdx+ndata*(in1+4)];
  s6[threadIdx]=a[threadIdx+ndata*(in1+5)];
  s7[threadIdx]=a[threadIdx+ndata*(in1+6)];
  s8[threadIdx]=a[threadIdx+ndata*(in1+7)];
 
  for (int i=1;i<nchunk;i++) {

    barrier(CLK_LOCAL_MEM_FENCE);
    
    s1[threadIdx+blockDim]=a[threadIdx+ndata*(in1)+i*blockDim];
    s2[threadIdx+blockDim]=a[threadIdx+ndata*(in1+1)+i*blockDim];
    s3[threadIdx+blockDim]=a[threadIdx+ndata*(in1+2)+i*blockDim];
    s4[threadIdx+blockDim]=a[threadIdx+ndata*(in1+3)+i*blockDim];
    s5[threadIdx+blockDim]=a[threadIdx+ndata*(in1+4)+i*blockDim];
    s6[threadIdx+blockDim]=a[threadIdx+ndata*(in1+5)+i*blockDim];
    s7[threadIdx+blockDim]=a[threadIdx+ndata*(in1+6)+i*blockDim];
    s8[threadIdx+blockDim]=a[threadIdx+ndata*(in1+7)+i*blockDim];

    barrier(CLK_LOCAL_MEM_FENCE);
    float tmp=s1[threadIdx]+s2[threadIdx]+s3[threadIdx]+s4[threadIdx]
      +s5[threadIdx]+s6[threadIdx]+s7[threadIdx]+s8[threadIdx];
    b[threadIdx+ndata*out1+(i-1)*blockDim]=tmp;

    tmp=s1[threadIdx]+s2[threadIdx]+s3[threadIdx]+s4[threadIdx]
      +s5[threadIdx+1]+s6[threadIdx+1]+s7[threadIdx+1]+s8[threadIdx+1];
    int ii=threadIdx+ndata*(out1+cursize)+(i-1)*blockDim-my_local_ind;
    if (ii>=0)
      b[ii]=tmp;

    tmp=s1[threadIdx]+s2[threadIdx]+s3[threadIdx+1]+s4[threadIdx+1]
      +s5[threadIdx+1]+s6[threadIdx+1]+s7[threadIdx+2]+s8[threadIdx+2];
    ii=threadIdx+ndata*(out1+2*cursize)+(i-1)*blockDim-2*my_local_ind;    
    if (ii>0)
      b[ii]=tmp;
    
    tmp=s1[threadIdx]+s2[threadIdx]+s3[threadIdx+1]+s4[threadIdx+1]
      +s5[threadIdx+2]+s6[threadIdx+2]+s7[threadIdx+3]+s8[threadIdx+3];
    ii=threadIdx+ndata*(out1+3*cursize)+(i-1)*blockDim-3*my_local_ind;    
    if (ii>0)
      b[ii]=tmp;

    tmp=s1[threadIdx]+s2[threadIdx+1]+s3[threadIdx+1]+s4[threadIdx+2]
      +s5[threadIdx+2]+s6[threadIdx+3]+s7[threadIdx+3]+s8[threadIdx+4];
    ii=threadIdx+ndata*(out1+4*cursize)+(i-1)*blockDim-4*my_local_ind;    
    if (ii>0)
      b[ii]=tmp;

    tmp=s1[threadIdx]+s2[threadIdx+1]+s3[threadIdx+1]+s4[threadIdx+2]
      +s5[threadIdx+3]+s6[threadIdx+4]+s7[threadIdx+4]+s8[threadIdx+5];
    ii=threadIdx+ndata*(out1+5*cursize)+(i-1)*blockDim-5*my_local_ind;    
    if (ii>0)
      b[ii]=tmp;
    
    tmp=s1[threadIdx]+s2[threadIdx+1]+s3[threadIdx+2]+s4[threadIdx+3]
      +s5[threadIdx+3]+s6[threadIdx+4]+s7[threadIdx+5]+s8[threadIdx+6];
    ii=threadIdx+ndata*(out1+6*cursize)+(i-1)*blockDim-6*my_local_ind;    
    if (ii>0)
      b[ii]=tmp;

    tmp=s1[threadIdx]+s2[threadIdx+1]+s3[threadIdx+2]+s4[threadIdx+3]
      +s5[threadIdx+4]+s6[threadIdx+5]+s7[threadIdx+6]+s8[threadIdx+7];
    ii=threadIdx+ndata*(out1+7*cursize)+(i-1)*blockDim-7*my_local_ind;    
    if (ii>0)
      b[ii]=tmp;


    barrier(CLK_LOCAL_MEM_FENCE);
    s1[threadIdx]=s1[threadIdx+blockDim];
    s2[threadIdx]=s2[threadIdx+blockDim];
    s3[threadIdx]=s3[threadIdx+blockDim];
    s4[threadIdx]=s4[threadIdx+blockDim];       
    s5[threadIdx]=s5[threadIdx+blockDim];       
    s6[threadIdx]=s6[threadIdx+blockDim];       
    s7[threadIdx]=s7[threadIdx+blockDim];       
    s8[threadIdx]=s8[threadIdx+blockDim];       
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  




  float tmp=s1[threadIdx]+s2[threadIdx]+s3[threadIdx]+s4[threadIdx]
    +s5[threadIdx]+s6[threadIdx]+s7[threadIdx]+s8[threadIdx];
  b[threadIdx+ndata*out1+(nchunk-1)*blockDim]=tmp;

  tmp=s1[threadIdx]+s2[threadIdx]+s3[threadIdx]+s4[threadIdx]
    +s5[threadIdx+1]+s6[threadIdx+1]+s7[threadIdx+1]+s8[threadIdx+1];
  if (threadIdx<blockDim-1)
    b[threadIdx+ndata*(out1+cursize)+(nchunk-1)*blockDim-my_local_ind]=tmp;

  tmp=s1[threadIdx]+s2[threadIdx]+s3[threadIdx+1]+s4[threadIdx+1]
    +s5[threadIdx+1]+s6[threadIdx+1]+s7[threadIdx+2]+s8[threadIdx+2];
  if (threadIdx<blockDim-2)
    b[threadIdx+ndata*(out1+2*cursize)+(nchunk-1)*blockDim-2*my_local_ind]=tmp;
  
  tmp=s1[threadIdx]+s2[threadIdx]+s3[threadIdx+1]+s4[threadIdx+1]
    +s5[threadIdx+2]+s6[threadIdx+2]+s7[threadIdx+3]+s8[threadIdx+3];
  if (threadIdx<blockDim-3)
    b[threadIdx+ndata*(out1+3*cursize)+(nchunk-1)*blockDim-3*my_local_ind]=tmp;

  tmp=s1[threadIdx]+s2[threadIdx+1]+s3[threadIdx+1]+s4[threadIdx+2]
    +s5[threadIdx+2]+s6[threadIdx+3]+s7[threadIdx+3]+s8[threadIdx+4];
  if (threadIdx<blockDim-4)
    b[threadIdx+ndata*(out1+4*cursize)+(nchunk-1)*blockDim-4*my_local_ind]=tmp;

  tmp=s1[threadIdx]+s2[threadIdx+1]+s3[threadIdx+1]+s4[threadIdx+2]
    +s5[threadIdx+3]+s6[threadIdx+4]+s7[threadIdx+4]+s8[threadIdx+5];
  if (threadIdx<blockDim-5)
    b[threadIdx+ndata*(out1+5*cursize)+(nchunk-1)*blockDim-5*my_local_ind]=tmp;

  tmp=s1[threadIdx]+s2[threadIdx+1]+s3[threadIdx+2]+s4[threadIdx+3]
    +s5[threadIdx+3]+s6[threadIdx+4]+s7[threadIdx+5]+s8[threadIdx+6];
  if (threadIdx<blockDim-6)
    b[threadIdx+ndata*(out1+6*cursize)+(nchunk-1)*blockDim-6*my_local_ind]=tmp;

  tmp=s1[threadIdx]+s2[threadIdx+1]+s3[threadIdx+2]+s4[threadIdx+3]
    +s5[threadIdx+4]+s6[threadIdx+5]+s7[threadIdx+6]+s8[threadIdx+7];
  if (threadIdx<blockDim-7)
    b[threadIdx+ndata*(out1+7*cursize)+(nchunk-1)*blockDim-7*my_local_ind]=tmp;
  barrier(CLK_LOCAL_MEM_FENCE);
}

/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
__kernel  void dedisperse_kernel_3pass_org(__global float *a, __global float *b, int nchan, int ndata, int cursize)
{

  __local float s1[2*THREADS_PER_BLOCK];
  __local float s2[2*THREADS_PER_BLOCK];
  __local float s3[2*THREADS_PER_BLOCK];
  __local float s4[2*THREADS_PER_BLOCK];
  __local float s5[2*THREADS_PER_BLOCK];
  __local float s6[2*THREADS_PER_BLOCK];
  __local float s7[2*THREADS_PER_BLOCK];
  __local float s8[2*THREADS_PER_BLOCK];

  //__shared__ float tmp[4][THREADS_PER_BLOCK];



  int blockDim=get_local_size(0);
  int nchunk=ndata/blockDim;

  int blockIdx=get_group_id(0);
  int threadIdx=get_local_id(0);


  
  int curchunk=(8*blockIdx)/cursize;


  int my_local_ind=blockIdx-(curchunk*cursize/8);

  //printf("my indices are %d %d %d %d %d\n",threadIdx,blockDim,blockIdx,my_local_ind,curchunk);

  int in1=cursize*curchunk+8*my_local_ind;

  int out1=cursize*curchunk+my_local_ind;
  int out2=cursize*curchunk+my_local_ind+(1*cursize)/8;
  int out3=cursize*curchunk+my_local_ind+(2*cursize)/8;
  int out4=cursize*curchunk+my_local_ind+(3*cursize)/8;
  int out5=cursize*curchunk+my_local_ind+(4*cursize)/8;
  int out6=cursize*curchunk+my_local_ind+(5*cursize)/8;
  int out7=cursize*curchunk+my_local_ind+(6*cursize)/8;
  int out8=cursize*curchunk+my_local_ind+(7*cursize)/8;

  if (threadIdx==0) {
    //printf("out on %d are %d %d %d %d %d %d %d %d\n",blockIdx,out1,out2,out3,out4,out5,out6,out7,out8);
  }

  


  s1[threadIdx]=a[threadIdx+ndata*(in1)];
  s2[threadIdx]=a[threadIdx+ndata*(in1+1)];
  s3[threadIdx]=a[threadIdx+ndata*(in1+2)];
  s4[threadIdx]=a[threadIdx+ndata*(in1+3)];
  s5[threadIdx]=a[threadIdx+ndata*(in1+4)];
  s6[threadIdx]=a[threadIdx+ndata*(in1+5)];
  s7[threadIdx]=a[threadIdx+ndata*(in1+6)];
  s8[threadIdx]=a[threadIdx+ndata*(in1+7)];
 
  for (int i=1;i<nchunk;i++) {

    barrier(CLK_LOCAL_MEM_FENCE);
    
    s1[threadIdx+blockDim]=a[threadIdx+ndata*(in1)+i*blockDim];
    s2[threadIdx+blockDim]=a[threadIdx+ndata*(in1+1)+i*blockDim];
    s3[threadIdx+blockDim]=a[threadIdx+ndata*(in1+2)+i*blockDim];
    s4[threadIdx+blockDim]=a[threadIdx+ndata*(in1+3)+i*blockDim];
    s5[threadIdx+blockDim]=a[threadIdx+ndata*(in1+4)+i*blockDim];
    s6[threadIdx+blockDim]=a[threadIdx+ndata*(in1+5)+i*blockDim];
    s7[threadIdx+blockDim]=a[threadIdx+ndata*(in1+6)+i*blockDim];
    s8[threadIdx+blockDim]=a[threadIdx+ndata*(in1+7)+i*blockDim];

    barrier(CLK_LOCAL_MEM_FENCE);
    float tmp1=s1[threadIdx]+s2[threadIdx]+s3[threadIdx]+s4[threadIdx]
      +s5[threadIdx]+s6[threadIdx]+s7[threadIdx]+s8[threadIdx];
    float tmp2=s1[threadIdx]+s2[threadIdx]+s3[threadIdx]+s4[threadIdx]
      +s5[threadIdx+1]+s6[threadIdx+1]+s7[threadIdx+1]+s8[threadIdx+1];
    float tmp3=s1[threadIdx]+s2[threadIdx]+s3[threadIdx+1]+s4[threadIdx+1]
      +s5[threadIdx+1]+s6[threadIdx+1]+s7[threadIdx+2]+s8[threadIdx+2];
    float tmp4=s1[threadIdx]+s2[threadIdx]+s3[threadIdx+1]+s4[threadIdx+1]
      +s5[threadIdx+2]+s6[threadIdx+2]+s7[threadIdx+3]+s8[threadIdx+3];
    float tmp5=s1[threadIdx]+s2[threadIdx+1]+s3[threadIdx+1]+s4[threadIdx+2]
      +s5[threadIdx+2]+s6[threadIdx+3]+s7[threadIdx+3]+s8[threadIdx+4];
    float tmp6=s1[threadIdx]+s2[threadIdx+1]+s3[threadIdx+1]+s4[threadIdx+2]
      +s5[threadIdx+3]+s6[threadIdx+4]+s7[threadIdx+4]+s8[threadIdx+5];
    float tmp7=s1[threadIdx]+s2[threadIdx+1]+s3[threadIdx+2]+s4[threadIdx+3]
      +s5[threadIdx+3]+s6[threadIdx+4]+s7[threadIdx+5]+s8[threadIdx+6];
    float tmp8=s1[threadIdx]+s2[threadIdx+1]+s3[threadIdx+2]+s4[threadIdx+3]
      +s5[threadIdx+4]+s6[threadIdx+5]+s7[threadIdx+6]+s8[threadIdx+7];

    //tmp1=1;
    //tmp2=1;
    //tmp3=1;
    //tmp4=1;
    //tmp5=1;
    //tmp6=1;
    //tmp7=1;
    //tmp8=1;

    b[threadIdx+ndata*out1+(i-1)*blockDim]=tmp1;

    int ii=threadIdx+ndata*out2+(i-1)*blockDim-my_local_ind;
    if (ii>=0)
      b[ii]=tmp2;

    ii=threadIdx+ndata*out3+(i-1)*blockDim-2*my_local_ind;    
    if (ii>0)
      b[ii]=tmp3;
    
    ii=threadIdx+ndata*out4+(i-1)*blockDim-3*my_local_ind;    
    if (ii>0)
      b[ii]=tmp4;

    ii=threadIdx+ndata*out5+(i-1)*blockDim-4*my_local_ind;    
    if (ii>0)
      b[ii]=tmp5;

    ii=threadIdx+ndata*out6+(i-1)*blockDim-5*my_local_ind;    
    if (ii>0)
      b[ii]=tmp6;
    ii=threadIdx+ndata*out7+(i-1)*blockDim-6*my_local_ind;    
    if (ii>0)
      b[ii]=tmp7;
    ii=threadIdx+ndata*out8+(i-1)*blockDim-7*my_local_ind;    
    if (ii>0)
      b[ii]=tmp8;


    barrier(CLK_LOCAL_MEM_FENCE);
    s1[threadIdx]=s1[threadIdx+blockDim];
    s2[threadIdx]=s2[threadIdx+blockDim];
    s3[threadIdx]=s3[threadIdx+blockDim];
    s4[threadIdx]=s4[threadIdx+blockDim];       
    s5[threadIdx]=s5[threadIdx+blockDim];       
    s6[threadIdx]=s6[threadIdx+blockDim];       
    s7[threadIdx]=s7[threadIdx+blockDim];       
    s8[threadIdx]=s8[threadIdx+blockDim];       
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  

  float tmp1=s1[threadIdx]+s2[threadIdx]+s3[threadIdx]+s4[threadIdx]
    +s5[threadIdx]+s6[threadIdx]+s7[threadIdx]+s8[threadIdx];
  float tmp2=s1[threadIdx]+s2[threadIdx]+s3[threadIdx]+s4[threadIdx]
    +s5[threadIdx+1]+s6[threadIdx+1]+s7[threadIdx+1]+s8[threadIdx+1];
  float tmp3=s1[threadIdx]+s2[threadIdx]+s3[threadIdx+1]+s4[threadIdx+1]
    +s5[threadIdx+1]+s6[threadIdx+1]+s7[threadIdx+2]+s8[threadIdx+2];
  float tmp4=s1[threadIdx]+s2[threadIdx]+s3[threadIdx+1]+s4[threadIdx+1]
    +s5[threadIdx+2]+s6[threadIdx+2]+s7[threadIdx+3]+s8[threadIdx+3];
  float tmp5=s1[threadIdx]+s2[threadIdx+1]+s3[threadIdx+1]+s4[threadIdx+2]
    +s5[threadIdx+2]+s6[threadIdx+3]+s7[threadIdx+3]+s8[threadIdx+4];
  float tmp6=s1[threadIdx]+s2[threadIdx+1]+s3[threadIdx+1]+s4[threadIdx+2]
    +s5[threadIdx+3]+s6[threadIdx+4]+s7[threadIdx+4]+s8[threadIdx+5];
  float tmp7=s1[threadIdx]+s2[threadIdx+1]+s3[threadIdx+2]+s4[threadIdx+3]
    +s5[threadIdx+3]+s6[threadIdx+4]+s7[threadIdx+5]+s8[threadIdx+6];
  float tmp8=s1[threadIdx]+s2[threadIdx+1]+s3[threadIdx+2]+s4[threadIdx+3]
    +s5[threadIdx+4]+s6[threadIdx+5]+s7[threadIdx+6]+s8[threadIdx+7];




  b[threadIdx+ndata*out1+(nchunk-1)*blockDim]=tmp1;
  if (threadIdx<blockDim-1)
    b[threadIdx+ndata*out2+(nchunk-1)*blockDim-my_local_ind]=tmp2;
  if (threadIdx<blockDim-2)
    b[threadIdx+ndata*out3+(nchunk-1)*blockDim-2*my_local_ind]=tmp3;
  if (threadIdx<blockDim-3)
    b[threadIdx+ndata*out4+(nchunk-1)*blockDim-3*my_local_ind]=tmp4;
  if (threadIdx<blockDim-4)
    b[threadIdx+ndata*out5+(nchunk-1)*blockDim-4*my_local_ind]=tmp5;
  if (threadIdx<blockDim-5)
    b[threadIdx+ndata*out6+(nchunk-1)*blockDim-5*my_local_ind]=tmp6;
  if (threadIdx<blockDim-6)
    b[threadIdx+ndata*out7+(nchunk-1)*blockDim-6*my_local_ind]=tmp7;
  if (threadIdx<blockDim-7)
    b[threadIdx+ndata*out8+(nchunk-1)*blockDim-7*my_local_ind]=tmp8;
  barrier(CLK_LOCAL_MEM_FENCE);
}

/*--------------------------------------------------------------------------------*/



/*--------------------------------------------------------------------------------*/
__kernel void dedisperse_kernel_2pass(__global float *a, __global float *b, int nchan, int ndata, int cursize)
{

  __local float s1[2*THREADS_PER_BLOCK];
  __local float s2[2*THREADS_PER_BLOCK];
  __local float s3[2*THREADS_PER_BLOCK];
  __local float s4[2*THREADS_PER_BLOCK];

  //__shared__ float tmp[4][THREADS_PER_BLOCK];


  int blockDim=get_local_size(0);
  int nchunk=ndata/blockDim;

  int blockIdx=get_group_id(0);
  int threadIdx=get_local_id(0);


  int curchunk=(4*blockIdx)/cursize;


  int my_local_ind=blockIdx-(curchunk*cursize/4);
  int in1=cursize*curchunk+4*my_local_ind;
  int out1=cursize*curchunk+my_local_ind;
  int out2=cursize*curchunk+my_local_ind+cursize/4;
  int out3=cursize*curchunk+my_local_ind+cursize/2;
  int out4=cursize*curchunk+my_local_ind+(3*cursize)/4;

  s1[threadIdx]=a[threadIdx+ndata*(in1)];
  s2[threadIdx]=a[threadIdx+ndata*(in1+1)];
  s3[threadIdx]=a[threadIdx+ndata*(in1+2)];
  s4[threadIdx]=a[threadIdx+ndata*(in1+3)];
 
  for (int i=1;i<nchunk;i++) {

    barrier(CLK_LOCAL_MEM_FENCE);
    
    s1[threadIdx+blockDim]=a[threadIdx+ndata*(in1)+i*blockDim];
    s2[threadIdx+blockDim]=a[threadIdx+ndata*(in1+1)+i*blockDim];
    s3[threadIdx+blockDim]=a[threadIdx+ndata*(in1+2)+i*blockDim];
    s4[threadIdx+blockDim]=a[threadIdx+ndata*(in1+3)+i*blockDim];
    barrier(CLK_LOCAL_MEM_FENCE);
    float tmp1=s1[threadIdx]+s2[threadIdx]+s3[threadIdx]+s4[threadIdx];
    float tmp2=s1[threadIdx]+s2[threadIdx]+s3[threadIdx+1]+s4[threadIdx+1];
    float tmp3=s1[threadIdx]+s2[threadIdx+1]+s3[threadIdx+1]+s4[threadIdx+2];
    float tmp4=s1[threadIdx]+s2[threadIdx+1]+s3[threadIdx+2]+s4[threadIdx+3];
    b[threadIdx+ndata*out1+(i-1)*blockDim]=tmp1;
    int ii=threadIdx+ndata*out2+(i-1)*blockDim-my_local_ind;
    if (ii>=0)
      b[ii]=tmp2;
    ii=threadIdx+ndata*out3+(i-1)*blockDim-2*my_local_ind;    
    if (ii>0)
      b[ii]=tmp3;
    
    ii=threadIdx+ndata*out4+(i-1)*blockDim-3*my_local_ind;    
    if (ii>0)
      b[ii]=tmp4;
    barrier(CLK_LOCAL_MEM_FENCE);
    s1[threadIdx]=s1[threadIdx+blockDim];
    s2[threadIdx]=s2[threadIdx+blockDim];
    s3[threadIdx]=s3[threadIdx+blockDim];
    s4[threadIdx]=s4[threadIdx+blockDim];       
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  
  float tmp1=s1[threadIdx]+s2[threadIdx]+s3[threadIdx]+s4[threadIdx];
  float tmp2=s1[threadIdx]+s2[threadIdx]+s3[threadIdx+1]+s4[threadIdx+1];
  float tmp3=s1[threadIdx]+s2[threadIdx+1]+s3[threadIdx+1]+s4[threadIdx+2];
  float tmp4=s1[threadIdx]+s2[threadIdx+1]+s3[threadIdx+2]+s4[threadIdx+3];
  b[threadIdx+ndata*out1+(nchunk-1)*blockDim]=tmp1;
  if (threadIdx<blockDim-1)
    b[threadIdx+ndata*out2+(nchunk-1)*blockDim-my_local_ind]=tmp2;
  if (threadIdx<blockDim-2)
    b[threadIdx+ndata*out3+(nchunk-1)*blockDim-2*my_local_ind]=tmp3;
  if (threadIdx<blockDim-3)
    b[threadIdx+ndata*out4+(nchunk-1)*blockDim-3*my_local_ind]=tmp4;
  barrier(CLK_LOCAL_MEM_FENCE);
}

/*--------------------------------------------------------------------------------*/



#if 0
/*--------------------------------------------------------------------------------*/
__global__ void dedisperse_kernel_test2(float *a, float *b, int nchan, int ndata, int cursize)
{

  __shared__ float s1[2*THREADS_PER_BLOCK];
  __shared__ float s2[2*THREADS_PER_BLOCK];
  int nchunk=ndata/blockDim.x;

  int curchunk=(2*blockIdx.x)/cursize;
  int my_local_ind=blockIdx.x-(curchunk*cursize/2);
  int in1=cursize*curchunk+2*my_local_ind;
  int out1=cursize*curchunk+my_local_ind;
  int out2=cursize*curchunk+my_local_ind+cursize/2;
  s1[threadIdx.x]=a[threadIdx.x+ndata*(in1)];
  s2[threadIdx.x]=a[threadIdx.x+ndata*(in1+1)];
  
  for (int i=1;i<nchunk;i++) {
    s1[threadIdx.x+blockDim.x]=a[threadIdx.x+ndata*in1+i*blockDim.x];
    s2[threadIdx.x+blockDim.x]=a[threadIdx.x+ndata*(in1+1)+i*blockDim.x];
    __syncthreads();
    b[threadIdx.x+ndata*(out1)+(i-1)*blockDim.x]=s1[threadIdx.x]+s2[threadIdx.x];
    if (threadIdx.x+(i-1)*blockDim.x>=my_local_ind) //cast it this way since compiler is making stuff unsigned
      b[threadIdx.x+ndata*(out2)+(i-1)*blockDim.x-my_local_ind]=s1[threadIdx.x]+s2[threadIdx.x+1];
    
    __syncthreads();
    
    s1[threadIdx.x]=s1[threadIdx.x+blockDim.x];
    s2[threadIdx.x]=s2[threadIdx.x+blockDim.x];
    __syncthreads();
  }
  
  b[threadIdx.x+ndata*(out1)+(nchunk-1)*blockDim.x]=s1[threadIdx.x]+s2[threadIdx.x];
  if (threadIdx.x<blockDim.x-1)
    b[threadIdx.x+ndata*(out2)+(nchunk-1)*blockDim.x-my_local_ind]=s1[threadIdx.x]+s2[threadIdx.x+1];
  __syncthreads();
}
#endif

#if 0


#include <stdio.h>
#include <assert.h>

#define DO_OMP
#ifdef DO_OMP
#include <omp.h>
#endif
#define THREADS_PER_BLOCK 128
#define MAX_BLOCKS 256

float **matrix(int n, int  m)
{
  float *vec=(float *)malloc(n*m*sizeof(float));
  float **mat=(float **)malloc(n*sizeof(float *));
  for (int i=0;i<n;i++)
    mat[i]=vec+i*m;
  return mat;
}

/*--------------------------------------------------------------------------------*/
float *dvec(int n) 
{
  float *dev;
  if (cudaMalloc( (void **) &dev, sizeof(float)*n )!=cudaSuccess) {
    fprintf(stderr,"alloc failure on cuda device in dvec.\n");
    assert(1==0);
  }
  cudaMemset(dev, 0, n*sizeof(float));
  return dev;

}
/*--------------------------------------------------------------------------------*/
float *mat2dev(float **mat, int n, int m)
{
  float *dev;
  if (cudaMalloc( (void **) &dev, sizeof(float)*n*m )!=cudaSuccess) {
    fprintf(stderr,"alloc failure on cuda device in mat2dev.\n");
    assert(1==0);
  }
  cudaMemcpy( dev, mat[0], sizeof(float)*n*m, cudaMemcpyHostToDevice );
  return dev;

}
/*--------------------------------------------------------------------------------*/
float **dev2mat(float *dev, int n, int m)
{
  float **mat=matrix(n,m);
  cudaMemcpy(mat[0],dev,sizeof(float)*n*m,cudaMemcpyDeviceToHost);
  return mat;
}
/*--------------------------------------------------------------------------------*/
__global__ void vector_add(float *a, float *b, int nchan, int ndata)
{
  int index=threadIdx.x+ndata*blockIdx.x;
  a[index]=-1;
  ///* insert code to calculate the index properly using blockIdx.x, blockDim.x, threadIdx.x */
  //int index = blockIdx.x * blockDim.x + threadIdx.x;
  //c[index] = a[index] + b[index];
}



/*--------------------------------------------------------------------------------*/
__global__ void dedisperse_kernel_3pass(float *a, float *b, int nchan, int ndata, int cursize)
{

  __shared__ float s1[2*THREADS_PER_BLOCK];
  __shared__ float s2[2*THREADS_PER_BLOCK];
  __shared__ float s3[2*THREADS_PER_BLOCK];
  __shared__ float s4[2*THREADS_PER_BLOCK];
  __shared__ float s5[2*THREADS_PER_BLOCK];
  __shared__ float s6[2*THREADS_PER_BLOCK];
  __shared__ float s7[2*THREADS_PER_BLOCK];
  __shared__ float s8[2*THREADS_PER_BLOCK];

  //__shared__ float tmp[4][THREADS_PER_BLOCK];

  int nchunk=ndata/blockDim.x;



  int curchunk=(8*blockIdx.x)/cursize;


  int my_local_ind=blockIdx.x-(curchunk*cursize/8);
  int in1=cursize*curchunk+8*my_local_ind;

  int out1=cursize*curchunk+my_local_ind;
  int out2=cursize*curchunk+my_local_ind+(1*cursize)/8;
  int out3=cursize*curchunk+my_local_ind+(2*cursize)/8;
  int out4=cursize*curchunk+my_local_ind+(3*cursize)/8;
  int out5=cursize*curchunk+my_local_ind+(4*cursize)/8;
  int out6=cursize*curchunk+my_local_ind+(5*cursize)/8;
  int out7=cursize*curchunk+my_local_ind+(6*cursize)/8;
  int out8=cursize*curchunk+my_local_ind+(7*cursize)/8;

  s1[threadIdx.x]=a[threadIdx.x+ndata*(in1)];
  s2[threadIdx.x]=a[threadIdx.x+ndata*(in1+1)];
  s3[threadIdx.x]=a[threadIdx.x+ndata*(in1+2)];
  s4[threadIdx.x]=a[threadIdx.x+ndata*(in1+3)];
  s5[threadIdx.x]=a[threadIdx.x+ndata*(in1+4)];
  s6[threadIdx.x]=a[threadIdx.x+ndata*(in1+5)];
  s7[threadIdx.x]=a[threadIdx.x+ndata*(in1+6)];
  s8[threadIdx.x]=a[threadIdx.x+ndata*(in1+7)];
 
  for (int i=1;i<nchunk;i++) {

    __syncthreads();
    
    s1[threadIdx.x+blockDim.x]=a[threadIdx.x+ndata*(in1)+i*blockDim.x];
    s2[threadIdx.x+blockDim.x]=a[threadIdx.x+ndata*(in1+1)+i*blockDim.x];
    s3[threadIdx.x+blockDim.x]=a[threadIdx.x+ndata*(in1+2)+i*blockDim.x];
    s4[threadIdx.x+blockDim.x]=a[threadIdx.x+ndata*(in1+3)+i*blockDim.x];
    s5[threadIdx.x+blockDim.x]=a[threadIdx.x+ndata*(in1+4)+i*blockDim.x];
    s6[threadIdx.x+blockDim.x]=a[threadIdx.x+ndata*(in1+5)+i*blockDim.x];
    s7[threadIdx.x+blockDim.x]=a[threadIdx.x+ndata*(in1+6)+i*blockDim.x];
    s8[threadIdx.x+blockDim.x]=a[threadIdx.x+ndata*(in1+7)+i*blockDim.x];
    __syncthreads();
    float tmp1=s1[threadIdx.x]+s2[threadIdx.x]+s3[threadIdx.x]+s4[threadIdx.x]
      +s5[threadIdx.x]+s6[threadIdx.x]+s7[threadIdx.x]+s8[threadIdx.x];
    float tmp2=s1[threadIdx.x]+s2[threadIdx.x]+s3[threadIdx.x]+s4[threadIdx.x]
      +s5[threadIdx.x+1]+s6[threadIdx.x+1]+s7[threadIdx.x+1]+s8[threadIdx.x+1];
    float tmp3=s1[threadIdx.x]+s2[threadIdx.x]+s3[threadIdx.x+1]+s4[threadIdx.x+1]
      +s5[threadIdx.x+1]+s6[threadIdx.x+1]+s7[threadIdx.x+2]+s8[threadIdx.x+2];
    float tmp4=s1[threadIdx.x]+s2[threadIdx.x]+s3[threadIdx.x+1]+s4[threadIdx.x+1]
      +s5[threadIdx.x+2]+s6[threadIdx.x+2]+s7[threadIdx.x+3]+s8[threadIdx.x+3];
    float tmp5=s1[threadIdx.x]+s2[threadIdx.x+1]+s3[threadIdx.x+1]+s4[threadIdx.x+2]
      +s5[threadIdx.x+2]+s6[threadIdx.x+3]+s7[threadIdx.x+3]+s8[threadIdx.x+4];
    float tmp6=s1[threadIdx.x]+s2[threadIdx.x+1]+s3[threadIdx.x+1]+s4[threadIdx.x+2]
      +s5[threadIdx.x+3]+s6[threadIdx.x+4]+s7[threadIdx.x+4]+s8[threadIdx.x+5];
    float tmp7=s1[threadIdx.x]+s2[threadIdx.x+1]+s3[threadIdx.x+2]+s4[threadIdx.x+3]
      +s5[threadIdx.x+3]+s6[threadIdx.x+4]+s7[threadIdx.x+5]+s8[threadIdx.x+6];
    float tmp8=s1[threadIdx.x]+s2[threadIdx.x+1]+s3[threadIdx.x+2]+s4[threadIdx.x+3]
      +s5[threadIdx.x+4]+s6[threadIdx.x+5]+s7[threadIdx.x+6]+s8[threadIdx.x+7];


    b[threadIdx.x+ndata*out1+(i-1)*blockDim.x]=tmp1;
    int ii=threadIdx.x+ndata*out2+(i-1)*blockDim.x-my_local_ind;
    if (ii>=0)
      b[ii]=tmp2;
    ii=threadIdx.x+ndata*out3+(i-1)*blockDim.x-2*my_local_ind;    
    if (ii>0)
      b[ii]=tmp3;
    
    ii=threadIdx.x+ndata*out4+(i-1)*blockDim.x-3*my_local_ind;    
    if (ii>0)
      b[ii]=tmp4;

    ii=threadIdx.x+ndata*out4+(i-1)*blockDim.x-4*my_local_ind;    
    if (ii>0)
      b[ii]=tmp5;

    ii=threadIdx.x+ndata*out4+(i-1)*blockDim.x-5*my_local_ind;    
    if (ii>0)
      b[ii]=tmp6;
    ii=threadIdx.x+ndata*out4+(i-1)*blockDim.x-6*my_local_ind;    
    if (ii>0)
      b[ii]=tmp7;
    ii=threadIdx.x+ndata*out4+(i-1)*blockDim.x-7*my_local_ind;    
    if (ii>0)
      b[ii]=tmp8;


    __syncthreads();
    s1[threadIdx.x]=s1[threadIdx.x+blockDim.x];
    s2[threadIdx.x]=s2[threadIdx.x+blockDim.x];
    s3[threadIdx.x]=s3[threadIdx.x+blockDim.x];
    s4[threadIdx.x]=s4[threadIdx.x+blockDim.x];       
    s5[threadIdx.x]=s5[threadIdx.x+blockDim.x];       
    s6[threadIdx.x]=s6[threadIdx.x+blockDim.x];       
    s7[threadIdx.x]=s7[threadIdx.x+blockDim.x];       
    s8[threadIdx.x]=s8[threadIdx.x+blockDim.x];       
    __syncthreads();
  }
  

  float tmp1=s1[threadIdx.x]+s2[threadIdx.x]+s3[threadIdx.x]+s4[threadIdx.x]
    +s5[threadIdx.x]+s6[threadIdx.x]+s7[threadIdx.x]+s8[threadIdx.x];
  float tmp2=s1[threadIdx.x]+s2[threadIdx.x]+s3[threadIdx.x]+s4[threadIdx.x]
    +s5[threadIdx.x+1]+s6[threadIdx.x+1]+s7[threadIdx.x+1]+s8[threadIdx.x+1];
  float tmp3=s1[threadIdx.x]+s2[threadIdx.x]+s3[threadIdx.x+1]+s4[threadIdx.x+1]
    +s5[threadIdx.x+1]+s6[threadIdx.x+1]+s7[threadIdx.x+2]+s8[threadIdx.x+2];
  float tmp4=s1[threadIdx.x]+s2[threadIdx.x]+s3[threadIdx.x+1]+s4[threadIdx.x+1]
    +s5[threadIdx.x+2]+s6[threadIdx.x+2]+s7[threadIdx.x+3]+s8[threadIdx.x+3];
  float tmp5=s1[threadIdx.x]+s2[threadIdx.x+1]+s3[threadIdx.x+1]+s4[threadIdx.x+2]
    +s5[threadIdx.x+2]+s6[threadIdx.x+3]+s7[threadIdx.x+3]+s8[threadIdx.x+4];
  float tmp6=s1[threadIdx.x]+s2[threadIdx.x+1]+s3[threadIdx.x+1]+s4[threadIdx.x+2]
    +s5[threadIdx.x+3]+s6[threadIdx.x+4]+s7[threadIdx.x+4]+s8[threadIdx.x+5];
  float tmp7=s1[threadIdx.x]+s2[threadIdx.x+1]+s3[threadIdx.x+2]+s4[threadIdx.x+3]
    +s5[threadIdx.x+3]+s6[threadIdx.x+4]+s7[threadIdx.x+5]+s8[threadIdx.x+6];
  float tmp8=s1[threadIdx.x]+s2[threadIdx.x+1]+s3[threadIdx.x+2]+s4[threadIdx.x+3]
    +s5[threadIdx.x+4]+s6[threadIdx.x+5]+s7[threadIdx.x+6]+s8[threadIdx.x+7];




  b[threadIdx.x+ndata*out1+(nchunk-1)*blockDim.x]=tmp1;
  if (threadIdx.x<blockDim.x-1)
    b[threadIdx.x+ndata*out2+(nchunk-1)*blockDim.x-my_local_ind]=tmp2;
  if (threadIdx.x<blockDim.x-2)
    b[threadIdx.x+ndata*out3+(nchunk-1)*blockDim.x-2*my_local_ind]=tmp3;
  if (threadIdx.x<blockDim.x-3)
    b[threadIdx.x+ndata*out4+(nchunk-1)*blockDim.x-3*my_local_ind]=tmp4;
  if (threadIdx.x<blockDim.x-4)
    b[threadIdx.x+ndata*out5+(nchunk-1)*blockDim.x-4*my_local_ind]=tmp5;
  if (threadIdx.x<blockDim.x-5)
    b[threadIdx.x+ndata*out6+(nchunk-1)*blockDim.x-5*my_local_ind]=tmp6;
  if (threadIdx.x<blockDim.x-6)
    b[threadIdx.x+ndata*out7+(nchunk-1)*blockDim.x-6*my_local_ind]=tmp7;
  if (threadIdx.x<blockDim.x-7)
    b[threadIdx.x+ndata*out8+(nchunk-1)*blockDim.x-7*my_local_ind]=tmp8;
  __syncthreads();
}

/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
__global__ void dedisperse_kernel_2pass(float *a, float *b, int nchan, int ndata, int cursize)
{

  __shared__ float s1[2*THREADS_PER_BLOCK];
  __shared__ float s2[2*THREADS_PER_BLOCK];
  __shared__ float s3[2*THREADS_PER_BLOCK];
  __shared__ float s4[2*THREADS_PER_BLOCK];

  //__shared__ float tmp[4][THREADS_PER_BLOCK];

  int nchunk=ndata/blockDim.x;



  int curchunk=(4*blockIdx.x)/cursize;


  int my_local_ind=blockIdx.x-(curchunk*cursize/4);
  int in1=cursize*curchunk+4*my_local_ind;
  int out1=cursize*curchunk+my_local_ind;
  int out2=cursize*curchunk+my_local_ind+cursize/4;
  int out3=cursize*curchunk+my_local_ind+cursize/2;
  int out4=cursize*curchunk+my_local_ind+(3*cursize)/4;

  s1[threadIdx.x]=a[threadIdx.x+ndata*(in1)];
  s2[threadIdx.x]=a[threadIdx.x+ndata*(in1+1)];
  s3[threadIdx.x]=a[threadIdx.x+ndata*(in1+2)];
  s4[threadIdx.x]=a[threadIdx.x+ndata*(in1+3)];
 
  for (int i=1;i<nchunk;i++) {

    __syncthreads();
    
    s1[threadIdx.x+blockDim.x]=a[threadIdx.x+ndata*(in1)+i*blockDim.x];
    s2[threadIdx.x+blockDim.x]=a[threadIdx.x+ndata*(in1+1)+i*blockDim.x];
    s3[threadIdx.x+blockDim.x]=a[threadIdx.x+ndata*(in1+2)+i*blockDim.x];
    s4[threadIdx.x+blockDim.x]=a[threadIdx.x+ndata*(in1+3)+i*blockDim.x];
    __syncthreads();
    float tmp1=s1[threadIdx.x]+s2[threadIdx.x]+s3[threadIdx.x]+s4[threadIdx.x];
    float tmp2=s1[threadIdx.x]+s2[threadIdx.x]+s3[threadIdx.x+1]+s4[threadIdx.x+1];
    float tmp3=s1[threadIdx.x]+s2[threadIdx.x+1]+s3[threadIdx.x+1]+s4[threadIdx.x+2];
    float tmp4=s1[threadIdx.x]+s2[threadIdx.x+1]+s3[threadIdx.x+2]+s4[threadIdx.x+3];
    b[threadIdx.x+ndata*out1+(i-1)*blockDim.x]=tmp1;
    int ii=threadIdx.x+ndata*out2+(i-1)*blockDim.x-my_local_ind;
    if (ii>=0)
      b[ii]=tmp2;
    ii=threadIdx.x+ndata*out3+(i-1)*blockDim.x-2*my_local_ind;    
    if (ii>0)
      b[ii]=tmp3;
    
    ii=threadIdx.x+ndata*out4+(i-1)*blockDim.x-3*my_local_ind;    
    if (ii>0)
      b[ii]=tmp4;
    __syncthreads();
    s1[threadIdx.x]=s1[threadIdx.x+blockDim.x];
    s2[threadIdx.x]=s2[threadIdx.x+blockDim.x];
    s3[threadIdx.x]=s3[threadIdx.x+blockDim.x];
    s4[threadIdx.x]=s4[threadIdx.x+blockDim.x];       
    __syncthreads();
  }
  
  float tmp1=s1[threadIdx.x]+s2[threadIdx.x]+s3[threadIdx.x]+s4[threadIdx.x];
  float tmp2=s1[threadIdx.x]+s2[threadIdx.x]+s3[threadIdx.x+1]+s4[threadIdx.x+1];
  float tmp3=s1[threadIdx.x]+s2[threadIdx.x+1]+s3[threadIdx.x+1]+s4[threadIdx.x+2];
  float tmp4=s1[threadIdx.x]+s2[threadIdx.x+1]+s3[threadIdx.x+2]+s4[threadIdx.x+3];
  b[threadIdx.x+ndata*out1+(nchunk-1)*blockDim.x]=tmp1;
  if (threadIdx.x<blockDim.x-1)
    b[threadIdx.x+ndata*out2+(nchunk-1)*blockDim.x-my_local_ind]=tmp2;
  if (threadIdx.x<blockDim.x-2)
    b[threadIdx.x+ndata*out3+(nchunk-1)*blockDim.x-2*my_local_ind]=tmp3;
  if (threadIdx.x<blockDim.x-3)
    b[threadIdx.x+ndata*out4+(nchunk-1)*blockDim.x-3*my_local_ind]=tmp4;
  __syncthreads();
}

/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
__global__ void dedisperse_kernel_test2(float *a, float *b, int nchan, int ndata, int cursize)
{

  __shared__ float s1[2*THREADS_PER_BLOCK];
  __shared__ float s2[2*THREADS_PER_BLOCK];
  int nchunk=ndata/blockDim.x;

  int curchunk=(2*blockIdx.x)/cursize;
  int my_local_ind=blockIdx.x-(curchunk*cursize/2);
  int in1=cursize*curchunk+2*my_local_ind;
  int out1=cursize*curchunk+my_local_ind;
  int out2=cursize*curchunk+my_local_ind+cursize/2;
  s1[threadIdx.x]=a[threadIdx.x+ndata*(in1)];
  s2[threadIdx.x]=a[threadIdx.x+ndata*(in1+1)];

  

  for (int i=1;i<nchunk;i++) {
    s1[threadIdx.x+blockDim.x]=a[threadIdx.x+ndata*in1+i*blockDim.x];
    s2[threadIdx.x+blockDim.x]=a[threadIdx.x+ndata*(in1+1)+i*blockDim.x];
    __syncthreads();
    b[threadIdx.x+ndata*(out1)+(i-1)*blockDim.x]=s1[threadIdx.x]+s2[threadIdx.x];
    if (threadIdx.x+(i-1)*blockDim.x>=my_local_ind) //cast it this way since compiler is making stuff unsigned
      b[threadIdx.x+ndata*(out2)+(i-1)*blockDim.x-my_local_ind]=s1[threadIdx.x]+s2[threadIdx.x+1];

    //b[threadIdx.x+ndata*(out1)+(i-1)*blockDim.x]=1.0;
    //b[threadIdx.x+ndata*(out2)+(i-1)*blockDim.x]=1.0;

    __syncthreads();

    s1[threadIdx.x]=s1[threadIdx.x+blockDim.x];
    s2[threadIdx.x]=s2[threadIdx.x+blockDim.x];
    __syncthreads();
  }
  
  b[threadIdx.x+ndata*(out1)+(nchunk-1)*blockDim.x]=s1[threadIdx.x]+s2[threadIdx.x];
  //if (threadIdx.x+(nchunk-1)*blockDim.x-my_local_ind>=0)
  if (threadIdx.x<blockDim.x-1)
    b[threadIdx.x+ndata*(out2)+(nchunk-1)*blockDim.x-my_local_ind]=s1[threadIdx.x]+s2[threadIdx.x+1];
  __syncthreads();

  //b[threadIdx.x+ndata*(out1)+(nchunk-1)*blockDim.x]=1.0;
  //b[threadIdx.x+ndata*(out2)+(nchunk-1)*blockDim.x]=1.0;
}

/*--------------------------------------------------------------------------------*/
__global__ void dedisperse_kernel_test(float *a, float *b, int nchan, int ndata, int cursize)
{
#if 1
#else
  int mystart=2*blockIdx.x/cursize;
  mystart=mystart*cursize;
  int mychan0=(2*blockIdx.x-mystart)/2; //This is the local post-combine channel, also the lag
  //b[blockIdx.x*ndata+threadIdx.x]=mychan0;
  //b[blockIdx.x*ndata+threadIdx.x]=mystart;

  __shared__ float s1[2*THREADS_PER_BLOCK];
  __shared__ float s2[2*THREADS_PER_BLOCK];
  s1[threadIdx.x]=a[threadIdx.x+ndata*(2*blockIdx.x)];
  s2[threadIdx.x]=a[threadIdx.x+ndata*(2*blockIdx.x+1)];
  
  int nchunk=ndata/blockDim.x;
  for (int i=1;i<nchunk;i++) {
    s1[threadIdx.x+blockDim.x]=a[threadIdx.x+i*blockDim.x+ndata*(2*blockIdx.x)];
    s2[threadIdx.x+blockDim.x]=a[threadIdx.x+i*blockDim.x+ndata*(2*blockIdx.x+1)];
    __syncthreads();
    
    b[threadIdx.x+(i-1)*THREADS_PER_BLOCK+ndata*(mystart+mychan0)+1]=s1[threadIdx.x]+s2[threadIdx.x+1];

    
    
    if (threadIdx.x+(i-1)*THREADS_PER_BLOCK+mychan0<ndata) {
      //////b[threadIdx.x+(i-1)*THREADS_PER_BLOCK+mychan0+ndata*(mystart+mychan0+cursize/2)]=s1[threadIdx.x]+s2[threadIdx.x+1];

      b[threadIdx.x+(i-1)*THREADS_PER_BLOCK+mychan0+ndata*(mystart+mychan0+cursize/2)+1]=s1[threadIdx.x+1]+s2[threadIdx.x];
      

      //////b[threadIdx.x+(i-1)*THREADS_PER_BLOCK+mychan0+ndata*(mystart+mychan0+cursize/2)]=mychan0;
    }
    s1[threadIdx.x]=s1[threadIdx.x+blockDim.x];
    s2[threadIdx.x]=s2[threadIdx.x+blockDim.x];
      
    __syncthreads();
  }
  int i=nchunk;
  //do the last block
  b[threadIdx.x+(i-1)*THREADS_PER_BLOCK+ndata*(mystart+mychan0)]=s1[threadIdx.x]+s2[threadIdx.x];  
  if (threadIdx.x+(i-1)*THREADS_PER_BLOCK+mychan0<ndata)
    b[threadIdx.x+(i-1)*THREADS_PER_BLOCK+mychan0+ndata*(mystart+mychan0+cursize/2)]=s1[threadIdx.x]+s2[threadIdx.x+1];

#endif
}

/*--------------------------------------------------------------------------------*/
__global__ void dedisperse_kernel_cap(float *a, float *b, int nchan, int ndata, int cursize)
//expect fewer thread blocks than nchan/2
{
  int nloop=nchan/2/gridDim.x;
  for (int ii=0;ii<nloop;ii++) {
    int blockx=blockIdx.x+ii*gridDim.x;

    
    int mystart=2*blockx/cursize;
    mystart=mystart*cursize;
    int mychan0=(2*blockx-mystart)/2; //This is the local post-combine channel, also the lag
    //b[blockIdx.x*ndata+threadIdx.x]=mychan0;
    //b[blockIdx.x*ndata+threadIdx.x]=mystart;
    
    __shared__ float s1[2*THREADS_PER_BLOCK];
    __shared__ float s2[2*THREADS_PER_BLOCK];
    s1[threadIdx.x]=a[threadIdx.x+ndata*(2*blockx)];
    s2[threadIdx.x]=a[threadIdx.x+ndata*(2*blockx+1)];
    int nchunk=ndata/blockDim.x;
    for (int i=1;i<nchunk;i++) {
      s1[threadIdx.x+blockDim.x]=a[threadIdx.x+i*blockDim.x+ndata*(2*blockx)];
      s2[threadIdx.x+blockDim.x]=a[threadIdx.x+i*blockDim.x+ndata*(2*blockx+1)];
      __syncthreads();
      
      b[threadIdx.x+(i-1)*THREADS_PER_BLOCK+ndata*(mystart+mychan0)+1]=s1[threadIdx.x]+s2[threadIdx.x+1];
      
      
      
      if (threadIdx.x+(i-1)*THREADS_PER_BLOCK+mychan0<ndata) {
	//b[threadIdx.x+(i-1)*THREADS_PER_BLOCK+mychan0+ndata*(mystart+mychan0+cursize/2)]=s1[threadIdx.x]+s2[threadIdx.x+1];
	b[threadIdx.x+(i-1)*THREADS_PER_BLOCK+mychan0+ndata*(mystart+mychan0+cursize/2)+1]=s1[threadIdx.x+1]+s2[threadIdx.x];
	//b[threadIdx.x+(i-1)*THREADS_PER_BLOCK+mychan0+ndata*(mystart+mychan0+cursize/2)]=mychan0;
      }
      s1[threadIdx.x]=s1[threadIdx.x+blockDim.x];
      s2[threadIdx.x]=s2[threadIdx.x+blockDim.x];
      
      __syncthreads();
    }
    int i=nchunk;
    //do the last block
    b[threadIdx.x+(i-1)*THREADS_PER_BLOCK+ndata*(mystart+mychan0)]=s1[threadIdx.x]+s2[threadIdx.x];  
    if (threadIdx.x+(i-1)*THREADS_PER_BLOCK+mychan0<ndata)
      b[threadIdx.x+(i-1)*THREADS_PER_BLOCK+mychan0+ndata*(mystart+mychan0+cursize/2)]=s1[threadIdx.x]+s2[threadIdx.x+1];
    
    
  } 
}
/*--------------------------------------------------------------------------------*/
__global__ void dedisperse_kernel(float *a, float *b, int nchan, int ndata)
{
  return;  //thiscode don't fly
  __shared__ float s1[2*THREADS_PER_BLOCK];
  __shared__ float s2[2*THREADS_PER_BLOCK];
  s1[threadIdx.x]=a[threadIdx.x+ndata*(2*blockIdx.x)];
  s2[threadIdx.x]=a[threadIdx.x+ndata*(2*blockIdx.x+1)];
  __syncthreads();
  int nchunk=ndata/blockDim.x;
  for (int i=1;i<nchunk;i++) {
    s1[threadIdx.x+blockDim.x]=a[threadIdx.x+i*blockDim.x+ndata*(2*blockIdx.x)];
    s2[threadIdx.x+blockDim.x]=a[threadIdx.x+i*blockDim.x+ndata*(2*blockIdx.x+1)];
    __syncthreads();
    
    
    
    b[threadIdx.x+(i-1)*THREADS_PER_BLOCK+ndata*(blockIdx.x)]=s1[threadIdx.x]+s2[threadIdx.x];
    b[threadIdx.x+(i-1)*THREADS_PER_BLOCK+ndata*(blockIdx.x+nchan/2)]=s1[threadIdx.x]+s2[threadIdx.x+1];
    

    s1[threadIdx.x]=s1[threadIdx.x+blockDim.x];
    s2[threadIdx.x]=s2[threadIdx.x+blockDim.x];
    __syncthreads();  //not sure if I need this, probably not but doesn't slow things very much
    

  }
  
  b[threadIdx.x+(nchunk-1)*THREADS_PER_BLOCK+ndata*(blockIdx.x)]=s1[threadIdx.x]+s2[threadIdx.x];
  if (threadIdx.x<blockDim.x-1)
    b[threadIdx.x+(nchunk-1)*THREADS_PER_BLOCK+ndata*(blockIdx.x+nchan/2)]=s1[threadIdx.x]+s2[threadIdx.x+1];


  //b[threadIdx.x+ndata*blockIdx.x]=nchunk;
}

/*================================================================================*/



int main(int argc, char *argv[])
{
  int ndata=2048;
  int depth=14;
  if (argc>1)
    ndata=atoi(argv[1]);
  if (argc>2)
    depth=atoi(argv[2]);		      

#if 0
  int nchan=1;
  for (int i=0;i<depth;i++)
    nchan*=2;
#else
  int nchan=1<<depth;
#endif
  printf("nchan is %d %d\n",nchan,1<<depth);

  printf("dimensions are %d %d %d\n",ndata,depth,nchan);
  float **mat=matrix(nchan,ndata);
  
  for (int i=0;i<nchan;i++)
    for (int j=0;j<ndata;j++) {
      mat[i][j]=0;
    }
  if (1)
    for (int i=0;i<nchan;i++) {
      //mat[i][ndata/2+150-(int)(0.43*i+0.0)]=1;
      //mat[i][nchan-(int)(0.43*i+0.0)]=1;
      //mat[i][2*nchan+(i+1)/2]=1;
    }
  for (int i=0;i<nchan;i++)
    for (int j=0;j<ndata;j++)
      mat[i][j]++;

  if (0)
    for (int i=0;i<ndata;i++)
      mat[nchan-1][i]=i+1;

#ifdef DO_OMP
  double t1=omp_get_wtime();
#endif
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float *d_mat=mat2dev(mat,nchan,ndata);
  float *d2=dvec(nchan*ndata);
  
  cudaDeviceSynchronize();
#ifdef DO_OMP
  double t2=omp_get_wtime();
  printf("delay is %12.4f\n",t2-t1);
#else
  printf("don't have delay measured.\n");
#endif

  //  vector_add<<<nchan,THREADS_PER_BLOCK>>>(d_mat,d2,nchan,ndata);

#ifdef DO_OMP
  t1=omp_get_wtime();
#endif

  cudaDeviceSynchronize();
  //int curchan=nchan/2;
  //int curchan=2;


  cudaEventRecord(start);
  

  for (int iloop=0;iloop<10;iloop++) {
    printf("doing loop %d\n",iloop);
    int curchan=nchan;
    while (curchan>1)
      {
	if (curchan>2) {
	  if (curchan>4) {
	  //if (0) {
	    //printf("dedispersing 3-pass.\n");
	    dedisperse_kernel_3pass<<<nchan/8,THREADS_PER_BLOCK>>>(d_mat,d2,nchan,ndata,curchan);
	    curchan/=8;
	  }
	  else {
	    //printf("dedispersing 2-pass.\n");
	    dedisperse_kernel_2pass<<<nchan/4,THREADS_PER_BLOCK>>>(d_mat,d2,nchan,ndata,curchan);
	    curchan/=4;
	  }
	}
	else {
	  //printf("dedispersing 1-pass.\n");
	  dedisperse_kernel_test2<<<nchan/2,THREADS_PER_BLOCK>>>(d_mat,d2,nchan,ndata,curchan);
	  curchan/=2;
	}
	float *tmp=d2;
	d2=d_mat;
	d_mat=tmp;
	cudaDeviceSynchronize();
      }
    cudaEventRecord(stop);
  }
  
  //dedisperse_kernel<<<nchan/2,THREADS_PER_BLOCK>>>(d_mat,d2,nchan,ndata);
  //dedisperse_kernel<<<nchan/2,THREADS_PER_BLOCK>>>(d2,d_mat,nchan,ndata);
  //dedisperse_kernel<<<nchan/2,THREADS_PER_BLOCK>>>(d_mat,d2,nchan,ndata);
  //dedisperse_kernel_test<<<nchan/2,THREADS_PER_BLOCK>>>(d_mat,d2,nchan,ndata,16);
  
  //cudaDeviceSynchronize();
#ifdef DO_OMP
  t2=omp_get_wtime();
  printf("kernel time is %12.4f\n",t2-t1);
#else
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("kernel has finishedin %12.4f milliseconds.\n",milliseconds);
#endif
  float **mat2=dev2mat(d_mat,nchan,ndata);
  float **mat3=dev2mat(d2,nchan,ndata);

  printf("elements are %12.4f %12.4f\n",mat[5][5],mat2[5][ndata-50]);

  float tot=0;
  float tot2=0;
  for (int i=0;i<nchan;i++)
    for (int j=0;j<ndata;j++) {
      tot+=mat2[i][j];
      tot2+=mat3[i][j];
    }
  printf("sums are %12.4g %12.4g\n",tot,tot2);

  FILE *outfile=fopen("test_out_2pass.dat","w");
  fwrite(&ndata,1,sizeof(int),outfile);
  fwrite(&nchan,1,sizeof(int),outfile);

  fwrite(mat2[0],ndata*nchan,sizeof(float),outfile);
  fclose(outfile);

#if 0
  for (int i=0;i<16;i++) {
    for (int j=0;j<16;j++)
      printf("%4.0f ",mat2[0+i*16+j][6000]);
    printf("\n");
  }

#endif

  cudaFree( d_mat );
  cudaFree(d2);

  return 0;
}

#endif
