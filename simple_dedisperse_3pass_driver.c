//    Copyright Jonathan Sievers, 2015.  All rights reserved.  This code may only be used with permission of the owner. 

#define THREADS_PER_BLOCK 128


float **matrix(int n, int  m)
{
  float *vec=(float *)malloc(n*m*sizeof(float));
  float **mat=(float **)malloc(n*sizeof(float *));
  for (int i=0;i<n;i++)
    mat[i]=vec+i*m;
  return mat;
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
