//
// File:       hello.c
//
// Abstract:   A simple "Hello World" compute example showing basic usage of OpenCL which
//             calculates the mathematical square (X[i] = pow(X[i],2)) for a buffer of
//             floating point values.
//             
//
// Version:    <1.0>
//
// Disclaimer: IMPORTANT:  This Apple software is supplied to you by Apple Inc. ("Apple")
//             in consideration of your agreement to the following terms, and your use,
//             installation, modification or redistribution of this Apple software
//             constitutes acceptance of these terms.  If you do not agree with these
//             terms, please do not use, install, modify or redistribute this Apple
//             software.
//
//             In consideration of your agreement to abide by the following terms, and
//             subject to these terms, Apple grants you a personal, non - exclusive
//             license, under Apple's copyrights in this original Apple software ( the
//             "Apple Software" ), to use, reproduce, modify and redistribute the Apple
//             Software, with or without modifications, in source and / or binary forms;
//             provided that if you redistribute the Apple Software in its entirety and
//             without modifications, you must retain this notice and the following text
//             and disclaimers in all such redistributions of the Apple Software. Neither
//             the name, trademarks, service marks or logos of Apple Inc. may be used to
//             endorse or promote products derived from the Apple Software without specific
//             prior written permission from Apple.  Except as expressly stated in this
//             notice, no other rights or licenses, express or implied, are granted by
//             Apple herein, including but not limited to any patent rights that may be
//             infringed by your derivative works or by other works in which the Apple
//             Software may be incorporated.
//
//             The Apple Software is provided by Apple on an "AS IS" basis.  APPLE MAKES NO
//             WARRANTIES, EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION THE IMPLIED
//             WARRANTIES OF NON - INFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A
//             PARTICULAR PURPOSE, REGARDING THE APPLE SOFTWARE OR ITS USE AND OPERATION
//             ALONE OR IN COMBINATION WITH YOUR PRODUCTS.
//
//             IN NO EVENT SHALL APPLE BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL OR
//             CONSEQUENTIAL DAMAGES ( INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//             SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//             INTERRUPTION ) ARISING IN ANY WAY OUT OF THE USE, REPRODUCTION, MODIFICATION
//             AND / OR DISTRIBUTION OF THE APPLE SOFTWARE, HOWEVER CAUSED AND WHETHER
//             UNDER THEORY OF CONTRACT, TORT ( INCLUDING NEGLIGENCE ), STRICT LIABILITY OR
//             OTHERWISE, EVEN IF APPLE HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright ( C ) 2008 Apple Inc. All Rights Reserved.
//

////////////////////////////////////////////////////////////////////////////////


// original appleness
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
//#include <OpenCL/opencl.h>
#include <CL/opencl.h>
#include <CL/cl_ext.h>
#include <omp.h>
#include "opencl_frb.h"
////////////////////////////////////////////////////////////////////////////////

// Use a static data size for simplicity
//

//#define DATA_SIZE (1024)

////////////////////////////////////////////////////////////////////////////////

// Simple compute kernel which computes the square of an input array 
//
const char *KernelSource = "\n" \
"__kernel void square(                                                       \n" \
"   __global float* input,                                              \n" \
"   __global float* output,                                             \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       output[i] = input[i] * input[i];                                \n" \
"}                                                                      \n" \
"\n";

////////////////////////////////////////////////////////////////////////////////



/*--------------------------------------------------------------------------------*/
char *read_file(char *fname)
{
  FILE *infile=fopen(fname,"r");
  if (!infile) {
    fprintf(stderr,"%s not available for reading.\n",fname);
    exit(-1);
  }
  fseek(infile, 0, SEEK_END);
  int nbytes = ftell(infile);
  rewind(infile);
  char *buf=(char *)malloc((nbytes+1)*sizeof(char));
  int sizeRead = fread(buf, sizeof(char),nbytes, infile);
  if (sizeRead==nbytes) {
    printf("successfully read %d bytes from %s\n",sizeRead,fname);
    return buf;
  }
  printf("something strange happened reading %s with sizes %d %d\n",fname,sizeRead,nbytes);
  free(buf);
  return NULL;
}
/*--------------------------------------------------------------------------------*/

float **matrix(int n, int  m)
{
  float *vec=(float *)malloc(n*m*sizeof(float));
  float **mat=(float **)malloc(n*sizeof(float *));
  for (int i=0;i<n;i++)
    mat[i]=vec+i*m;
  return mat;
}
/*================================================================================*/

int main(int argc, char** argv)
{
    int err;                            // error code returned from api calls
      
    unsigned int correct;               // number of correct results returned

    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation

    cl_device_id device_id;             // compute device id 
    //cl_device_id device_id[5];             // compute device id 
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
    
    cl_mem input;                       // device memory used for the input array
    cl_mem output;                      // device memory used for the output array


  int ndata=2048;
  int depth=3;
  if (argc>1)
    ndata=atoi(argv[1]);
  if (argc>2)
    depth=atoi(argv[2]);		      

  int nchan=1<<depth;

  int DATA_SIZE=ndata*nchan;


  float **mat=matrix(nchan,ndata);
  float *data=mat[0];
  float **mat2=matrix(nchan,ndata);
  float *results=mat2[0];

  //float data[DATA_SIZE];              // original data set given to device
  //float results[DATA_SIZE];           // results returned from device



  
  // Fill our data set with random float values
  //

  memset(data,0,sizeof(float)*DATA_SIZE);
  memset(mat2[0],0,sizeof(float)*DATA_SIZE);
  
  //for (int i=0;i<nchan;i++)
  //for (int j=0;j<ndata;j++)
  //  mat[i][j]=1.0;

  for (int i=0;i<nchan;i++) {
    mat[i][ndata/2+150+(int)(0.43*i+0.0)]=0.0;
    //mat[i][ndata/2+150]=1;
  }
  unsigned int count = DATA_SIZE;
  //for(i = 0; i < count; i++)
  //data[i] = rand() / (float)RAND_MAX;
    
  cl_platform_id platform;
  clGetPlatformIDs( 1, &platform, NULL );
  

  
  // Connect to a compute device
  //
  int gpu = 1;
  err = clGetDeviceIDs(platform, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
  if (err != CL_SUCCESS)
    {
      printf("Error: Failed to create a device group!\n");
      return EXIT_FAILURE;
    }
  else 
    printf("passed.\n");
  
  // Create a compute context 
  //
  context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
  if (!context)
    {
      printf("Error: Failed to create a compute context!\n");
      return EXIT_FAILURE;
    }
  
  // Create a command commands
  //
  commands = clCreateCommandQueue(context, device_id, 0, &err);
  if (!commands)
    {
      printf("Error: Failed to create a command commands!\n");
      return EXIT_FAILURE;
    }
  
  // Create the compute program from the source buffer
  //
  char *myprog=read_file("simple_dedisperse_3pass.cl");
  //program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
  program = clCreateProgramWithSource(context, 1,(const char **)(&myprog), NULL, &err);
  if (!program)
    {
      printf("Error: Failed to create compute program!\n");
      return EXIT_FAILURE;
    }

    // Build the program executable
    //
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }

    // Create the compute kernel in the program we wish to run
    //
    //kernel = clCreateKernel(program, "square", &err);

    //kernel = clCreateKernel(program, "dedisperse_1pass", &err);
    kernel = clCreateKernel(program, "dedisperse_kernel_3pass", &err);

    if (!kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }
    cl_kernel kernel2 = clCreateKernel(program, "dedisperse_kernel_3pass", &err);
    cl_kernel kernel3 = clCreateKernel(program, "dedisperse_kernel_3pass", &err);
    if (!kernel2 || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel2!\n");
        exit(1);
    }
    

    // Create the input and output arrays in device memory for our calculation
    //
    input = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, NULL);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, NULL);
    if (!input || !output)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }    
    
    // Write our data set into the input array in device memory 
    //
    err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(float) * count, data, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }

    // Set the arguments to our compute kernel
    //
    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &nchan);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &ndata);
    int curchan=nchan;
    err |= clSetKernelArg(kernel, 4, sizeof(unsigned int), &curchan);

    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }

    err = 0;
    err  = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel2, 1, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel2, 2, sizeof(unsigned int), &nchan);
    err |= clSetKernelArg(kernel2, 3, sizeof(unsigned int), &ndata);
    curchan=nchan/8;
    err |= clSetKernelArg(kernel2, 4, sizeof(unsigned int), &curchan);



    err = 0;
    err  = clSetKernelArg(kernel3, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel3, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel3, 2, sizeof(unsigned int), &nchan);
    err |= clSetKernelArg(kernel3, 3, sizeof(unsigned int), &ndata);
    curchan=nchan/64;
    err |= clSetKernelArg(kernel3, 4, sizeof(unsigned int), &curchan);

    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel2 arguments! %d\n", err);
        exit(1);
    }

    // Get the maximum work group size for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }

    err = clGetKernelWorkGroupInfo(kernel2, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    err = clGetKernelWorkGroupInfo(kernel3, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel2 work group info! %d\n", err);
        exit(1);
    }

    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    //global = count;
    //local = 256;
    local = THREADS_PER_BLOCK;
    global = nchan/8*local;
    //printf("global is %d\n",global);
    
    double t1=omp_get_wtime();

    for (int i=0;i<50*depth;i++) {
      for (int i=0;i<1;i++) {
	err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	if (err) {
	  printf("Error: Failed to execute kernel!\n");
	  return EXIT_FAILURE;
	}
	
#if 0
	err = clEnqueueNDRangeKernel(commands, kernel2, 1, NULL, &global, &local, 0, NULL, NULL);
	if (err) {
	  printf("Error: Failed to execute kernel2!\n");
	  return EXIT_FAILURE;
	}
	
	err = clEnqueueNDRangeKernel(commands, kernel3, 1, NULL, &global, &local, 0, NULL, NULL);
	if (err) {
	  printf("Error: Failed to execute kernel3!\n");
	  return EXIT_FAILURE;
      }
#endif
      }
    }
    // Wait for the command commands to get serviced before reading back results
    //


    clFinish(commands);
    double t2=omp_get_wtime();
    printf("kernel took %12.6f seconds.\n",t2-t1);

    // Read back the results from the device to verify the output
    //
    err = clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(float) * count, results, 0, NULL, NULL );  
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }
    
    // Validate our results
    //
    correct = 0;
    for(int i = 0; i < count; i++)
    {
        if(results[i] == data[i] * data[i])
            correct++;
    }


#if 0
    FILE *outfile=fopen("test_out_2pass.dat","w");
    fwrite(&ndata,1,sizeof(int),outfile);
    fwrite(&nchan,1,sizeof(int),outfile);
    
    fwrite(mat2[0],ndata*nchan,sizeof(float),outfile);
    fclose(outfile);

    outfile=fopen("test_in_2pass.dat","w");
    fwrite(&ndata,1,sizeof(int),outfile);
    fwrite(&nchan,1,sizeof(int),outfile);
    
    fwrite(mat[0],ndata*nchan,sizeof(float),outfile);
    fclose(outfile);
#endif

    
    // Print a brief summary detailing the results
    //
    printf("Computed '%d/%d' correct values!\n", correct, count);
    
    // Shutdown and cleanup
    //
    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    return 0;
}

