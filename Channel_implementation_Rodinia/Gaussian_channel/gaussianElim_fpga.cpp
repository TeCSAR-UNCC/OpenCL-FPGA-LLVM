#include "gaussianElim_fpga.h"
#include <cstdlib>
#include <iostream>
#include <string>
#include <cstring>
#include "timer.h"
#include <stdio.h>
#include <CL/opencl.h>
//#include "CLHelper_fpga.h"
#include "util.h"
#include "./util/opencl/opencl.h"

double timestamp(){
    double ms=0.0;
    timeval time;
    gettimeofday(&time,NULL);
    ms=(time.tv_sec*1000.0)+(time.tv_usec/1000.0);
    return ms;
} 


#ifdef RD_WG_SIZE_0_0
        #define BLOCK_SIZE_0 RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define BLOCK_SIZE_0 RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define BLOCK_SIZE_0 RD_WG_SIZE
#else
        #define BLOCK_SIZE_0 0
#endif

//2D defines. Go from specific to general                                                
#ifdef RD_WG_SIZE_1_0
        #define BLOCK_SIZE_1_X RD_WG_SIZE_1_0
#elif defined(RD_WG_SIZE_1)
        #define BLOCK_SIZE_1_X RD_WG_SIZE_1
#elif defined(RD_WG_SIZE)
        #define BLOCK_SIZE_1_X RD_WG_SIZE
#else
        #define BLOCK_SIZE_1_X 0
#endif

#ifdef RD_WG_SIZE_1_1
        #define BLOCK_SIZE_1_Y RD_WG_SIZE_1_1
#elif defined(RD_WG_SIZE_1)
        #define BLOCK_SIZE_1_Y RD_WG_SIZE_1
#elif defined(RD_WG_SIZE)
        #define BLOCK_SIZE_1_Y RD_WG_SIZE
#else
        #define BLOCK_SIZE_1_Y 0
#endif
cl_context context=NULL;
//unsigned num_devices = 0;//////////////////////////////////////////////********************************************
void Usage(int argc, char**argv){
		fprintf(stderr,"Usage: %s <input_file>\n", argv[0]);
}
float eventTime(cl_event event,cl_command_queue command_queue){
    cl_int error=0;
    cl_ulong eventStart,eventEnd;
    clFinish(command_queue);
    error = clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&eventStart,NULL);
    cl_errChk(error,"ERROR in Event Profiling.",true); 
    error = clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&eventEnd,NULL);
    cl_errChk(error,"ERROR in Event Profiling.",true);
    return (float)((eventEnd-eventStart)/1e9);
}

void create_matrix(float *m, int size){
  int i,j;
  float lamda = -0.01;
  float coe[2*size-1];
  float coe_i =0.0;
  for (i=0; i < size; i++)
    {
      coe_i = 10*exp(lamda*i); 
      j=size-1+i;     
      coe[j]=coe_i;
      j=size-1-i;     
      coe[j]=coe_i;
    }
  for (i=0; i < size; i++) {
      for (j=0; j < size; j++) {
	m[i*size+j]=coe[size-1-i+j];
      }
  }
}

int main(int argc, char * argv[])
{
	printf("WG size of kernel 1 = %d, WG size of kernel 2= %d X %d\n", BLOCK_SIZE_0, BLOCK_SIZE_1_X, BLOCK_SIZE_1_Y);
    float *a=NULL, *b=NULL, *finalVec=NULL;
    float *m=NULL;
    int size = -1;
    FILE *fp;
    // args
    char *filename;
	int quiet=1,timing=0,platform=-1,device=-1;
	cl_context context = cl_init_context(platform,device,quiet);
	filename = argv[1];
	if(size < 1){
		fp = fopen(filename, "r");
		fscanf(fp, "%d", &size);

		a = (float *) malloc(size * size * sizeof(float));
		InitMat(fp,size, a, size, size);

		b = (float *) malloc(size * sizeof(float));
		InitAry(fp, b, size);

		fclose(fp);
	}
	else{
		printf("create input internally before create, size = %d \n", size);
		a = (float *) malloc(size * size * sizeof(float));
		create_matrix(a, size);

		b = (float *) malloc(size * sizeof(float));
		for (int i =0; i< size; i++)
		  b[i]=1.0;
    }
    if (!quiet){    
      printf("The input matrix a is:\n");
      PrintMat(a, size, size, size);

      printf("The input array b is:\n");
      PrintAry(b, size);
    }
    // create the solution matrix
    m = (float *) malloc(size * size * sizeof(float));
	 
    // create a new vector to hold the final answer

    finalVec = (float *) malloc(size * sizeof(float));
    
    InitPerRun(size,m);
    	
    // run kernels
	ForwardSub(context,a,b,m,size);
	
	if (!quiet) {
        printf("The result of matrix m is: \n");
        
        PrintMat(m, size, size, size);
        printf("The result of matrix a is: \n");
        PrintMat(a, size, size, size);
        printf("The result of array b is: \n");
        PrintAry(b, size);
        
        BackSub(a,b,finalVec,size);
        printf("The final solution is: \n");
        PrintAry(finalVec,size);
    }
    
    free(m);
    free(a);
    free(b);
    free(finalVec);
    cl_cleanup();
    return 0;
}

void InitPerRun(int size,float *m) 
{
	int i;
	for (i=0; i<size*size; i++)
			*(m+i) = 0.0;
}

// create both matrix and right hand side, Ke Wang 2013/08/12 11:51:06

void BackSub(float *a, float *b, float *finalVec, int size)
{
	// solve "bottom up"
	int i,j;
	for(i=0;i<size;i++){
		finalVec[size-i-1]=b[size-i-1];
		for(j=0;j<i;j++)
		{
			finalVec[size-i-1]-=*(a+size*(size-i-1)+(size-j-1)) * finalVec[size-j-1];
		}
		finalVec[size-i-1]=finalVec[size-i-1]/ *(a+size*(size-i-1)+(size-i-1));
	}
}
void InitMat(FILE *fp, int size, float *ary, int nrow, int ncol)
{
	int i, j;
	
	for (i=0; i<nrow; i++) {
		for (j=0; j<ncol; j++) {
			fscanf(fp, "%f",  ary+size*i+j);
		}
	}  
}
/*------------------------------------------------------
 ** InitAry() -- Initialize the array (vector) by reading
 ** data from the data file
 **------------------------------------------------------
 */
void InitAry(FILE *fp, float *ary, int ary_size)
{
	int i;
	
	for (i=0; i<ary_size; i++) {
		fscanf(fp, "%f",  &ary[i]);
	}
}  
/*------------------------------------------------------
 ** PrintMat() -- Print the contents of the matrix
 **------------------------------------------------------
 */
void PrintMat(float *ary, int size, int nrow, int ncol)
{
	int i, j;
	
	for (i=0; i<nrow; i++) {
		for (j=0; j<ncol; j++) {
			printf("%8.2e ", *(ary+size*i+j));
		}
		printf("\n");
	}
	printf("\n");
}

/*------------------------------------------------------
 ** PrintAry() -- Print the contents of the array (vector)
 **------------------------------------------------------
 */
void PrintAry(float *ary, int ary_size)
{
	int i;
	for (i=0; i<ary_size; i++) {
		printf("%.2e ", ary[i]);
	}
	printf("\n\n");
}


void ForwardSub(cl_context context,float *a, float *b, float *m, int size) {

    // 1. set up kernel
		cl_kernel gaussian_kernel_read,gaussian_kernel_compute,gaussian_kernel_store;
        cl_int status;
        float writeTime=0, kernel_read=0, kernel_compute=0, kernel_store=0, readTime=0;
        cl_program cl_gaussian_program;
        //printf("I am here\n");
        cl_gaussian_program = cl_compileProgram((char *)"./binary/gaussian_default_read.aocx",NULL);
	    //printf("I am passed\n");   
        //~ gaussian_kernel_1 = clCreateKernel(cl_gaussian_program, "Fan1", &status);
        //~ status = cl_errChk(status, (char *)"Error Creating Gaussian Fan1 kernel",true);
        //~ if(status)exit(1);
        
        gaussian_kernel_read = clCreateKernel(cl_gaussian_program, "Fan2_Read", &status);
        status = cl_errChk(status, (char *)"Error Creating Gaussian kernel_read",true);
        if(status)exit(1);

		gaussian_kernel_compute = clCreateKernel(cl_gaussian_program, "Fan2_CU", &status);
        status = cl_errChk(status, (char *)"Error Creating Gaussian kernel_compute",true);
        if(status)exit(1);
        
                           
		gaussian_kernel_store = clCreateKernel(cl_gaussian_program, "Fan2_store", &status);
        status = cl_errChk(status, (char *)"Error Creating Gaussian kernel_store",true);
        if(status)exit(1);
		
	//printf("I am writing global value\n");
////////////////////////////////////////////////////////////////////////
	// 3. Determine block sizes
    size_t globalWorksizeFan2[2];
    size_t localWorksizeFan2Buf[2]={BLOCK_SIZE_1_X, BLOCK_SIZE_1_Y};
    size_t *localWorksizeFan2=NULL;

        globalWorksizeFan2[0] = size;
        globalWorksizeFan2[1] = size;

        if(localWorksizeFan2Buf[0]){
                localWorksizeFan2=localWorksizeFan2Buf;
                globalWorksizeFan2[0]=(int)ceil(globalWorksizeFan2[0]/(double)localWorksizeFan2Buf[0])*localWorksizeFan2Buf[0];
                globalWorksizeFan2[1]=(int)ceil(globalWorksizeFan2[1]/(double)localWorksizeFan2Buf[1])*localWorksizeFan2Buf[1];
        }   
//////////////////////////////////////////////////////////////////////// 
	
	//printf("I am Creating buffer\n");
	char h_over;
    // 2. set up memory on device and send ipts data to device copy ipts to device
    cl_mem a_dev, b_dev, m_dev;

    cl_int error=0;
    int t;
    a_dev = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*size*size, NULL, &error);
    b_dev = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*size, NULL, &error);
    m_dev = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float) * size * size, NULL, &error);
 
	cl_command_queue command_queue = cl_getCommandQueue();
	cl_command_queue command_queue1 = cl_getCommandQueue1();
	cl_command_queue command_queue2 = cl_getCommandQueue2();
	cl_event writeEvent,kernelEvent,kernelEvent1,kernelEvent2,kernelEvent3,readEvent;

    const double first_stamp=timestamp();

    error = clEnqueueWriteBuffer(command_queue,a_dev,1,0,sizeof(float)*size*size,a,0, NULL,&writeEvent);
    writeTime+=eventTime(writeEvent,command_queue);
    clReleaseEvent(writeEvent);
    error = clEnqueueWriteBuffer(command_queue,b_dev,1,0,sizeof(float)*size,b,0,NULL,&writeEvent);
    writeTime+=eventTime(writeEvent,command_queue);
    clReleaseEvent(writeEvent);
    error = clEnqueueWriteBuffer(command_queue,m_dev,1,0,sizeof(float)*size*size,m,0,NULL,&writeEvent);
    writeTime+=eventTime(writeEvent,command_queue);
    clReleaseEvent(writeEvent);

   status = clFinish(command_queue);
//**************************Compute Kernel*****************************************************
    //printf("I am setting args\n");
    cl_int argchk;
    for (t=0; t<(size-1); t++) {
//~ // 3. send arguments to device
		//~ argchk  = clSetKernelArg(gaussian_kernel_1, 0, sizeof(cl_mem), (void *)&m_dev);
		//~ argchk |= clSetKernelArg(gaussian_kernel_1, 1, sizeof(cl_mem), (void *)&a_dev);
		//~ argchk |= clSetKernelArg(gaussian_kernel_1, 2, sizeof(cl_mem), (void *)&b_dev);
		//~ argchk |= clSetKernelArg(gaussian_kernel_1, 3, sizeof(int), (void *)&size);
		//~ argchk |= clSetKernelArg(gaussian_kernel_1, 4, sizeof(int), (void *)&t);
		//~ printf("Fan1 kernel Starts\n");
		//~ cl_errChk(argchk,"ERROR in Setting Gaussian Fan1 kernel args",true);
//~ // 4. enqueue kernel
		//~ error = clEnqueueNDRangeKernel(command_queue,gaussian_kernel_1,1,0,globalWorksizeFan1,localWorksizeFan1,0, NULL, &kernelEvent);
		//~ cl_errChk(error,"ERROR in Executing Kernel Gaussian_Fan1",true);
		//~ printf("Fan1 kernel Executes\n");
		//~ status = clFinish(command_queue);
	
	// 3. send arguments to device
		argchk  = clSetKernelArg(gaussian_kernel_read, 0, sizeof(cl_mem), (void *)&m_dev);
		argchk |= clSetKernelArg(gaussian_kernel_read, 1, sizeof(cl_mem), (void *)&a_dev);
		argchk |= clSetKernelArg(gaussian_kernel_read, 2, sizeof(cl_mem), (void *)&b_dev);
		argchk |= clSetKernelArg(gaussian_kernel_read, 3, sizeof(int), (void *)&size);
		argchk |= clSetKernelArg(gaussian_kernel_read, 4, sizeof(int), (void *)&t);
		cl_errChk(argchk,"ERROR in Setting Gaussian_Fan2_read kernel args",true);
		//printf("Fan2 Read kernel Starts\n");
		
	// 3. send arguments to device
		argchk  = clSetKernelArg(gaussian_kernel_compute, 0, sizeof(cl_mem), (void *)&b_dev);
        argchk |= clSetKernelArg(gaussian_kernel_compute, 1, sizeof(int), (void *)&size);
        argchk |= clSetKernelArg(gaussian_kernel_compute, 2, sizeof(int), (void *)&t);
		cl_errChk(argchk,"ERROR in Setting Gaussian_fan2_compute kernel args",true);
		//printf("Fan2 Compute kernel Starts\n");
		
	// 3. send arguments to device
		argchk = clSetKernelArg(gaussian_kernel_store, 0, sizeof(cl_mem), (void *)&a_dev);
        argchk |= clSetKernelArg(gaussian_kernel_store, 1, sizeof(cl_mem), (void *)&b_dev);
        argchk |= clSetKernelArg(gaussian_kernel_store, 2, sizeof(int), (void *)&size);
        argchk |= clSetKernelArg(gaussian_kernel_store, 3, sizeof(int), (void *)&t);
		cl_errChk(argchk,"ERROR in Setting Gaussian_fan2_WB kernel args",true);
		//printf("Fan2 Store kernel Starts\n");
	
    // 4. enqueue kernel
        error = clEnqueueNDRangeKernel(command_queue,gaussian_kernel_read,2,0,globalWorksizeFan2,NULL,0, NULL, &kernelEvent);
        cl_errChk(error,"ERROR in Executing Kernel Gaussian_fan2_read",true);
        //printf("Fan2 Read kernel Executes\n");

    // 4. enqueue kernel
        error = clEnqueueNDRangeKernel(command_queue1,gaussian_kernel_compute, 2, 0,globalWorksizeFan2,NULL,0, NULL, &kernelEvent1);
        cl_errChk(error,"ERROR in Executing Kernel Gaussian_fan2_compute",true);
        //printf("Fan2 Compute kernel Executes\n");

    // 4. enqueue kernel
		error = clEnqueueNDRangeKernel(command_queue2,gaussian_kernel_store, 2, 0,globalWorksizeFan2,NULL,0, NULL, &kernelEvent2);
		cl_errChk(error,"ERROR in Executing Kernel Gaussian_fan2_wb",true);
		//printf("Fan2 Store kernel Executes\n");
		
		kernel_read+=eventTime(kernelEvent,command_queue);
		kernel_compute+=eventTime(kernelEvent1,command_queue1);
		kernel_store+=eventTime(kernelEvent2,command_queue2);
		status = clFinish(command_queue2);
	}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////END OF OPERATION

    // 5. transfer data off of device
    error = clEnqueueReadBuffer(command_queue2,a_dev,1,0,sizeof(float) * size * size,a,0,NULL,&readEvent);
	readTime+=eventTime(readEvent,command_queue2);
	clReleaseEvent(readEvent);
	error = clEnqueueReadBuffer(command_queue2,b_dev,1,0,sizeof(float) * size,b,0,NULL,&readEvent);
	readTime+=eventTime(readEvent,command_queue2);
	clReleaseEvent(readEvent);
	error = clEnqueueReadBuffer(command_queue2,m_dev,1,0,sizeof(float) * size * size,m,0,NULL,&readEvent);
	readTime+=eventTime(readEvent,command_queue2);
	clReleaseEvent(readEvent);
	status = clFinish(command_queue2);
	//printf("Read final results from Device\n");
    cl_errChk(error,"ERROR with clEnqueueReadBuffer",true);

    const double second_stamp=timestamp();
    const double time_elapsed=second_stamp - first_stamp;
    printf("\n Total Kernel execution time=%0.3f ms\n", time_elapsed);

    if (1) {
        clFinish(command_queue);
        clFinish(command_queue1);
        clFinish(command_queue2);
        cl_ulong eventStart,eventEnd,totalTime=0,total_kernel_time=0;
        printf("# Write(s)\tKernel_Read(s)\tKernel_Compute(s)\tKernel_Wb(s)\tRead(s)\n");
        // Write Buffer
        printf("%f \t",writeTime);
        // Kernel_read
        printf("%f\t",kernel_read);
        // Kernel_Compute
        printf("%f\t\t",kernel_compute);
        // Kernel_wb
        printf("%f \t",kernel_store);
        // Read Buffer
        printf("%f \t\n",readTime);      
    }
    // 6. return finalized data and release buffers
    clReleaseMemObject(a_dev);
    clReleaseMemObject(b_dev);
    clReleaseMemObject(m_dev);
}

