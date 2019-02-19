
#include <stdio.h>									// (in path known to compiler)	needed by printf
#include <stdlib.h>									// (in path known to compiler)	needed by malloc, free
#include <CL/opencl.h>		
#include "clutils.h"
#include "./main.h"						// (in current path)
#include "util/graphics/graphics.h"				// (in specified path)
#include "util/graphics/resize.h"					// (in specified path)
//#include "srad_fpga.h"
#include "utils.h"
#include "./util/opencl/opencl.h"
//#include "OpenCL.h"
//#include "clutils_fpga.cpp"

cl_context context=NULL;
//unsigned num_devices = 0;//////////////////////////////////////////////********************************************

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
void kernel_fpga_opencl_wrapper( cl_context,fp*,int,int,long,int,fp,long,int*,int*,int*,int*,int,int,int);
int main(int argc, char *argv[]){
  printf("WG size of kernel = %d \n", NUMBER_THREADS);


	// inputs image, input paramenters
	fp* image_ori;																// originalinput image
	int image_ori_rows;
	int image_ori_cols;
	long image_ori_elem;

	// inputs image, input paramenters
	fp* image;															// input image
	int Nr,Nc;													// IMAGE nbr of rows/cols/elements
	long Ne;

	// algorithm parameters
	int niter;																// nbr of iterations
	fp lambda;															// update step size

	// size of IMAGE
	int r1,r2,c1,c2;												// row/col coordinates of uniform ROI
	long NeROI;														// ROI nbr of elements

	// surrounding pixel indicies
	int* iN;
	int* iS;
	int* jE;
	int* jW;    

	// counters
	int iter;   // primary loop
	long i;    // image row
	long j;    // image col

	// memory sizes
	int mem_size_i;
	int mem_size_j;

	//======================================================================================================================================================150
	//	INPUT ARGUMENTS
	//======================================================================================================================================================150
	int quiet=1,timing=0,platform=-1,device=-1;
	cl_context context = cl_init_context(platform,device,quiet);


	if(argc != 5){
		printf("ERROR: wrong number of arguments\n");
		return 0;
	}
	else{
		niter = atoi(argv[1]);
		lambda = atof(argv[2]);
		Nr = atoi(argv[3]);						// it is 502 in the original image
		Nc = atoi(argv[4]);						// it is 458 in the original image
	}

	
	//======================================================================================================================================================150
	// 	READ INPUT FROM FILE
	//======================================================================================================================================================150

	//====================================================================================================100
	// 	READ IMAGE (SIZE OF IMAGE HAS TO BE KNOWN)
	//====================================================================================================100

	image_ori_rows = 502;
	image_ori_cols = 458;
	image_ori_elem = image_ori_rows * image_ori_cols;

	image_ori = (fp*)malloc(sizeof(fp) * image_ori_elem);

	read_graphics(	"./image.pgm",
					image_ori,
					image_ori_rows,
					image_ori_cols,
					1);

	//====================================================================================================100
	// 	RESIZE IMAGE (ASSUMING COLUMN MAJOR STORAGE OF image_orig)
	//====================================================================================================100

	Ne = Nr*Nc;

	image = (fp*)malloc(sizeof(fp) * Ne);

	resize(	image_ori,
				image_ori_rows,
				image_ori_cols,
				image,
				Nr,
				Nc,
				1);

	//====================================================================================================100
	// 	End
	//====================================================================================================100

	
	//======================================================================================================================================================150
	// 	SETUP
	//======================================================================================================================================================150

	// variables
	r1     = 0;											// top row index of ROI
	r2     = Nr - 1;									// bottom row index of ROI
	c1     = 0;											// left column index of ROI
	c2     = Nc - 1;									// right column index of ROI

	// ROI image size
	NeROI = (r2-r1+1)*(c2-c1+1);											// number of elements in ROI, ROI size

	// allocate variables for surrounding pixels
	mem_size_i = sizeof(int) * Nr;											//
	iN = (int *)malloc(mem_size_i) ;										// north surrounding element
	iS = (int *)malloc(mem_size_i) ;										// south surrounding element
	mem_size_j = sizeof(int) * Nc;											//
	jW = (int *)malloc(mem_size_j) ;										// west surrounding element
	jE = (int *)malloc(mem_size_j) ;										// east surrounding element

	// N/S/W/E indices of surrounding pixels (every element of IMAGE)
	for (i=0; i<Nr; i++) {
		iN[i] = i-1;														// holds index of IMAGE row above
		iS[i] = i+1;														// holds index of IMAGE row below
	}
	for (j=0; j<Nc; j++) {
		jW[j] = j-1;														// holds index of IMAGE column on the left
		jE[j] = j+1;														// holds index of IMAGE column on the right
	}

	// N/S/W/E boundary conditions, fix surrounding indices outside boundary of image
	iN[0]    = 0;															// changes IMAGE top row index from -1 to 0
	iS[Nr-1] = Nr-1;														// changes IMAGE bottom row index from Nr to Nr-1 
	jW[0]    = 0;															// changes IMAGE leftmost column index from -1 to 0
	jE[Nc-1] = Nc-1;														// changes IMAGE rightmost column index from Nc to Nc-1

	
	//======================================================================================================================================================150
	// 	KERNEL
	//======================================================================================================================================================150
	
	//printf("I am here /n");
	kernel_fpga_opencl_wrapper(context,image,Nr,Nc,Ne,niter,lambda,NeROI,iN,iS,jE,jW,iter,mem_size_i,mem_size_j);

	//======================================================================================================================================================150
	// 	WRITE OUTPUT IMAGE TO FILE
	//======================================================================================================================================================150

	write_graphics(	"./image_out.pgm",
					image,
					Nr,
					Nc,
					1,
					255);
					
	//======================================================================================================================================================150
	// 	FREE MEMORY
	//======================================================================================================================================================150

	free(image_ori);
	free(image);
	free(iN); 
	free(iS); 
	free(jW); 
	free(jE);
	
}


void kernel_fpga_opencl_wrapper( cl_context context,fp *image,int Nr,int Nc,long Ne,int niter,fp lambda,long NeROI,int *iN,
							int *iS,int *jE,int *jW,int iter,int mem_size_i,int mem_size_j){

int blocks_x;
size_t local_work_size[1];
	local_work_size[0] = NUMBER_THREADS;

	// workgroups
	int blocks_work_size;
	size_t global_work_size[1];
	blocks_x = Ne/(int)local_work_size[0];
	if (Ne % (int)local_work_size[0] != 0){												// compensate for division remainder above by adding one grid
		blocks_x = blocks_x + 1;																	
	}
	blocks_work_size = blocks_x;
	global_work_size[0] = blocks_work_size * local_work_size[0];						// define the number of blocks in the grid

	//printf("max # of workgroups = %d, # of threads/workgroup = %d (ensure that device can handle)\n", (int)(global_work_size[0]/local_work_size[0]), (int)local_work_size[0]);

int mem_size;															// matrix memory size
	mem_size = sizeof(fp) * Ne;												// get the size of float representation of input IMAGE
	
	//****************************App-specific naming convention, binary path*********************
    // 1. set up kernel
				
		cl_kernel extract_kernel_read,extract_kernel_compute,extract_kernel_wb;
        cl_int status;
        float writeTime=0, kernel_Time=0, readTime=0;
        cl_program cl_srad_extract_program;
        
        //cl_srad_extract_program = cl_compileProgram((char *)"./binary/srad_extract_channel_emu.aocx",NULL);
        //printf("I am here$$$$$$$$$ /n");
      	cl_srad_extract_program = cl_compileProgram((char *)"./binary/srad_extract_channel_fpga.aocx",NULL);
       	//cl_srad_extract_program = cl_compileProgram((char *)"./binary/srad_extract_fpga_8.aocx",NULL);
        //cl_srad_extract_program = cl_compileProgram((char *)"./binary/srad_extract_fpga_16.aocx",NULL);
        //cl_srad_extract_program = cl_compileProgram((char *)"./binary/srad_extract_fpga_32.aocx",NULL);
		//cl_srad_extract_program = cl_compileProgram((char *)"./binary/srad_extract_fpga_64.aocx",NULL);
		//cl_srad_extract_program = cl_compileProgram((char *)"./binary/srad_extract_fpga_128.aocx",NULL);
	    //cl_srad_extract_program = cl_compileProgram((char *)"./binary/srad_extract_fpga_256.aocx",NULL);
	       
        extract_kernel_read = clCreateKernel(cl_srad_extract_program, "extract_kernel_read", &status);
        status = cl_errChk(status, (char *)"Error Creating extract_kernel_read",true);
        if(status)exit(1);

		extract_kernel_compute = clCreateKernel(cl_srad_extract_program, "extract_kernel_compute", &status);
        status = cl_errChk(status, (char *)"Error Creating extract_kernel__compute",true);
        if(status)exit(1);
        
                           
		extract_kernel_wb = clCreateKernel(cl_srad_extract_program, "extract_kernel_wb", &status);
        status = cl_errChk(status, (char *)"Error Creating extract_kernel_wb",true);
        if(status)exit(1);
//**********************************************END*************************************************88


	cl_mem d_I;
	cl_int error=0;
	d_I = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size, NULL, &error);
	
	//====================================================================================================100
	// allocate memory for coordinates on DEVICE
	//====================================================================================================100


	cl_command_queue command_queue =  cl_getCommandQueue(); 
	cl_command_queue command_queue1 = cl_getCommandQueue1();
	cl_command_queue command_queue2 = cl_getCommandQueue2();
	   
	cl_event writeEvent,kernelEvent,kernelEvent1,kernelEvent2,readEvent;
//	cl_event writeEvent,kernelEvent,readEvent;
	//======================================================================================================================================================150
	// 	COPY INPUT TO CPU
	//======================================================================================================================================================150

	//====================================================================================================100
	// Image
	//====================================================================================================100

    error = clEnqueueWriteBuffer(command_queue,d_I, 1, 0, mem_size, image, 0, 0,&writeEvent); //Copy the var from hosto to device
    
//====================================================================================================100
	// coordinates
	//====================================================================================================100

	/*error = clEnqueueWriteBuffer(command_queue, d_iN, 1, 0, mem_size_i, iN, 0, 0, &writeEvent);
	
	error = clEnqueueWriteBuffer(command_queue, d_iS, 1, 0, mem_size_i, iS, 0, 0, &writeEvent);
	
	error = clEnqueueWriteBuffer(command_queue, d_jE, 1, 0, mem_size_j, jE, 0, 0, &writeEvent);
	
	error = clEnqueueWriteBuffer(command_queue, d_jW, 1, 0, mem_size_j, jW, 0, 0, &writeEvent);*/
	
	
	//====================================================================================================100
	// End
	//====================================================================================================100


//********************************************END*************************************************************
   
//////////////////////////////////////////////////////////////////////////////////////////////////////////READ STARTS/////



//**************************************App-specific read kernel init and call ********************
    // 3. send arguments to device
    cl_int argchk;
    //argchk  = clSetKernelArg(extract_kernel_read, 0, sizeof(long), (void *) &Ne);
    argchk  = clSetKernelArg(extract_kernel_read, 0, sizeof(cl_mem), (void *) &d_I);
   //printf("Read kernel Starts\n");
    //printf("Extract_read kernel Starts\n");

    //cl_errChk(argchk,"ERROR in Setting Nearest Neighbor_read kernel args",true);
    cl_errChk(argchk,"ERROR in Setting Extract_read kernel args",true);

// 4. enqueue kernel
    error = clEnqueueNDRangeKernel(command_queue,extract_kernel_read, 1, NULL, global_work_size, local_work_size, 0, NULL, &kernelEvent);
	//cl_errChk(error,"ERROR in Executing Kernel NearestNeighbor_read",true);
	//printf("Read kernel Executes\n");
	
	cl_errChk(error,"ERROR in Executing Extract kernel",true);
	//printf("Extract kernel read Executes\n");
	
//************************************** END ******************************************************
	


//**************************************App-specific comp kernel init and call **************
    argchk  = clSetKernelArg(extract_kernel_compute, 0, sizeof(long), (void *) &Ne);
    //argchk  = clSetKernelArg(extract_kernel_compute, 1, sizeof(cl_mem), (void *) &d_I);
	//printf("Compute kernel Starts\n");
    cl_errChk(argchk,"ERROR in Setting Extract_compute kernel args",true);

    error = clEnqueueNDRangeKernel(command_queue1,extract_kernel_compute, 1, NULL, global_work_size, local_work_size, 0, NULL, &kernelEvent1);
	//printf("Compute kernel Executes\n");
    cl_errChk(error,"ERROR in Executing Kernel Extract_compute",true);
  
//************************************** END ******************************************************



//**************************************App-specific write kernel init and call **************
	//argchk  = clSetKernelArg(extract_kernel_wb, 0, sizeof(long), (void *) &Ne);
    argchk  = clSetKernelArg(extract_kernel_wb, 0, sizeof(cl_mem), (void *) &d_I);

    //printf("Writeback kernel Starts\n");
    cl_errChk(argchk,"ERROR in Setting Extract_wb kernel args",true);
    error = clEnqueueNDRangeKernel(command_queue2,extract_kernel_wb, 1, NULL, global_work_size, local_work_size, 0, NULL, &kernelEvent2);
    //printf("Writeback kernel Executes\n");
    cl_errChk(error,"ERROR in Executing Extract wb kernel",true);

//*************************************END **************
    
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////END OF OPERATION




//**************************************App-specific transfer from device to host **************
    kernel_Time+=eventTime(kernelEvent2,command_queue2);
    status = clFinish(command_queue2);
    error = clEnqueueReadBuffer( command_queue2,d_I, CL_TRUE, 0, mem_size, image, 0, NULL,&readEvent);
    readTime+=eventTime(readEvent,command_queue2);
        //status = clFinish(command_queue2);
	//printf("Read final results from Device\n");
    cl_errChk(error,"ERROR with clEnqueueReadBuffer",true);
//************************************** END ******************************************************    
    
    
    if (1) {
        clFinish(command_queue);
        clFinish(command_queue1);
        clFinish(command_queue2);
        cl_ulong eventStart,eventEnd,totalTime=0,total_kernel_time=0;
        printf("# Write(s)\tKernel_Read(s)\tKernel_Compute(s)\tKernel_Wb(s)\tRead(s)\n");
        // Write Buffer
        printf("%f \t",writeTime);
        // Kernel_read
        printf("%f\t",kernel_Time);
        // Read Buffer
        printf("%f \t\n",readTime);      
    }
    
     
    
//**************************************App-specific clear device mem **************
    error = clReleaseMemObject(d_I);
	
/*	error = clReleaseMemObject(d_c);
	
	error = clReleaseMemObject(d_iN);
	
	error = clReleaseMemObject(d_iS);
	
	error = clReleaseMemObject(d_jE);
	
	error = clReleaseMemObject(d_jW);
	
	error = clReleaseMemObject(d_dN);
	
	error = clReleaseMemObject(d_dS);
	
	error = clReleaseMemObject(d_dE);
	
	error = clReleaseMemObject(d_dW);
	
	error = clReleaseMemObject(d_sums);
	
	error = clReleaseMemObject(d_sums2);*/
		
	//************************************** END ****************************************************** 
}

