#include "hotspot_fpga.h"
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

cl_context context=NULL;
//unsigned num_devices = 0;//////////////////////////////////////////////********************************************
void usage(int argc, char **argv) {
	fprintf(stderr, "Usage: %s <grid_rows/grid_cols> <pyramid_height> <sim_time> <temp_file> <power_file> <output_file>\n", argv[0]);
	fprintf(stderr, "\t<grid_rows/grid_cols>  - number of rows/cols in the grid (positive integer)\n");
	fprintf(stderr, "\t<pyramid_height> - pyramid heigh(positive integer)\n");
	fprintf(stderr, "\t<sim_time>   - number of iterations\n");
	fprintf(stderr, "\t<temp_file>  - name of the file containing the initial temperature values of each cell\n");
	fprintf(stderr, "\t<power_file> - name of the file containing the dissipated power values of each cell\n");
	fprintf(stderr, "\t<output_file> - name of the output file\n");
	exit(1);
}

long long get_time() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (tv.tv_sec * 1000000) + tv.tv_usec;
}

void writeoutput(float *vect, int grid_rows, int grid_cols, char *file) {

	int i,j, index=0;
	FILE *fp;
	char str[STR_SIZE];

	if( (fp = fopen(file, "w" )) == 0 )
          printf( "The file was not opened\n" );


	for (i=0; i < grid_rows; i++) 
	 for (j=0; j < grid_cols; j++)
	 {
		 sprintf(str, "%d\t%g\n", index, vect[i*grid_cols+j]);
		 fputs(str,fp);
		 index++;
	 }
		
      fclose(fp);	
}

void readinput(float *vect, int grid_rows, int grid_cols, char *file) {

  	int i,j;
	FILE *fp;
	char str[STR_SIZE];
	float val;

	if( (fp  = fopen(file, "r" )) ==0 )
            fprintf(stderr, "Error: The file was not opened\n");


	for (i=0; i <= grid_rows-1; i++) 
	 for (j=0; j <= grid_cols-1; j++)
	 {
		if (fgets(str, STR_SIZE, fp) == NULL) 
			fprintf(stderr, "Error: Error reading file\n");
		if (feof(fp))
			fprintf(stderr, "Error: not enough lines in file\n");
		//if ((sscanf(str, "%d%f", &index, &val) != 2) || (index != ((i-1)*(grid_cols-2)+j-1)))
		if ((sscanf(str, "%f", &val) != 1))
			fprintf(stderr, "Error: invalid file format\n");
		vect[i*grid_cols+j] = val;
	}

	fclose(fp);	

}



int main(int argc, char * argv[])
{
	int quiet=0,platform=-1,device=-1;
	cl_context context = cl_init_context(platform,device,quiet);
    int grid_rows=0,grid_cols = 0;
    
    float *FilesavingTemp,*FilesavingPower; //,*MatrixOut; 
    char *tfile, *pfile, *ofile;
    
    int total_iterations = 60;
    int pyramid_height = 1; // number of iterations
	
	if (argc < 7)
		usage(argc, argv);
	if((grid_rows = atoi(argv[1]))<=0||(grid_cols = atoi(argv[1]))<=0||(pyramid_height = atoi(argv[2]))<=0||(total_iterations = atoi(argv[3]))<=0)
		usage(argc, argv);
		
	tfile=argv[4];
    pfile=argv[5];
    ofile=argv[6];
	int size=grid_rows*grid_cols;
    
    // --------------- pyramid parameters --------------- 
    int borderCols = (pyramid_height)*EXPAND_RATE/2;
    int borderRows = (pyramid_height)*EXPAND_RATE/2;
    int smallBlockCol = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
    int smallBlockRow = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
    int blockCols = grid_cols/smallBlockCol+((grid_cols%smallBlockCol==0)?0:1);
    int blockRows = grid_rows/smallBlockRow+((grid_rows%smallBlockRow==0)?0:1);

    FilesavingTemp = (float *) malloc(size*sizeof(float));
    FilesavingPower = (float *) malloc(size*sizeof(float));
    if( !FilesavingPower || !FilesavingTemp) // || !MatrixOut)
        fprintf(stderr, "Error: unable to allocate memory\n");
	
	// Read input data from disk
    readinput(FilesavingTemp, grid_rows, grid_cols, tfile);
    readinput(FilesavingPower, grid_rows, grid_cols, pfile);
    run_hotspot_gpu(context, grid_cols, grid_rows, total_iterations, pyramid_height, blockCols, blockRows, borderCols, borderRows, FilesavingTemp, FilesavingPower, ofile);
    return 0;
}

void run_hotspot_gpu(cl_context context, int col, int row, int total_iterations, int num_iterations, int blockCols, int blockRows, int borderCols, int borderRows, float *TempCPU, float *PowerCPU, char *outputfile) {

    // 1. set up kernel
		cl_kernel hotspot_kernel_read,hotspot_kernel_compute, hotspot_kernel_store;
        cl_int status;
        float writeTime=0, kernel_read=0, kernel_compute=0, kernel_store=0, readTime=0;
        cl_program cl_hotspot_program;
        cl_hotspot_program = cl_compileProgram((char *)"./binary/hotspot_default.aocx",NULL);
	       
        hotspot_kernel_read = clCreateKernel(cl_hotspot_program, "hotspot_read", &status);
        status = cl_errChk(status, (char *)"Error Creating hotspot kernel_read",true);
        if(status)exit(1);

		hotspot_kernel_compute = clCreateKernel(cl_hotspot_program, "hotspot_CU", &status);
        status = cl_errChk(status, (char *)"Error Creating hotspot kernel_compute",true);
        if(status)exit(1);
        
                           
		hotspot_kernel_store = clCreateKernel(cl_hotspot_program, "hotspot_store", &status);
        status = cl_errChk(status, (char *)"Error Creating hotspot kernel_store",true);
        if(status)exit(1);
		
////////////////////////////////////////////////////////////////////////
	float grid_height = chip_height / row;
	float grid_width = chip_width / col;
	int size=row*col;

	float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
	float Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
	float Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
	float Rz = t_chip / (K_SI * grid_height * grid_width);

	float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
	float step = PRECISION / max_slope;
	int t;

	int src = 0, dst = 1;

	size_t global_work_size[2];
	global_work_size[0] = BLOCK_SIZE * blockCols;
	global_work_size[1] = BLOCK_SIZE * blockRows;
	size_t local_work_size[2];
	local_work_size[0] = BLOCK_SIZE;
	local_work_size[1] = BLOCK_SIZE;
//////////////////////////////////////////////////////////////////////// 

	// 2. set up memory on device and send ipts data to device copy ipts to device
    cl_mem MatrixTemp[2], MatrixPower;

    cl_int error=0;

    MatrixTemp[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * size, TempCPU, &error);
    MatrixTemp[1] = clCreateBuffer(context, CL_MEM_READ_WRITE , sizeof(float) * size, NULL, &error);
    MatrixPower = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * size, PowerCPU, &error);
 
	cl_command_queue command_queue = cl_getCommandQueue();
	cl_command_queue command_queue1 = cl_getCommandQueue1();
	cl_command_queue command_queue2 = cl_getCommandQueue2();
	cl_event writeEvent,kernelEvent,kernelEvent1,kernelEvent2,kernelEvent3,readEvent;
    
//////////////////////////////////////////////////////////////////////////////////////////////////////////READ STARTS/////

	long long start_time = get_time();
    //printf("num_iterations = %d\n",num_iterations);
    cl_int argchk;
    for (t = 0; t < total_iterations; t += num_iterations) {
	// 3. send arguments to device
		int iter = MIN(num_iterations, total_iterations - t);
		argchk  = clSetKernelArg(hotspot_kernel_read, 0, sizeof(int), (void *) &iter);
		argchk  = clSetKernelArg(hotspot_kernel_read, 1, sizeof(cl_mem), (void *) &MatrixPower);
		argchk  = clSetKernelArg(hotspot_kernel_read, 2, sizeof(cl_mem), (void *) &MatrixTemp[0]);
		argchk  = clSetKernelArg(hotspot_kernel_read, 3, sizeof(int), (void *) &col);
		argchk  = clSetKernelArg(hotspot_kernel_read, 4, sizeof(int), (void *) &row);
		argchk  = clSetKernelArg(hotspot_kernel_read, 5, sizeof(int), (void *) &borderCols);
		argchk  = clSetKernelArg(hotspot_kernel_read, 6, sizeof(int), (void *) &borderRows);
		cl_errChk(argchk,"ERROR in Setting hotspot kernel args",true);
	
	// 3. send arguments to device	
		argchk  = clSetKernelArg(hotspot_kernel_compute, 0, sizeof(int), (void *) &iter);
		argchk  = clSetKernelArg(hotspot_kernel_compute, 1, sizeof(int), (void *) &col);
		argchk  = clSetKernelArg(hotspot_kernel_compute, 2, sizeof(int), (void *) &row);
		argchk  = clSetKernelArg(hotspot_kernel_compute, 3, sizeof(int), (void *) &borderCols);
		argchk  = clSetKernelArg(hotspot_kernel_compute, 4, sizeof(int), (void *) &borderRows);
		argchk  = clSetKernelArg(hotspot_kernel_compute, 5, sizeof(float), (void *) &Cap);
		argchk  = clSetKernelArg(hotspot_kernel_compute, 6, sizeof(float), (void *) &Rx);
		argchk  = clSetKernelArg(hotspot_kernel_compute, 7, sizeof(float), (void *) &Ry);
		argchk  = clSetKernelArg(hotspot_kernel_compute, 8, sizeof(float), (void *) &Rz);
		argchk  = clSetKernelArg(hotspot_kernel_compute, 9, sizeof(float), (void *) &step);
		cl_errChk(argchk,"ERROR in Setting hotspot kernel args",true);
			
	// 3. send arguments to device	
		argchk  = clSetKernelArg(hotspot_kernel_store, 0, sizeof(int), (void *) &iter);
		argchk  = clSetKernelArg(hotspot_kernel_store, 1, sizeof(cl_mem), (void *) &MatrixTemp[1]);
		argchk  = clSetKernelArg(hotspot_kernel_store, 2, sizeof(int), (void *) &col);
		argchk  = clSetKernelArg(hotspot_kernel_store, 3, sizeof(int), (void *) &row);
		argchk  = clSetKernelArg(hotspot_kernel_store, 4, sizeof(int), (void *) &borderCols);
		argchk  = clSetKernelArg(hotspot_kernel_store, 5, sizeof(int), (void *) &borderRows);
		cl_errChk(argchk,"ERROR in Setting hotspot kernel args",true);

	// 4. enqueue kernel
		error = clEnqueueNDRangeKernel(command_queue, hotspot_kernel_read, 2, 0, global_work_size, local_work_size , 0, NULL, &kernelEvent);
		cl_errChk(error,"ERROR in Executing Hotspot Kernel",true);
		//printf("Read kernel complete\n");


	// 4. enqueue kernel
		error = clEnqueueNDRangeKernel(command_queue1, hotspot_kernel_compute, 2, 0, global_work_size, local_work_size, 0, NULL, &kernelEvent1);
		cl_errChk(error,"ERROR in Executing Hotspot Kernel",true);
		//printf("Compute kernel complete\n");

	
	// 4. enqueue kernel
		error = clEnqueueNDRangeKernel(command_queue2, hotspot_kernel_store, 2, 0, global_work_size, local_work_size, 0, NULL, &kernelEvent2);
		cl_errChk(error,"ERROR in Executing Hotspot Kernel",true);
		//printf("Store kernel complete\n");
		// Swap input and output GPU matrices
		src = 1 - src;
		dst = 1 - dst;
		//printf("SRC = %d , DST = %d\n",src,dst);
    }
		error = clFinish(command_queue);
		error = clFinish(command_queue1);
		error = clFinish(command_queue2);
		//error = clFinish(command_queue);
	//printf("I am here1\n");
    cl_errChk(error,"ERROR in Finishing command queue",true);
    long long end_time = get_time();
    long long total_time = (end_time - start_time);
    printf("\nKernel time: %.3f seconds\n", ((float) total_time) / (1000*1000));
    cl_float *MatrixOut = (cl_float *) clEnqueueMapBuffer(command_queue2, MatrixTemp[src], CL_TRUE, CL_MAP_READ, 0, sizeof(float) * size, 0, NULL, NULL, &error);

    // Write final output to output file
    writeoutput(MatrixOut, row, col, outputfile);
    
    error = clEnqueueUnmapMemObject(command_queue2, MatrixTemp[src], (void *) MatrixOut, 0, NULL, NULL);
    
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////END OF OPERATION

    // 5. transfer data off of device

    clReleaseMemObject(MatrixTemp[0]);
	clReleaseMemObject(MatrixTemp[1]);
	clReleaseMemObject(MatrixPower);
}
