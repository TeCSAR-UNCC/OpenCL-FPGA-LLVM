#include "lud.h"
#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <assert.h>
#include "common.h"
#include <sys/time.h>
#include <CL/cl.h>
#include <string.h>
#include <string>
#ifdef RD_WG_SIZE_0_0
        #define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define BLOCK_SIZE RD_WG_SIZE
#else
        #define BLOCK_SIZE 64
	
#endif


cl_context context=NULL;
//unsigned num_devices = 0;//////////////////////////////////////////////********************************************

//static int do_verify = 0;
//void lud_cuda(float *d_m, int matrix_dim);

static struct option long_options[] = {
      /* name, has_arg, flag, val */
      {"input", 1, NULL, 'i'},
      {"size", 1, NULL, 's'},
      {"verify", 0, NULL, 'v'},
      {0,0,0,0}
};


/*float eventTime(cl_event event,cl_command_queue command_queue){
    cl_int error=0;
    cl_ulong eventStart,eventEnd;
    clFinish(command_queue);
    error = clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&eventStart,NULL);
    cl_errChk(error,"ERROR in Event Profiling.",true); 
    error = clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&eventEnd,NULL);
    cl_errChk(error,"ERROR in Event Profiling.",true);

    return (float)((eventEnd-eventStart)/1e9);
}*/


int main(int argc, char *argv[]) {
	
//********************************app-specific host variables***************************	
    printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);
	int matrix_dim = 32; /* default matrix_dim */
	int opt, option_index=0;
	func_ret_t ret;
	const char *input_file = NULL;
	float *m;
	float *recordDistances;
	
	while ((opt = getopt_long(argc, argv, "::vs:i:", 
                            long_options, &option_index)) != -1 ) {
		switch(opt){
			case 'i':
			input_file = optarg;
			break;
		//	case 'v':
		//	do_verify = 1;
			break;
        case 's':
			matrix_dim = atoi(optarg);
			printf("Generate input matrix internally, size =%d\n", matrix_dim);
			// fprintf(stderr, "Currently not supported, use -i instead\n");
			// fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
			// exit(EXIT_FAILURE);
			break;
        case '?':
			fprintf(stderr, "invalid option\n");
			break;
        case ':':
			fprintf(stderr, "missing argument\n");
			break;
        default:
			fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n",
                  argv[0]);
			exit(EXIT_FAILURE);
		}
	}
  
	if ( (optind < argc) || (optind == 1)) {
		fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
		exit(EXIT_FAILURE);
	}	

	if (input_file) {
		printf("Reading matrix from file %s\n", input_file);
		ret = create_matrix_from_file(&m, input_file, &matrix_dim);
		if (ret != RET_SUCCESS) {
			m = NULL;
			fprintf(stderr, "error create matrix from file %s\n", input_file);
			exit(EXIT_FAILURE);
		}
	} 
	
	else if (matrix_dim) {
	  printf("Creating matrix internally size=%d\n", matrix_dim);
	  ret = create_matrix(&m, matrix_dim);
	  if (ret != RET_SUCCESS) {
	    m = NULL;
	    fprintf(stderr, "error create matrix internally size=%d\n", matrix_dim);
	    exit(EXIT_FAILURE);
	  }
	}

	else {
	  printf("No input file specified!\n");
	  exit(EXIT_FAILURE);
	}

/*	if (do_verify){
		printf("Before LUD\n");
		print_matrix(m, matrix_dim);
		matrix_duplicate(m, &mm, matrix_dim);
	}*/
	
	int sourcesize = 1024*1024;
	char * source = (char *)calloc(sourcesize, sizeof(char)); 
	if(!source) { printf("ERROR: calloc(%d) failed\n", sourcesize); return -1; }
	
/*#ifdef BLOCK_SIZE
	sprintf(clOptions + strlen(clOptions), " -DBLOCK_SIZE=%d", BLOCK_SIZE);
#endif*/
//**************************END*******************************************************

    

  int quiet=0,timing=0,platform=-1,device=-1;  
 
//*************app-specific number of dataset*****************
//number of jobs need to be done, number of data need to be processed, can be translated to total number of threads (assuming each thread can just do one job).
  
 // int sourcesize = loadData(filename,records,locations);
  
//**************************END*******************************************************

  if (!quiet) {
    printf("Number of records: %d\n",sourcesize);
    //printf("Finding the %d closest neighbors.\n",resultsCount);
  }

  //if (resultsCount > sourcesize) resultsCount = sourcesize;


  context = cl_init_context(platform,device,quiet);
  
  recordDistances = OpenCl_LUD(context,sourcesize, matrix_dim, m, timing); 

  free(recordDistances);
  return 0;

}



float *OpenCl_LUD(cl_context context,int sourcesize,int matrix_dim,	float *m, int timing) {



//****************************App-spoecific naming convention, binary path*********************
    // 1. set up kernel
		//cl_kernel lud_diagonal,lud_perimeter, lud_internal;
        cl_kernel lud_diagonal;
        cl_int status;
        float writeTime=0, kernel_1=0, readTime=0;
        cl_program cl_NN_program;
        //cl_NN_program = cl_compileProgram((char *)"./binary/lud_kernel_base_cu4.aocx",NULL);
        cl_NN_program = cl_compileProgram((char *)"./binary/lud_kernel_base.aocx",NULL);
	    //cl_NN_program = cl_compileProgram((char *)"./binary/ludemu.aocx",NULL);   
        
        lud_diagonal = clCreateKernel(cl_NN_program, "lud_diagonal", &status);
        status = cl_errChk(status, (char *)"Error Creating lud_diagonal",true);
        if(status)exit(1);

//**********************************************END*************************************************88

    // 2. set up memory on device and send ipts data to device copy ipts to device
    // also need to alloate memory for the distancePoints
//**************************************App-specific device var*************************************    
    cl_mem d_m; //define var
    
    cl_int error=0;
	
	d_m = clCreateBuffer(context, CL_MEM_READ_WRITE, matrix_dim*matrix_dim * sizeof(float), NULL, &error); // allocate mem

    cl_command_queue command_queue =  cl_getCommandQueue(); //command queues for three parallel kernels (read, comp, write)
		   
	cl_event writeEvent,kernelEvent,readEvent;
	error = clEnqueueWriteBuffer(command_queue,d_m,1,0,matrix_dim*matrix_dim*sizeof(float), m,0,0,&writeEvent); //Copy the var from host to device
	
//********************************************END*************************************************************
   
//////////////////////////////////////////////////////////////////////////////////////////////////////////READ STARTS/////



//**************************************App-specific read kernel init and call ********************
    // 3. send arguments to device
    //printf("I am here\n");
    int i=0;
    //printf("I am here&&&&&&&&&&&&&&&&&&&&&&&&&\n");
    printf("Matrix_dim=%d\n",matrix_dim);
    printf("Block dim=%d\n",BLOCK_SIZE);
	//for (i=0; i < matrix_dim-BLOCK_SIZE; i += BLOCK_SIZE) {          
    cl_int argchk;
    argchk  = clSetKernelArg(lud_diagonal, 0, sizeof(void *), (void*) NULL);
    argchk |= clSetKernelArg(lud_diagonal, 1, sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, (void*)NULL );
	argchk |= clSetKernelArg(lud_diagonal, 2, sizeof(cl_int), (void*) &matrix_dim);
	argchk |= clSetKernelArg(lud_diagonal, 3, sizeof(cl_int), (void*) &i);
	
	 size_t global_work1[3]  = {BLOCK_SIZE, 1, 1};
	 size_t local_work1[3]  = {BLOCK_SIZE, 1, 1};
		     
  //  printf("Read kernel Starts\n");

    cl_errChk(argchk,"ERROR in Setting Nearest lud_diagonal args",true);
	error = clEnqueueNDRangeKernel(command_queue,lud_diagonal,2, NULL, global_work1, local_work1, 0, 0, &kernelEvent);
	//kernel_1+=eventTime(kernelEvent,command_queue);
	cl_errChk(error,"ERROR in Executing Kernel lud_diagonal",true);
	
//}
//	printf("Read kernel Executes\n");
//************************************** END ******************************************************
	

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////END OF OPERATION



//**************************************App-specific transfer from device to host **************
    status = clFinish(command_queue);
    error = clEnqueueReadBuffer( command_queue,d_m, 1, 0, matrix_dim*matrix_dim*sizeof(float), m, 0,0,&readEvent);
    //readTime+=eventTime(readEvent,command_queue);
	printf("Read final results from Device\n");
    cl_errChk(error,"ERROR with clEnqueueReadBuffer",true);
//************************************** END ******************************************************    
    
    if (1) {
        clFinish(command_queue);
        cl_ulong eventStart,eventEnd,totalTime=0,total_kernel_time=0;
        printf("# Records\tKernel_1(s)\n");
        printf("%d        \t",sourcesize);
        //printf("%d        \t",no_of_nodes);
        // Kernel_1
        //printf("%f\t",kernel_1);
        
        
	}
    
    
   //**************************************App-specific clear device mem **************
    clReleaseMemObject(d_m);
    
	free(m);
	
//	if(shutdown()) 
	return 0;	
	//************************************** END ****************************************************** 
}


