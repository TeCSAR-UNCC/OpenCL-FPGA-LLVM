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
        #define BLOCK_SIZE 16
	
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
		//	break;
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
 
/*  // parse command line
  if (parseCommandline(argc, argv, filename, &resultsCount, &lat, &lng,
                     &quiet, &timing, &platform, &device)) {
    printUsage();
    return 0;
  }

*/

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

  

    
  // print out results
 /* if (!quiet)
    for(i=0;i<resultsCount;i++) { 
      printf("%s --> Distance=%f\n",records[i].recString,records[i].distance);
    }*/
  free(recordDistances);
  return 0;

}



float *OpenCl_LUD(cl_context context,int sourcesize,int matrix_dim,	float *m, int timing) {



//****************************App-spoecific naming convention, binary path*********************
    // 1. set up kernel
		cl_kernel lud_diagonal_read,lud_diagonal_compute, lud_diagonal_wb;
        cl_int status;
        float writeTime=0, kernel_1=0, readTime=0, kernel_2=0;
        cl_program cl_NN_program;
        //cl_NN_program = cl_compileProgram((char *)"./binary/ludemu.aocx",NULL);
        //cl_NN_program = cl_compileProgram((char *)"./binary/ludfpga.aocx",NULL);
        //cl_NN_program = cl_compileProgram((char *)"./binary/lud_kernel_4.aocx",NULL);
        //cl_NN_program = cl_compileProgram((char *)"./binary/lud_kernel_wob4.aocx",NULL);
	    //cl_NN_program = cl_compileProgram((char *)"./binary/lud_kernel_8.aocx",NULL);
	    //cl_NN_program = cl_compileProgram((char *)"./binary/lud_kernel_wob8.aocx",NULL);
	    cl_NN_program = cl_compileProgram((char *)"./binary/lud_kernel_16.aocx",NULL);
	    //cl_NN_program = cl_compileProgram((char *)"./binary/lud_kernel_wob16.aocx",NULL);
	    //cl_NN_program = cl_compileProgram((char *)"./binary/lud_kernel_32.aocx",NULL);
	    //cl_NN_program = cl_compileProgram((char *)"./binary/lud_kernel_wob32.aocx",NULL);
	    //cl_NN_program = cl_compileProgram((char *)"./binary/lud_kernel_64.aocx",NULL);
	    //cl_NN_program = cl_compileProgram((char *)"./binary/lud_kernel_wob64.aocx",NULL);
	    //cl_NN_program = cl_compileProgram((char *)"./binary/lud_kernel_128.aocx",NULL);
	    //cl_NN_program = cl_compileProgram((char *)"./binary/lud_kernel_wob128.aocx",NULL);
	    //cl_NN_program = cl_compileProgram((char *)"./binary/lud_kernel_256.aocx",NULL);
	    //cl_NN_program = cl_compileProgram((char *)"./binary/lud_kernel_wob256.aocx",NULL);	    
	       
        lud_diagonal_read = clCreateKernel(cl_NN_program, "lud_diagonal_read", &status);
        status = cl_errChk(status, (char *)"Error Creating lud kernel_read",true);
        if(status)exit(1);

		lud_diagonal_compute = clCreateKernel(cl_NN_program, "lud_diagonal_compute", &status);
        status = cl_errChk(status, (char *)"Error Creating lud kernel_compute",true);
        if(status)exit(1);    
                     
		lud_diagonal_wb = clCreateKernel(cl_NN_program, "lud_diagonal_wb", &status);
        status = cl_errChk(status, (char *)"Error Creating lud kernel_wb",true);
        if(status)exit(1);
//**********************************************END*************************************************88

	//size_t size;
	//size_t globalWorkSize[1];
	//globalWorkSize[0] = sourcesize;
	//if (sourcesize % 64) globalWorkSize[0] += 64 - (sourcesize % 64);
	//if (sourcesize % 256) globalWorkSize[0] += 256 - (sourcesize % 256);
	//printf("Global Work Size: %zu\n",globalWorkSize[0]);   



    // 2. set up memory on device and send ipts data to device copy ipts to device
    // also need to alloate memory for the distancePoints
//**************************************App-specific device var*************************************    
    cl_mem d_m; //define var
    
    cl_int error=0;
	
	d_m = clCreateBuffer(context, CL_MEM_READ_WRITE, matrix_dim*matrix_dim * sizeof(float), NULL, &error); // allocate mem

    cl_command_queue command_queue =  cl_getCommandQueue(); //command queues for three parallel kernels (read, comp, write)
	cl_command_queue command_queue1 = cl_getCommandQueue1(); 
	cl_command_queue command_queue2 = cl_getCommandQueue2();
		   
	cl_event writeEvent,kernelEvent,kernelEvent1,kernelEvent2,readEvent;
	error = clEnqueueWriteBuffer(command_queue,d_m,1,0,matrix_dim*matrix_dim*sizeof(float), m,0,0,&writeEvent); //Copy the var from host to device
	
//********************************************END*************************************************************
   
//////////////////////////////////////////////////////////////////////////////////////////////////////////READ STARTS/////



//**************************************App-specific read kernel init and call ********************
    // 3. send arguments to device
    //printf("I am here\n");
    int i=0;
    printf("Matrix_dim=%d\n",matrix_dim);
    printf("Block dim=%d\n",BLOCK_SIZE);
	//for (i=0; i < matrix_dim-BLOCK_SIZE; i += BLOCK_SIZE) {          
    cl_int argchk;
    argchk  = clSetKernelArg(lud_diagonal_read, 0, sizeof(void *), (void*) &d_m);
    //argchk |= clSetKernelArg(lud_diagonal_read, 1, sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, (void*)NULL );
	argchk |= clSetKernelArg(lud_diagonal_read, 1, sizeof(cl_int), (void*) &matrix_dim);
	argchk |= clSetKernelArg(lud_diagonal_read, 2, sizeof(cl_int), (void*) &i);
	
	 size_t global_work1[3]  = {BLOCK_SIZE, 1, 1};
	 size_t local_work1[3]  = {BLOCK_SIZE, 1, 1};
		     
  //  printf("Read kernel Starts\n");

    cl_errChk(argchk,"ERROR in Setting lud_read kernel args",true);
	error = clEnqueueNDRangeKernel(command_queue,lud_diagonal_read,2, NULL, global_work1, local_work1, 0, 0, &kernelEvent);
	//kernel_1+=eventTime(kernelEvent,command_queue);
	cl_errChk(error,"ERROR in Executing Kernel lud_read kernel",true);
//	printf("Read kernel Executes\n");
//************************************** END ******************************************************
	


//**************************************App-specific comp kernel init and call **************
	  argchk = clSetKernelArg(lud_diagonal_compute, 0, sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, (void*)NULL);
	  argchk |=clSetKernelArg(lud_diagonal_compute, 1, sizeof(cl_int), (void*) &matrix_dim );
	  argchk |=clSetKernelArg(lud_diagonal_compute, 2, sizeof(cl_int), (void*) &i );
	  //argchk |=clSetKernelArg(perimeter, 3, sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, (void*)NULL );
	 // argchk |=clSetKernelArg(perimeter, 4, sizeof(cl_int), (void*) &matrix_dim);
	 // argchk |=clSetKernelArg(perimeter, 5, sizeof(cl_int), (void*) &i);
	  
	 
//	  printf("Compute kernel Starts\n");
	
	cl_errChk(argchk,"ERROR in Setting lud_read kernel args",true);  	 	 
    error = clEnqueueNDRangeKernel(command_queue1,lud_diagonal_compute, 2, NULL, global_work1, local_work1, 0, 0, &kernelEvent1);
	cl_errChk(error,"ERROR in Executing Kernel lud_compute kernel",true);
//	printf("Compute kernel Executes\n");
	
//************************************** END ******************************************************

	  

//**************************************App-specific write kernel init and call **************
	
	argchk = clSetKernelArg(lud_diagonal_wb, 0, sizeof(void *), (void*) &d_m);
	//argchk |= clSetKernelArg(internal, 1, sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, (void*)NULL );
	//argchk |= clSetKernelArg(internal, 2, sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, (void*)NULL );
	argchk |= clSetKernelArg(lud_diagonal_wb, 1, sizeof(cl_int), (void*) &matrix_dim);
	argchk |= clSetKernelArg(lud_diagonal_wb, 2, sizeof(cl_int), (void*) &i);
	
	
      
	//printf("Writeback kernel Starts\n");
	
	cl_errChk(argchk,"ERROR in Setting lud_WB kernel args",true);
    error = clEnqueueNDRangeKernel(command_queue2,lud_diagonal_wb, 2, NULL, global_work1, local_work1, 0, 0, &kernelEvent2);
    cl_errChk(error,"ERROR in Executing Kernel lud_wb kernel",true);
    
 // }
  //  printf("Writeback kernel Executes\n");
    
    //*************************************END **************
    
    
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////END OF OPERATION




//**************************************App-specific transfer from device to host **************
    status = clFinish(command_queue2);
    error = clEnqueueReadBuffer( command_queue2,d_m, 1, 0, matrix_dim*matrix_dim*sizeof(float), m, 0,0,&readEvent);
   // readTime+=eventTime(readEvent,command_queue2);
        //status = clFinish(command_queue2);
	printf("Read final results from Device\n");
    cl_errChk(error,"ERROR with clEnqueueReadBuffer",true);
//************************************** END ******************************************************    
    
  /*  if (1) {
        clFinish(command_queue);
        clFinish(command_queue1);
        clFinish(command_queue2);
        cl_ulong eventStart,eventEnd,totalTime=0,total_kernel_time=0;
        printf("# Records\tKernel(s)\n" );
        printf("%d        \t",sourcesize);
        // Kernel_1
       // printf("%f\t",kernel_1);
        // Read Buffer
        
	}*/
    
    
    
     if (1) {
        clFinish( command_queue2);
        cl_ulong eventStart,eventEnd,totalTime=0;
        printf("# Records\tWrite(s) [size]\t\tKernel_Read(s)\tKernel_Compute(s)\tKernel_Wb(s)\tRead(s)  [size]\t\tTotal(s)\n");
        printf("%d        \t",sourcesize);
        // Write Buffer
        error = clGetEventProfilingInfo(writeEvent,CL_PROFILING_COMMAND_START,
                                        sizeof(cl_ulong),&eventStart,NULL);
        cl_errChk(error,"ERROR in Event Profiling (Write Start)",true); 
        error = clGetEventProfilingInfo(writeEvent,CL_PROFILING_COMMAND_END,
                                        sizeof(cl_ulong),&eventEnd,NULL);
        cl_errChk(error,"ERROR in Event Profiling (Write End)",true);

        //printf("%f [%.2fMB]\t",(float)((eventEnd-eventStart)/1e9),(float)((sizeof(LatLong) * numRecords)/1e6));
        totalTime += eventEnd-eventStart;
        // Kernel_read
        error = clGetEventProfilingInfo(kernelEvent,CL_PROFILING_COMMAND_START,
                                        sizeof(cl_ulong),&eventStart,NULL);
        cl_errChk(error,"ERROR in Event Profiling (Kernel Start)",true); 
        error = clGetEventProfilingInfo(kernelEvent,CL_PROFILING_COMMAND_END,
                                        sizeof(cl_ulong),&eventEnd,NULL);
        cl_errChk(error,"ERROR in Event Profiling (Kernel End)",true);

        printf("%f\t",(float)((eventEnd-eventStart)/1e9));
        totalTime += eventEnd-eventStart;
        // Kernel_Compute
        error = clGetEventProfilingInfo(kernelEvent1,CL_PROFILING_COMMAND_START,
                                        sizeof(cl_ulong),&eventStart,NULL);
        cl_errChk(error,"ERROR in Event Profiling (Kernel Start)",true); 
        error = clGetEventProfilingInfo(kernelEvent1,CL_PROFILING_COMMAND_END,
                                        sizeof(cl_ulong),&eventEnd,NULL);
        cl_errChk(error,"ERROR in Event Profiling (Kernel End)",true);

        printf("%f\t",(float)((eventEnd-eventStart)/1e9));
        totalTime += eventEnd-eventStart;
        // Kernel_wb
        error = clGetEventProfilingInfo(kernelEvent2,CL_PROFILING_COMMAND_START,
                                        sizeof(cl_ulong),&eventStart,NULL);
        cl_errChk(error,"ERROR in Event Profiling (Kernel Start)",true); 
        error = clGetEventProfilingInfo(kernelEvent2,CL_PROFILING_COMMAND_END,
                                        sizeof(cl_ulong),&eventEnd,NULL);
        cl_errChk(error,"ERROR in Event Profiling (Kernel End)",true);

        printf("%f\t",(float)((eventEnd-eventStart)/1e9));
        totalTime += eventEnd-eventStart;
        // Read Buffer
        error = clGetEventProfilingInfo(readEvent,CL_PROFILING_COMMAND_START,
                                        sizeof(cl_ulong),&eventStart,NULL);
        cl_errChk(error,"ERROR in Event Profiling (Read Start)",true); 
        error = clGetEventProfilingInfo(readEvent,CL_PROFILING_COMMAND_END,
                                        sizeof(cl_ulong),&eventEnd,NULL);
        cl_errChk(error,"ERROR in Event Profiling (Read End)",true);

        //printf("%f [%.2fMB]\t",(float)((eventEnd-eventStart)/1e9),(float)((sizeof(float) * numRecords)/1e6));
        totalTime += eventEnd-eventStart;
        
        printf("%f\n\n",(float)(totalTime/1e9));
        
    }
    
    
//**************************************App-specific clear device mem **************
    clReleaseMemObject(d_m);
    
   	free(m);
	
	
//	if(shutdown()) 
	return 0;	
	//************************************** END ****************************************************** 
}


