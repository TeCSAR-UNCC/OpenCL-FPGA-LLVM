#ifndef __NEAREST_NEIGHBOR__
#define __NEAREST_NEIGHBOR__
#include "nearestNeighbor.h"

cl_context context=NULL;
//unsigned num_devices = 0;//////////////////////////////////////////////********************************************

int main(int argc, char *argv[]) {
	
//********************************app-specific host variables***************************	
  std::vector<Record> records;  
  float *recordDistances;     
  //LatLong locations[REC_WINDOW];
  std::vector<LatLong> locations;
  int i;
  // args
  char filename[100];
  float lat=0.0,lng=0.0;
//**************************END*******************************************************

    
  
  int resultsCount=10,quiet=0,timing=0,platform=-1,device=-1;  
 
  // parse command line
  if (parseCommandline(argc, argv, filename, &resultsCount, &lat, &lng,
                     &quiet, &timing, &platform, &device)) {
    printUsage();
    return 0;
  }



//*************app-specific number of dataset*****************
//number of jobs need to be done, number of data need to be processed, can be translated to total number of threads (assuming each thread can just do one job).
  
  int numRecords = loadData(filename,records,locations);
  
//**************************END*******************************************************

  if (!quiet) {
    printf("Number of records: %d\n",numRecords);
    printf("Finding the %d closest neighbors.\n",resultsCount);
  }

  if (resultsCount > numRecords) resultsCount = numRecords;


  context = cl_init_context(platform,device,quiet);
  
  recordDistances = OpenClFindNearestNeighbors(context,numRecords,locations,lat,lng,timing);
  //recordDistances = OpenClFindNearestNeighbors(context,40000,locations,lat,lng,timing);//////////////global records changed

  // find the resultsCount least distances
  findLowest(records,recordDistances,numRecords,resultsCount);

  // print out results
  if (!quiet)
    for(i=0;i<resultsCount;i++) { 
      printf("%s --> Distance=%f\n",records[i].recString,records[i].distance);
    }
  free(recordDistances);
  return 0;
}

float *OpenClFindNearestNeighbors(
	cl_context context,
	int numRecords,
	std::vector<LatLong> &locations,float lat,float lng,
	int timing) {



//****************************App-spoecific naming convention, binary path*********************
    // 1. set up kernel
		cl_kernel NN_kernel_read,NN_kernel_compute, NN_kernel_wb;
        cl_int status;
        cl_program cl_NN_program;
        cl_NN_program = cl_compileProgram((char *)"./binary/nn_channel.aocx",NULL);
        	       
        NN_kernel_read = clCreateKernel(cl_NN_program, "NearestNeighbor_read", &status);
        status = cl_errChk(status, (char *)"Error Creating Nearest Neighbor kernel_read",true);
        if(status)exit(1);

		    NN_kernel_compute = clCreateKernel(cl_NN_program, "NearestNeighbor_compute", &status);
        status = cl_errChk(status, (char *)"Error Creating Nearest Neighbor kernel_compute",true);
        if(status)exit(1);
        
                           
		    NN_kernel_wb = clCreateKernel(cl_NN_program, "NearestNeighbor_wb", &status);
        status = cl_errChk(status, (char *)"Error Creating Nearest Neighbor kernel_wb",true);
        if(status)exit(1);
//**********************************************END*************************************************88


	size_t globalWorkSize[1];
    globalWorkSize[0] = numRecords;
	if (numRecords % 64) globalWorkSize[0] += 64 - (numRecords % 64);
//	if (numRecords % 256) globalWorkSize[0] += 256 - (numRecords % 256);
    printf("Global Work Size: %zu\n",globalWorkSize[0]);   



    // 2. set up memory on device and send ipts data to device copy ipts to device
    // also need to alloate memory for the distancePoints
//**************************************App-specific device var*************************************    
    cl_mem d_locations; //define var
    cl_mem d_distances;

    cl_int error=0;

    d_locations = clCreateBuffer(context, CL_MEM_READ_ONLY,sizeof(LatLong) * numRecords, NULL, &error); // allocate mem

    d_distances = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float) * numRecords, NULL, &error); //allocate mem

	cl_command_queue command_queue =  cl_getCommandQueue(); //command queues for three parallel kernels (read, comp, write)
	cl_command_queue command_queue1 = cl_getCommandQueue1(); 
	cl_command_queue command_queue2 = cl_getCommandQueue2();
	   
	cl_event writeEvent,kernelEvent,kernelEvent1,kernelEvent2,readEvent;
	
    error = clEnqueueWriteBuffer(command_queue,d_locations,1,0,sizeof(LatLong) * numRecords,&locations[0],0,NULL,&writeEvent); //Copy the var from hosto to device
//********************************************END*************************************************************
   
//////////////////////////////////////////////////////////////////////////////////////////////////////////READ STARTS/////



//**************************************App-specific read kernel init and call ********************
    // 3. send arguments to device
    cl_int argchk;
    argchk  = clSetKernelArg(NN_kernel_read, 0, sizeof(cl_mem), (void *)&d_locations);
   // printf("Read kernel Starts\n");

    cl_errChk(argchk,"ERROR in Setting Nearest Neighbor_read kernel args",true);

    error = clEnqueueNDRangeKernel(command_queue,NN_kernel_read,1,0,globalWorkSize,NULL,0, NULL, &kernelEvent);
	cl_errChk(error,"ERROR in Executing Kernel NearestNeighbor_read",true);
	//printf("Read kernel Executes\n");
//************************************** END ******************************************************
	


//**************************************App-specific comp kernel init and call **************
    argchk = clSetKernelArg(NN_kernel_compute, 0, sizeof(float), (void *)&lat);
    argchk |= clSetKernelArg(NN_kernel_compute, 1, sizeof(float), (void *)&lng);
	//printf("Compute kernel Starts\n");
    cl_errChk(argchk,"ERROR in Setting Nearest Neighbor_compute kernel args",true);

    error = clEnqueueNDRangeKernel(command_queue1,NN_kernel_compute, 1, 0,globalWorkSize,NULL,0, NULL, &kernelEvent1);
	
	//printf("Compute kernel Executes\n");
    cl_errChk(error,"ERROR in Executing Kernel NearestNeighbor_compute",true);
//************************************** END ******************************************************



//**************************************App-specific write kernel init and call **************
	  argchk = clSetKernelArg(NN_kernel_wb, 0, sizeof(cl_mem), (void *)&d_distances);

   // printf("Writeback kernel Starts\n");
    cl_errChk(argchk,"ERROR in Setting Nearest Neighbor_WB kernel args",true);
    error = clEnqueueNDRangeKernel(command_queue2,NN_kernel_wb, 1, 0,globalWorkSize,NULL,0, NULL, &kernelEvent2);

    //printf("Writeback kernel Executes\n");
    cl_errChk(error,"ERROR in Executing Kernel NearestNeighbor_wb",true);
    //*************************************END **************
    
    
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////END OF OPERATION




//**************************************App-specific transfer from device to host **************
    float *distances = (float *)malloc(sizeof(float) * numRecords);
    status = clFinish(command_queue2);

    error = clEnqueueReadBuffer( command_queue2,d_distances,1,0,sizeof(float) * numRecords,distances,0,NULL,&readEvent);
        //status = clFinish(command_queue2);
	//printf("Read final results from Device\n");
    cl_errChk(error,"ERROR with clEnqueueReadBuffer",true);
//************************************** END ******************************************************    
    
    
    
    
    
    
    if (timing) {
        clFinish( command_queue2);
        cl_ulong eventStart,eventEnd,totalTime=0;
        printf("# Records\tWrite(s) [size]\t\tKernel_Read(s)\tKernel_Compute(s)\tKernel_Wb(s)\tRead(s)  [size]\t\tTotal(s)\n");
        printf("%d        \t",numRecords);
        // Write Buffer
        error = clGetEventProfilingInfo(writeEvent,CL_PROFILING_COMMAND_START,
                                        sizeof(cl_ulong),&eventStart,NULL);
        cl_errChk(error,"ERROR in Event Profiling (Write Start)",true); 
        error = clGetEventProfilingInfo(writeEvent,CL_PROFILING_COMMAND_END,
                                        sizeof(cl_ulong),&eventEnd,NULL);
        cl_errChk(error,"ERROR in Event Profiling (Write End)",true);

        printf("%f [%.2fMB]\t",(float)((eventEnd-eventStart)/1e9),(float)((sizeof(LatLong) * numRecords)/1e6));
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

        printf("%f [%.2fMB]\t",(float)((eventEnd-eventStart)/1e9),(float)((sizeof(float) * numRecords)/1e6));
        totalTime += eventEnd-eventStart;
        
        printf("%f\n\n",(float)(totalTime/1e9));
        
    }



//**************************************App-specific clear device mem **************
    clReleaseMemObject(d_locations);
    clReleaseMemObject(d_distances);
	return distances;	
	//************************************** END ****************************************************** 
}






	int loadData(char *filename,std::vector<Record> &records,std::vector<LatLong> &locations){
    FILE   *flist,*fp;
	int    i=0;
	char dbname[64];
	int recNum=0;
	
    /**Main processing **/
    
    flist = fopen(filename, "r");
	while(!feof(flist)) {
		/**
		* Read in REC_WINDOW records of length REC_LENGTH
		* If this is the last file in the filelist, then done
		* else open next file to be read next iteration
		*/
		if(fscanf(flist, "%s\n", dbname) != 1) {
            fprintf(stderr, "error reading filelist\n");
            exit(0);
        }
        fp = fopen(dbname, "r");
        if(!fp) {
            printf("error opening a db\n");
            exit(1);
        }
        // read each record
        while(!feof(fp)){
            Record record;
            LatLong latLong;
            fgets(record.recString,49,fp);
            fgetc(fp); // newline
            if (feof(fp)) break;
            
            // parse for lat and long
            char substr[6];
            
            for(i=0;i<5;i++) substr[i] = *(record.recString+i+28);
            substr[5] = '\0';
            latLong.lat = atof(substr);
            
            for(i=0;i<5;i++) substr[i] = *(record.recString+i+33);
            substr[5] = '\0';
            latLong.lng = atof(substr);
            
            locations.push_back(latLong);
            records.push_back(record);
            recNum++;
        }
        fclose(fp);
    }
    fclose(flist);
    return recNum;
}

void findLowest(std::vector<Record> &records,float *distances,int numRecords,int topN){
  int i,j;
  float val;
  int minLoc;
  Record *tempRec;
  float tempDist;
  
  for(i=0;i<topN;i++) {
    minLoc = i;
    for(j=i;j<numRecords;j++) {
      val = distances[j];
      if (val < distances[minLoc]) minLoc = j;
    }
    // swap locations and distances
    tempRec = &records[i];
    records[i] = records[minLoc];
    records[minLoc] = *tempRec;
    
    tempDist = distances[i];
    distances[i] = distances[minLoc];
    distances[minLoc] = tempDist;
    
    // add distance to the min we just found
    records[i].distance = distances[i];
  }
}

int parseCommandline(int argc, char *argv[], char* filename,int *r,float *lat,float *lng,
                     int *q, int *t, int *p, int *d){
    int i;
    if (argc < 2) return 1; // error
    strncpy(filename,argv[1],100);
    char flag;
    
    for(i=1;i<argc;i++) {
      if (argv[i][0]=='-') {// flag
        flag = argv[i][1];
          switch (flag) {
            case 'r': // number of results
              i++;
              *r = atoi(argv[i]);
              break;
            case 'l': // lat or lng
              if (argv[i][2]=='a') {//lat
                *lat = atof(argv[i+1]);
              }
              else {//lng
                *lng = atof(argv[i+1]);
              }
              i++;
              break;
            case 'h': // help
              return 1;
              break;
            case 'q': // quiet
              *q = 1;
              break;
            case 't': // timing
              *t = 1;
              break;
            case 'p': // platform
              i++;
              *p = atoi(argv[i]);
              break;
            case 'd': // device
              i++;
              *d = atoi(argv[i]);
              break;
        }
      }
    }
    if ((*d >= 0 && *p<0) || (*p>=0 && *d<0)) // both p and d must be specified if either are specified
      return 1;
    return 0;
}

void printUsage(){
  printf("Nearest Neighbor Usage\n");
  printf("\n");
  printf("nearestNeighbor [filename] -r [int] -lat [float] -lng [float] [-hqt] [-p [int] -d [int]]\n");
  printf("\n");
  printf("example:\n");
  printf("$ ./nearestNeighbor filelist.txt -r 5 -lat 30 -lng 90\n");
  printf("\n");
  printf("filename     the filename that lists the data input files\n");
  printf("-r [int]     the number of records to return (default: 10)\n");
  printf("-lat [float] the latitude for nearest neighbors (default: 0)\n");
  printf("-lng [float] the longitude for nearest neighbors (default: 0)\n");
  printf("\n");
  printf("-h, --help   Display the help file\n");
  printf("-q           Quiet mode. Suppress all text output.\n");
  printf("-t           Print timing information.\n");
  printf("\n");
  printf("-p [int]     Choose the platform (must choose both platform and device)\n");
  printf("-d [int]     Choose the device (must choose both platform and device)\n");
  printf("\n");
  printf("\n");
  printf("Notes: 1. The filename is required as the first parameter.\n");
  printf("       2. If you declare either the device or the platform,\n");
  printf("          you must declare both.\n\n");
}

#endif
