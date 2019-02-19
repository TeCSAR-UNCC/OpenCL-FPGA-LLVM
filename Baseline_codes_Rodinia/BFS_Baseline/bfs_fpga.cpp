#include "bfs_fpga.h"
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

#define MAX_THREADS_PER_BLOCK 256
#define WORK_DIM 2

double timestamp(){
	double ms=0.0;
	timeval time;
	gettimeofday(&time,NULL);
	ms=(time.tv_sec*1000.0)+(time.tv_usec/1000.0);
	return ms;
}


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

//----------------------------------------------------------
//--bfs on cpu
//--programmer:	jianbin
//--date:	26/01/2011
//--note: width is changed to the new_width
//----------------------------------------------------------
void run_bfs_cpu(int no_of_nodes, Node *h_graph_nodes, int edge_list_size, \
		int *h_graph_edges, char *h_graph_mask, char *h_updating_graph_mask, \
		char *h_graph_visited, int *h_cost_ref){
	char stop;
	int k = 0;
	do{
		//if no thread changes this value then the loop stops
		stop=false;
		for(int tid = 0; tid < no_of_nodes; tid++ )
		{
			if (h_graph_mask[tid] == true){ 
				h_graph_mask[tid]=false;
				for(int i=h_graph_nodes[tid].starting; i<(h_graph_nodes[tid].no_of_edges + h_graph_nodes[tid].starting); i++){
					int id = h_graph_edges[i];	//--cambine: node id is connected with node tid
					if(!h_graph_visited[id]){	//--cambine: if node id has not been visited, enter the body below
						h_cost_ref[id]=h_cost_ref[tid]+1;
						h_updating_graph_mask[id]=true;
					}
				}
			}		
		}

  		for(int tid=0; tid< no_of_nodes ; tid++ )
		{
			if (h_updating_graph_mask[tid] == true){
			h_graph_mask[tid]=true;
			h_graph_visited[tid]=true;
			stop=true;
			h_updating_graph_mask[tid]=false;
			}
		}
		k++;
	}
	while(stop);
}

int main(int argc, char * argv[])
{
	int no_of_nodes;
	int edge_list_size;
	int quiet=0,platform=-1,device=-1;
	FILE *fp;
	Node* h_graph_nodes;
	char *h_graph_mask, *h_updating_graph_mask, *h_graph_visited;
	cl_context context = cl_init_context(platform,device,quiet);
	try{
		char *input_f;
		if(argc!=2){
		  Usage(argc, argv);
		  exit(0);
		}
	
		input_f = argv[1];
		printf("Reading File\n");
		//Read in Graph from a file
		fp = fopen(input_f,"r");
		if(!fp){
		  printf("Error Reading graph file\n");
		  return 0;
		}

		int source = 0;

		fscanf(fp,"%d",&no_of_nodes);

		int num_of_blocks = 1;
		int num_of_threads_per_block = no_of_nodes;

		//Make execution Parameters according to the number of nodes
		//Distribute threads across multiple Blocks if necessary
		if(no_of_nodes>MAX_THREADS_PER_BLOCK){
			num_of_blocks = (int)ceil(no_of_nodes/(double)MAX_THREADS_PER_BLOCK); 
			num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
		}
		int work_group_size = num_of_threads_per_block;
		// allocate host memory
		h_graph_nodes = (Node*) malloc(sizeof(Node)*no_of_nodes);
		h_graph_mask = (char*) malloc(sizeof(char)*no_of_nodes);
		h_updating_graph_mask = (char*) malloc(sizeof(char)*no_of_nodes);
		h_graph_visited = (char*) malloc(sizeof(char)*no_of_nodes);
	
		int start, edgeno;   
		// initalize the memory
		for(int i = 0; i < no_of_nodes; i++){
			fscanf(fp,"%d %d",&start,&edgeno);
			h_graph_nodes[i].starting = start;
			h_graph_nodes[i].no_of_edges = edgeno;
			h_graph_mask[i]=false;
			h_updating_graph_mask[i]=false;
			h_graph_visited[i]=false;
		}
		//read the source node from the file
		fscanf(fp,"%d",&source);
		source=0;
		//set the source node as true in the mask
		h_graph_mask[source]=true;
		h_graph_visited[source]=true;
    	fscanf(fp,"%d",&edge_list_size);
   		int id,cost;
		int* h_graph_edges = (int*) malloc(sizeof(int)*edge_list_size);
		for(int i=0; i < edge_list_size ; i++){
			fscanf(fp,"%d",&id);
			fscanf(fp,"%d",&cost);
			h_graph_edges[i] = id;
		}

		if(fp)
			fclose(fp);    
		// allocate mem for the result on host side
		int	*h_cost = (int*) malloc(sizeof(int)*no_of_nodes);
		int *h_cost_ref = (int*)malloc(sizeof(int)*no_of_nodes);
		for(int i=0;i<no_of_nodes;i++){
			h_cost[i]=-1;
			h_cost_ref[i] = -1;
		}
		h_cost[source]=0;
		h_cost_ref[source]=0;
				
		//---------------------------------------------------------
		//--gpu entry
		run_bfs_gpu(context,no_of_nodes,h_graph_nodes,edge_list_size,h_graph_edges, h_graph_mask, h_updating_graph_mask, h_graph_visited, h_cost);	
		//---------------------------------------------------------
		//Cpu entry
		for(int i = 0; i < no_of_nodes; i++){
			h_graph_mask[i]=false;
			h_updating_graph_mask[i]=false;
			h_graph_visited[i]=false;
		}
		//set the source node as true in the mask
		source=0;
		h_graph_mask[source]=true;
		h_graph_visited[source]=true;
		run_bfs_cpu(no_of_nodes,h_graph_nodes,edge_list_size,h_graph_edges, h_graph_mask, h_updating_graph_mask, h_graph_visited, h_cost_ref);
		//--result varification
		compare_results<int>(h_cost_ref, h_cost, no_of_nodes);
		//release host memory		
		free(h_graph_nodes);
		free(h_graph_mask);
		free(h_updating_graph_mask);
		free(h_graph_visited);

	}
	catch(std::string msg){
		std::cout<<"--cambine: exception in main ->"<<msg<<std::endl;
		//release host memory
		free(h_graph_nodes);
		free(h_graph_mask);
		free(h_updating_graph_mask);
		free(h_graph_visited);		
	}
		
    return 0;
}

void run_bfs_gpu(cl_context context,int no_of_nodes, Node *h_graph_nodes, int edge_list_size,int *h_graph_edges, char *h_graph_mask, char *h_updating_graph_mask,char *h_graph_visited, int *h_cost) {

    // 1. set up kernel
		cl_kernel BFS_kernel_1,BFS_kernel_2;
        cl_int status;
        float writeTime=0, kernel_1=0, readTime=0, kernel_2=0;
        cl_program cl_BFS_program;
        cl_BFS_program = cl_compileProgram((char *)"./binary/BFS_1_default.aocx",NULL);
	       
        BFS_kernel_1 = clCreateKernel(cl_BFS_program, "BFS_1", &status);
        status = cl_errChk(status, (char *)"Error Creating BFS kernel_1",true);
        if(status)exit(1);

		BFS_kernel_2 = clCreateKernel(cl_BFS_program, "BFS_2", &status);
        status = cl_errChk(status, (char *)"Error Creating BFS kernel_2",true);
        if(status)exit(1);
        
                           


////////////////////////////////////////////////////////////////////////


	size_t globalWorkSize[1];
    globalWorkSize[0] = no_of_nodes;
 	if (no_of_nodes % 64) globalWorkSize[0] += 64 - (no_of_nodes % 64);
//	if (numRecords % 256) globalWorkSize[0] += 256 - (numRecords % 256);
    printf("Global Work Size: %zu\n",globalWorkSize[0]);   
//////////////////////////////////////////////////////////////////////// 

	char h_over;
    // 2. set up memory on device and send ipts data to device copy ipts to device
    cl_mem d_graph_nodes, d_graph_edges, d_graph_mask, d_updating_graph_mask,d_graph_visited, d_cost, d_over;

    cl_int error=0;

    d_graph_nodes = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,no_of_nodes*sizeof(Node), h_graph_nodes, &error);
    d_graph_edges = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, edge_list_size*sizeof(int), h_graph_edges, &error);
    d_graph_mask = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, no_of_nodes*sizeof(char), h_graph_mask, &error);
    d_updating_graph_mask = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, no_of_nodes*sizeof(char), h_updating_graph_mask, &error);
    d_graph_visited = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, no_of_nodes*sizeof(char), h_graph_visited, &error);
    d_cost = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, no_of_nodes*sizeof(int), h_cost, &error);
    d_over = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(char), &h_over, &error);
 
	cl_command_queue command_queue = cl_getCommandQueue();
	//cl_event writeEvent,kernelEvent,kernelEvent1,kernelEvent2,kernelEvent3,readEvent;
	cl_event writeEvent,kernelEvent,kernelEvent1,readEvent;

	double first_stamp=timestamp();

    error = clEnqueueWriteBuffer(command_queue, d_graph_nodes, CL_TRUE, 0, no_of_nodes*sizeof(Node), h_graph_nodes, 0, NULL, &writeEvent);
    writeTime+=eventTime(writeEvent,command_queue);
    clReleaseEvent(writeEvent);
    error = clEnqueueWriteBuffer(command_queue, d_graph_edges, CL_TRUE, 0, edge_list_size*sizeof(int), h_graph_edges, 0, NULL, &writeEvent);
    writeTime+=eventTime(writeEvent,command_queue);
    clReleaseEvent(writeEvent);
    error = clEnqueueWriteBuffer(command_queue, d_graph_mask, CL_TRUE, 0, no_of_nodes*sizeof(char), h_graph_mask, 0, NULL, &writeEvent);
    writeTime+=eventTime(writeEvent,command_queue);
    clReleaseEvent(writeEvent);
    error = clEnqueueWriteBuffer(command_queue, d_updating_graph_mask, CL_TRUE, 0, no_of_nodes*sizeof(char), h_updating_graph_mask, 0, NULL, &writeEvent);
    writeTime+=eventTime(writeEvent,command_queue);
    clReleaseEvent(writeEvent);
    error = clEnqueueWriteBuffer(command_queue, d_graph_visited, CL_TRUE, 0, no_of_nodes*sizeof(char), h_graph_visited, 0, NULL, &writeEvent);
    writeTime+=eventTime(writeEvent,command_queue);
    clReleaseEvent(writeEvent);
    error = clEnqueueWriteBuffer(command_queue, d_cost, CL_TRUE, 0, no_of_nodes*sizeof(int), h_cost, 0, NULL, &writeEvent);
    writeTime+=eventTime(writeEvent,command_queue);
    clReleaseEvent(writeEvent);

   
//////////////////////////////////////////////////////////////////////////////////////////////////////////READ STARTS/////
    // 3. send arguments to device
    cl_int argchk;
    do{
		h_over = false;
		error = clEnqueueWriteBuffer(command_queue, d_over, CL_TRUE, 0, sizeof(char), &h_over, 0, NULL, &writeEvent);
		writeTime+=eventTime(writeEvent,command_queue);
		clReleaseEvent(writeEvent);
		
		argchk  = clSetKernelArg(BFS_kernel_1, 0, sizeof(cl_mem), (void *)&d_graph_nodes);
		argchk  = clSetKernelArg(BFS_kernel_1, 1, sizeof(cl_mem), (void *)&d_graph_edges);
		argchk  = clSetKernelArg(BFS_kernel_1, 2, sizeof(cl_mem), (void *)&d_graph_mask);
		argchk  = clSetKernelArg(BFS_kernel_1, 3, sizeof(cl_mem), (void *)&d_updating_graph_mask);
		argchk  = clSetKernelArg(BFS_kernel_1, 4, sizeof(cl_mem), (void *)&d_graph_visited);
		argchk  = clSetKernelArg(BFS_kernel_1, 5, sizeof(cl_mem), (void *)&d_cost);
		argchk  = clSetKernelArg(BFS_kernel_1, 6, sizeof(int), (void *)&no_of_nodes);
		//printf("kernel_1 Starts\n");
		cl_errChk(argchk,"ERROR in Setting BFS_kernel_1 args",true);
		// 4. enqueue kernel
		

		// 3. send arguments to device
		argchk = clSetKernelArg(BFS_kernel_2, 0, sizeof(cl_mem), (void *)&d_graph_mask);
		argchk = clSetKernelArg(BFS_kernel_2, 1, sizeof(cl_mem), (void *)&d_updating_graph_mask);
		argchk = clSetKernelArg(BFS_kernel_2, 2, sizeof(cl_mem), (void *)&d_graph_visited);
		argchk = clSetKernelArg(BFS_kernel_2, 3, sizeof(cl_mem), (void *)&d_over);
		argchk = clSetKernelArg(BFS_kernel_2, 4, sizeof(int), (void *)&no_of_nodes);
		//printf("kernel 2 Starts\n");
		cl_errChk(argchk,"ERROR in Setting BFS_2 kernel args",true);

		// 4. enqueue kernel
		error = clEnqueueNDRangeKernel(command_queue,BFS_kernel_1,1,0,globalWorkSize,NULL,0, NULL, &kernelEvent);
		//kernel_1+=eventTime(kernelEvent,command_queue);
		cl_errChk(error,"ERROR in Executing Kernel BFS_1",true);
		//printf("kernel_1 Executes\n");

		error = clEnqueueNDRangeKernel(command_queue,BFS_kernel_2, 1, 0,globalWorkSize,NULL,0, NULL, &kernelEvent1);
		status = clFinish(command_queue);
		//kernel_2+=eventTime(kernelEvent1,command_queue);
		clReleaseEvent(kernelEvent1);
		//printf("kernel_2 Executes\n");
		cl_errChk(error,"ERROR in Executing Kernel BFS_2",true);
		
		error = clEnqueueReadBuffer(command_queue, d_over, 1, 0, sizeof(char), &h_over, 0,NULL,&readEvent);
		readTime+=eventTime(readEvent,command_queue);
		clReleaseEvent(readEvent);
    	}while(h_over);
    	status = clFinish(command_queue);
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////END OF OPERATION

    // 5. transfer data off of device

    error = clEnqueueReadBuffer(command_queue, d_cost, 1, 0, no_of_nodes*sizeof(int), h_cost, 0,NULL,&readEvent);
	
	double second_stamp=timestamp();
    double time_elapsed=second_stamp - first_stamp;
    printf("\n Total Kernel execution time=%0.3f ms\n", time_elapsed);
	
	readTime+=eventTime(readEvent,command_queue);
	clReleaseEvent(readEvent);
	printf("Read final results from Device\n");
	
    cl_errChk(error,"ERROR with clEnqueueReadBuffer",true);
    /*if (1) {
        clFinish(command_queue);
        cl_ulong eventStart,eventEnd,totalTime=0,total_kernel_time=0;
        printf("# Records\tKernel_1(s)\tKernel_2(s)\n");
        printf("%d        \t",no_of_nodes);
        // Kernel_1
        printf("%f\t",kernel_1);
        // Read Buffer
        printf("%f \t\n",kernel_2);
        
    }*/
    // 6. return finalized data and release buffers
    clReleaseMemObject(d_graph_nodes);
    clReleaseMemObject(d_graph_edges);
    clReleaseMemObject(d_graph_mask);
    clReleaseMemObject(d_updating_graph_mask);
    clReleaseMemObject(d_graph_visited);
    clReleaseMemObject(d_cost);
    clReleaseMemObject(d_over);
	
}
