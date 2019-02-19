#include <iostream>
#include <vector>
#include <float.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// All OpenCL headers
#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif 

#include "clutils.h"
//#include "utils.h"

#include <algorithm>

struct Node
{
	int starting;
	int no_of_edges;
};
	
#define REC_LENGTH 49 // size of a record in db
void run_bfs_gpu(cl_context context,int no_of_nodes, Node *h_graph_nodes, int edge_list_size,int *h_graph_edges, char *h_graph_mask, char *h_updating_graph_mask,char *h_graph_visited, int *h_cost);
