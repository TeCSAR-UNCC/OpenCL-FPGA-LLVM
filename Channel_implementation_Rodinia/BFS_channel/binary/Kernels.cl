/* 
============================================================
//--cambine: kernel funtion of Breadth-First-Search
//--author:	created by Jianbin Fang
//--date:	06/12/2010
============================================================ */
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store: enable
# pragma OPENCL EXTENSION cl_altera_channels : enable

//Structure to hold a node information
typedef struct type_{
	int starting;
	int no_of_edges;
} Node;
//--7 parameters

/*
//__attribute__((num_simd_work_items(4)))
//__attribute__((reqd_work_group_size(64,1,1)))
__kernel void BFS_1( const __global Node* g_graph_nodes,
					const __global int* g_graph_edges, 
					__global char* g_graph_mask, 
					__global char* g_updating_graph_mask, 
					__global char* g_graph_visited, 
					__global int* g_cost, 
					const  int no_of_nodes){
	int tid = get_global_id(0);
	if( tid<no_of_nodes && g_graph_mask[tid]){
		g_graph_mask[tid]=false;
		for(int i=g_graph_nodes[tid].starting; i<(g_graph_nodes[tid].no_of_edges + g_graph_nodes[tid].starting); i++){
			int id = g_graph_edges[i];
			if(!g_graph_visited[id]){
				g_cost[id]=g_cost[tid]+1;
				g_updating_graph_mask[id]=true;
				}
			}
	}	
}
*/


channel Node foo  __attribute((depth(16)));
channel char mask  __attribute((depth(16)));
channel char Resultmask  __attribute((depth(16)));
 
__kernel void BFS_1_Reader(const __global Node *g_graph_nodes, 
					__global char *g_graph_mask){
			int tid = get_global_id(0);
		  //  printf("Kernel Read before = %d\n", tid);

			write_channel_altera(foo, g_graph_nodes[tid]);
			write_channel_altera(mask, g_graph_mask[tid]);
		//	write_channel_altera(cost, g_cost[tid]);
		//	printf("Kernel Read after = %d\n", tid);
		
}

__kernel void BFS_1_CU(const __global int *g_graph_edges,
						__global char *g_updating_graph_mask,
						__global char *g_graph_visited,
						__global int *g_cost){
				int tid = get_global_id(0);
			//	printf("Kernel comp-before-read CU = %d\n", tid); 
				Node localNode = read_channel_altera(foo);
				char localmask = read_channel_altera(mask);    // character type 0 or 1
				//int localCost = read_channel_altera(cost);
			//	printf("Kernel comp-after-read CU = %d\n", tid); 
				if(localmask){
					localmask = false;
					for( int i = localNode.starting; i< (localNode.starting + localNode.no_of_edges);i++){
						int id = g_graph_edges[i];
						if(!g_graph_visited[id]){
							g_cost[id]=g_cost[tid]+1;
							g_updating_graph_mask[id]=true;
						}
					}
			}	
	write_channel_altera(Resultmask, localmask);
	
}

__kernel void BFS_1_store(__global char* g_graph_mask){
		
				int tid = get_global_id(0);
			//	printf("Kernel Store before = %d\n", tid);
				*g_graph_mask = read_channel_altera(Resultmask);
			//	printf("Kernel Store after= %d\n", tid);

}

__kernel void BFS_2(__global char* g_graph_mask, 
		    __global char* g_updating_graph_mask, 
           	    __global char* g_graph_visited, 
		    __global char* g_over,
		    const  int no_of_nodes) {
	int tid = get_global_id(0);
	if( tid<no_of_nodes && g_updating_graph_mask[tid]){

		g_graph_mask[tid]=true;
		g_graph_visited[tid]=true;
		*g_over=true;
		g_updating_graph_mask[tid]=false;
	}
}
