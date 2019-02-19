#define fp float

#define DEFAULT_ORDER_2 256

typedef struct knode {
	int location;
	int indices [DEFAULT_ORDER_2 + 1];
	int  keys [DEFAULT_ORDER_2 + 1];
	bool is_leaf;
	int num_keys;
} knode; 

//========================================================================================================================================================================================================200
//	findRangeK function
//========================================================================================================================================================================================================200
__kernel void findRangeK(long height,
			__global knode *knodesD,
			long knodes_elem,

			__global long *currKnodeD,
			__global long *offsetD,
			__global long *lastKnodeD,
			__global long *offset_2D,
			__global int *startD,
			__global int *endD,
			__global int *RecstartD, 
			__global int *ReclenD)
{

	// private thread IDs
	int thid = get_local_id(0);
	int bid = get_group_id(0);

	int i;
	for(i = 0; i < height; i++){

		if((knodesD[currKnodeD[bid]].keys[thid] <= startD[bid]) && (knodesD[currKnodeD[bid]].keys[thid+1] > startD[bid])){
			if(knodesD[currKnodeD[bid]].indices[thid] < knodes_elem){
				offsetD[bid] = knodesD[currKnodeD[bid]].indices[thid];
			}
		}
		if((knodesD[lastKnodeD[bid]].keys[thid] <= endD[bid]) && (knodesD[lastKnodeD[bid]].keys[thid+1] > endD[bid])){
			if(knodesD[lastKnodeD[bid]].indices[thid] < knodes_elem){
				offset_2D[bid] = knodesD[lastKnodeD[bid]].indices[thid];
			}
		}
		//__syncthreads();
		//barrier(CLK_LOCAL_MEM_FENCE);
		if(thid==0){
			currKnodeD[bid] = offsetD[bid];
			lastKnodeD[bid] = offset_2D[bid];
		}
		//	__syncthreads();
		//barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Find the index of the starting record
	if(knodesD[currKnodeD[bid]].keys[thid] == startD[bid]){
		RecstartD[bid] = knodesD[currKnodeD[bid]].indices[thid];
	}
	//	__syncthreads();
	//barrier(CLK_LOCAL_MEM_FENCE);

	if(knodesD[lastKnodeD[bid]].keys[thid] == endD[bid]){
		ReclenD[bid] = knodesD[lastKnodeD[bid]].indices[thid] - RecstartD[bid]+1;
	}

}
