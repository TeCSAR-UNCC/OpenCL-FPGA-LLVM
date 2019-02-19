#ifdef AMDAPP
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#define fp float

#define DEFAULT_ORDER 256

typedef struct record {
	int value;
} record;

typedef struct knode {
	int location;
	int indices [DEFAULT_ORDER + 1];
	int  keys [DEFAULT_ORDER + 1];
	bool is_leaf;
	int num_keys;
} knode; 

__kernel void findK(long height,
		__global knode *knodesD,
		long knodes_elem,
		__global record *recordsD,

		__global long *currKnodeD,
		__global long *offsetD,
		__global int *keysD, 
		__global record *ansD)
{

	int thid = get_local_id(0);
	int bid = get_group_id(0);

	int i;
	for(i = 0; i < height; i++){
		if((knodesD[currKnodeD[bid]].keys[thid]) <= keysD[bid] && (knodesD[currKnodeD[bid]].keys[thid+1] > keysD[bid])){
			if(knodesD[offsetD[bid]].indices[thid] < knodes_elem){
				offsetD[bid] = knodesD[offsetD[bid]].indices[thid];
			}
		}
		if(thid==0){
			currKnodeD[bid] = offsetD[bid];
		}

	}

	if(knodesD[currKnodeD[bid]].keys[thid] == keysD[bid]){
		ansD[bid].value = recordsD[knodesD[currKnodeD[bid]].indices[thid]].value;
	}

}
