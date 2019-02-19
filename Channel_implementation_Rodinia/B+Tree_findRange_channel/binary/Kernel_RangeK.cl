#define fp float

#define DEFAULT_ORDER_2 256
#pragma OPENCL EXTENSION cl_altera_channels : enable

typedef struct knode {
	int location;
	int indices [DEFAULT_ORDER_2 + 1];
	int  keys [DEFAULT_ORDER_2 + 1];
	bool is_leaf;
	int num_keys;
} knode;

channel long foo  __attribute((depth(16)));
channel int foo1  __attribute((depth(16)));
channel long foo2  __attribute((depth(16)));
channel long foo3  __attribute((depth(16)));
channel int foo4  __attribute((depth(16)));
channel long foo5  __attribute((depth(16)));
channel long foo6  __attribute((depth(16)));
channel long foo7  __attribute((depth(16)));
channel long foo8  __attribute((depth(16)));
channel long foo9  __attribute((depth(16)));
channel long foo10  __attribute((depth(16)));
__kernel void findRangeK_Read(__global long *currKnodeD,
							  __global long *lastKnodeD,
							__global int *startD,
						    __global int *endD,
						    __global int *RecstartD){
	
	// private thread IDs
	int thid = get_local_id(0);
	int bid = get_group_id(0);
	
	write_channel_altera(foo, currKnodeD[bid]);
	write_channel_altera(foo1, startD[bid]);
	write_channel_altera(foo3, lastKnodeD[bid]);
	write_channel_altera(foo4, endD[bid]);
	write_channel_altera(foo6, RecstartD[bid]);
}

__kernel void findRangeK_compute(long height,__global knode *knodesD,long knodes_elem){
	
	int thid = get_local_id(0);
	int bid = get_group_id(0);
	long local_offsetD, local_offset_2D, ReclenDResult, RecstartDResult;
	long local_currKnodeD = read_channel_altera(foo);
	long local_startD = read_channel_altera(foo1);
	long local_lastKnodeD = read_channel_altera(foo3);
	long local_endD = read_channel_altera(foo4);
	long local_RecstartD = read_channel_altera(foo6);
	int i;
	for(i = 0; i < height; i++){
		if((knodesD[local_currKnodeD].keys[thid] <= local_startD) && (knodesD[local_currKnodeD].keys[thid+1] > local_startD)){
			if(knodesD[local_currKnodeD].indices[thid] < knodes_elem){
				local_offsetD = knodesD[local_currKnodeD].indices[thid];
			}
		}
		if((knodesD[local_lastKnodeD].keys[thid] <= local_endD) && (knodesD[local_lastKnodeD].keys[thid+1] > local_endD)){
			if(knodesD[local_lastKnodeD].indices[thid] < knodes_elem){
				local_offset_2D = knodesD[local_lastKnodeD].indices[thid];
			}
		}
		if(thid==0){
			local_currKnodeD = local_offsetD;
			local_lastKnodeD = local_offset_2D;
		}
		
	}
	
	if(knodesD[local_currKnodeD].keys[thid] == local_startD){
		RecstartDResult = knodesD[local_currKnodeD].indices[thid];
	}
	
	if(knodesD[local_lastKnodeD].keys[thid] == local_endD){
		ReclenDResult = knodesD[local_lastKnodeD].indices[thid] - local_RecstartD+1;
	}
	write_channel_altera(foo7, RecstartDResult);
	write_channel_altera(foo8, ReclenDResult);
	write_channel_altera(foo9, local_offsetD);
	write_channel_altera(foo10, local_offset_2D);
	
}

__kernel void findRangeK_store(__global long *offsetD, __global long *offset_2D, __global int *RecstartD,__global int *ReclenD){
	int thid = get_local_id(0);
	int bid = get_group_id(0);
	RecstartD[bid] = read_channel_altera(foo7);
	ReclenD[bid] = read_channel_altera(foo8);
	offsetD[bid] = read_channel_altera(foo9);
	offset_2D[bid] = read_channel_altera(foo10);
}
	
