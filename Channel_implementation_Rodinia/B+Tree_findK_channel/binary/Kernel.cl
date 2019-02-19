#ifdef AMDAPP
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
# pragma OPENCL EXTENSION cl_altera_channels : enable 

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

channel long foo  __attribute((depth(16)));
channel long foo1  __attribute((depth(16)));
channel int foo2  __attribute((depth(16)));
channel long foo3  __attribute((depth(16)));
channel int foo4  __attribute((depth(16)));
__kernel void findK_Read(__global long *currKnodeD,
		__global long *offsetD,
		__global int *keysD){
		
		int thid = get_local_id(0);
		int bid = get_group_id(0);	
		int i;
		
		write_channel_altera(foo, currKnodeD[bid]);
		write_channel_altera(foo1, offsetD[bid]);
		write_channel_altera(foo2, keysD[bid]);

}  

__kernel void findK_compute(long height,
		__global knode *knodesD,
		long knodes_elem,
		__global record *recordsD){
		
		int thid = get_local_id(0);
		int bid = get_group_id(0);	
		int i;
		int local_ansD;
		long local_currKnodeD = read_channel_altera(foo);
		long local_offsetD = read_channel_altera(foo1);
		long local_keysD = read_channel_altera(foo2);
		for(i = 0; i < height; i++){
			if((knodesD[local_currKnodeD].keys[thid]) <= local_keysD && (knodesD[local_currKnodeD].keys[thid+1] > local_keysD)){
				if(knodesD[local_offsetD].indices[thid] < knodes_elem){
					local_offsetD = knodesD[local_offsetD].indices[thid];
				}
			}
			if(thid==0){
				local_currKnodeD = local_offsetD;
			}
		}
		if(knodesD[local_currKnodeD].keys[thid] == local_keysD){
			local_ansD = recordsD[knodesD[local_currKnodeD].indices[thid]].value;
	}
	write_channel_altera(foo3, local_offsetD);
	write_channel_altera(foo4, local_ansD);

}

__kernel void findK_store(__global long *offsetD, 
		__global record *ansD){
		
		int thid = get_local_id(0);
		int bid = get_group_id(0);
		
		offsetD[bid] = read_channel_altera(foo3);
		ansD[bid].value = read_channel_altera(foo4);
}
