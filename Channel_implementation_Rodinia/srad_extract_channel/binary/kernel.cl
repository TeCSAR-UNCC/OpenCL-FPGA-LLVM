#pragma OPENCL EXTENSION cl_altera_channels : enable
#include "../main.h"


channel float c0 __attribute((depth(16)));
channel float c1 __attribute((depth(16)));

//__attribute__((num_simd_work_items(WORK_ITEMS)))
//__attribute__((reqd_work_group_size(WORKGROUP_SIZE,1,1)))

__kernel void extract_kernel_read(__global fp* d_I){

// indexes
	int bx = get_group_id(0);	      // get current horizontal block index
	int tx = get_local_id(0);	      // get current horizontal thread index
	int ei = (bx*NUMBER_THREADS)+tx;  // unique thread id, more threads than actual elements !!!
	
	//printf("Kernel id before read= %d\n", bx);
	write_channel_altera(c0, d_I[ei]);
	//printf("Kernel id after read=%d\n",bx);
}

__kernel void extract_kernel_compute(long d_Ne){

// indexes
	int bx = get_group_id(0);	      // get current horizontal block index
	int tx = get_local_id(0);	      // get current horizontal thread index
	int ei = (bx*NUMBER_THREADS)+tx;      // unique thread id, more threads than actual elements !!!
	
	float var_dI= read_channel_altera(c0);
	
	if(ei<d_Ne){			      // do only for the number of elements, omit extra threads

		var_dI = exp(var_dI/255);   // exponentiate input IMAGE and copy to output image

	}
	
	//printf("Kernel id before compute= %d\n", bx);
	write_channel_altera(c1, var_dI);
	//printf("Kernel id after compute=%d\n",bx);
}



__kernel void extract_kernel_wb(__global fp* d_I){	      // pointer to input image (DEVICE GLOBAL MEMORY)

	// indexes
	int bx = get_group_id(0);	      // get current horizontal block index
	int tx = get_local_id(0);	      // get current horizontal thread index
	int ei = (bx*NUMBER_THREADS)+tx;      // unique thread id, more threads than actual elements !!!
	
	//printf("Kernel id before write= %d\n", bx);
	d_I[ei] = read_channel_altera(c1);
	//printf("Kernel id after write= %d\n", bx);
}
