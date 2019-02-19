
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_intel_channels : enable

typedef struct type_{
        float lat;
        float lng;
    } LatLong;


channel LatLong c0 __attribute((depth(16)));
channel float c1 __attribute((depth(16)));


__attribute__((reqd_work_group_size(64,1,1)))
//__attribute__((num_compute_units(2)))
__kernel void NearestNeighbor_read(__global LatLong *d_locations) {

	int globalId = get_global_id(0);
	//printf("Kernel id before read= %d\n", globalId);
	write_channel_intel(c0, d_locations[globalId]);
	//printf("Kernel id after read=%d\n",globalId);
		
}

__attribute__((reqd_work_group_size(64,1,1)))
//__attribute__((num_compute_units(2)))
__kernel void NearestNeighbor_compute(const float lat, const float lng){
	
	int globalId = get_global_id(0);
	LatLong loc_lat = read_channel_intel(c0);
	//printf("Kernel id before compute read= %d\n", globalId);
    float d_dist = (lat-loc_lat.lat)*(lat-loc_lat.lat)+(lng-loc_lat.lng)*(lng-loc_lat.lng);
    float d_distances = (float)sqrt(d_dist);
	write_channel_intel(c1, d_distances);
	//printf("Kernel id after compute write= %d\n", globalId);

}

__attribute__((reqd_work_group_size(64,1,1)))
//__attribute__((num_compute_units(2)))
__kernel void NearestNeighbor_wb(__global float * d_distances) {

	int globalId = get_global_id(0);
	//printf("Kernel id before write= %d\n", globalId);
    *(d_distances+globalId) = read_channel_intel(c1);
    //printf("Kernel id after write= %d\n", globalId);
}


/*
///////////////////////////////////ORIGINAL CODE/////////////////////////////////////////////
//#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

typedef struct latLong
    {
        float lat;
        float lng;
    } LatLong;

//__attribute__((num_simd_work_items(2)))
__kernel void NearestNeighbor(__global LatLong *d_locations,
							  __global float *d_distances,
							  const int numRecords,
							  const float lat,
							  const float lng) {
	 int globalId = get_global_id(0);
							  
     if (globalId < numRecords) {
         __global LatLong *latLong = d_locations+globalId;
    
         __global float *dist=d_distances+globalId;
         *dist = (float)sqrt((lat-latLong->lat)*(lat-latLong->lat)+(lng-latLong->lng)*(lng-latLong->lng));
	 }
}
*/
