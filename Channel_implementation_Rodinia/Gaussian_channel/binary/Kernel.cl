//#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
# pragma OPENCL EXTENSION cl_altera_channels : enable 
typedef struct latLong
    {
        float lat;
        float lng;
    } LatLong;

// Channel Creation
channel float obj0_mdev_channel_00  __attribute((depth(16)));
channel float obj0_adev_channel_00  __attribute((depth(16)));
channel float obj0_R_adev_channel_00  __attribute((depth(16)));
channel float obj0_mdev_Idy0_channel_00  __attribute((depth(16)));
channel float obj0_bdev_Idy0_channel_00  __attribute((depth(16)));
channel float obj0_adev_result_channel_00  __attribute((depth(16)));
channel float obj0_bdev_result_channel_00  __attribute((depth(16)));

__kernel void Fan2_Read(__global float* restrict m_dev, __global float* restrict a_dev, __global float* restrict b_dev, const int size, const int t)
{
	int globalIdx = get_global_id(0);
	int globalIdy = get_global_id(1);
	//printf("Fan2 IdX Read = %d\n",globalIdx);
	//printf("Fan2 IdY Read = %d\n",globalIdy);
	if(globalIdx < size-1-t && globalIdy < size-t){
		if(globalIdy == 0){
			int localbdevIdIN = globalIdx+1+t;
			int localmdevId = size*(globalIdx+1+t)+(globalIdy+t);
			write_channel_altera(obj0_mdev_Idy0_channel_00, m_dev[localmdevId]);
			write_channel_altera(obj0_bdev_Idy0_channel_00, b_dev[localbdevIdIN]);
		}
		int localXIndex = (globalIdx+1+t)* size +t;
		int localYIndex = size*t + (globalIdy+t);
		int RIndexIN = size*(globalIdx+1+t)+(globalIdy+t);
		write_channel_altera(obj0_mdev_channel_00, m_dev[localXIndex]);
		write_channel_altera(obj0_adev_channel_00, a_dev[localYIndex]);
		write_channel_altera(obj0_R_adev_channel_00, a_dev[RIndexIN]);
	}
}

__kernel void Fan2_CU(__global float* restrict b_dev, const int size, const int t)
{ 
		int globalIdx = get_global_id(0);
		int globalIdy = get_global_id(1);
		//printf("Fan2 IdX compute = %d\n",globalIdx);
		//printf("Fan2 IdY compute = %d\n",globalIdy);
		if(globalIdx < size-1-t && globalIdy < size-t){
			float localbdev = b_dev[t];
			float localb_dev_data, localm_dev_data, localm_dev, locala_dev, locala_dev2;
			if(globalIdy == 0){
				localb_dev_data = read_channel_altera(obj0_bdev_Idy0_channel_00);
				localm_dev_data = read_channel_altera(obj0_mdev_Idy0_channel_00);
				localb_dev_data -= localm_dev_data * localbdev;
				write_channel_altera(obj0_bdev_result_channel_00, localb_dev_data);
			}
			localm_dev = read_channel_altera(obj0_mdev_channel_00);
			locala_dev = read_channel_altera(obj0_adev_channel_00);
			locala_dev2 = read_channel_altera(obj0_R_adev_channel_00);
			locala_dev2 -= localm_dev * locala_dev;
			write_channel_altera(obj0_adev_result_channel_00, locala_dev2);
		}
}

__kernel void Fan2_store(__global float* restrict a_dev, __global float* restrict b_dev, const int size, const int t)
{
		int globalIdx = get_global_id(0);
		int globalIdy = get_global_id(1);
		//printf("Fan2 IdX store = %d\n",globalIdx);
		//printf("Fan2 IdY store = %d\n",globalIdy);
		if(globalIdx < size-1-t && globalIdy < size-t){
			if( globalIdy == 0){
				int localbdevId = globalIdx+1+t;
				b_dev[localbdevId] = read_channel_altera(obj0_bdev_result_channel_00);
			}
			int RIndex = size*(globalIdx+1+t)+(globalIdy+t);
			a_dev[RIndex] = read_channel_altera(obj0_adev_result_channel_00);
		}
}


