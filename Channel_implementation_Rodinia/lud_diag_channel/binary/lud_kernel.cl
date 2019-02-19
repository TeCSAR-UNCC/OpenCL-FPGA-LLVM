#define BLOCK_SIZE 16

channel float c0 __attribute((depth(DEPTHSIZE)));
channel float c1 __attribute((depth(DEPTHSIZE)));


//__attribute__((num_simd_work_items(4)))
//__attribute__((reqd_work_group_size(64,1,1)))
//__attribute__((num_compute_units(CU)))

__kernel void lud_diagonal_read(__global float *m,int matrix_dim,int offset)
			 
{ 
	int tx = get_global_id(0);
	int array_offset = offset*matrix_dim+offset;
	
	//printf("Kernel id before read= %d\n", tx);
	write_channel_altera(c0, m[array_offset + tx]);
	//printf("Kernel id after read=%d\n",tx);
}

//__attribute__((num_simd_work_items(4)))
//__attribute__((reqd_work_group_size(64,1,1)))
//__attribute__((num_compute_units(CU)))

__kernel void lud_diagonal_compute(__local  float *shadow,int   matrix_dim, int   offset){

	 int tx = get_local_id(0);
	 int array_offset = offset*matrix_dim+offset;
	 float var_m= read_channel_altera(c0);
	// printf("Kernel id before compute read= %d\n",tx);
	int i=0;
	int j=0;
	
	for(i=0; i < BLOCK_SIZE; i++){
		shadow[i * BLOCK_SIZE + tx]=var_m;
		array_offset += matrix_dim;
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
  
	for(i=0; i < BLOCK_SIZE-1; i++) {
	
	  if (tx>i){
      for(j=0; j < i; j++)
        shadow[tx * BLOCK_SIZE + i] -= shadow[tx * BLOCK_SIZE + j] * shadow[j * BLOCK_SIZE + i];
		shadow[tx * BLOCK_SIZE + i] /= shadow[i * BLOCK_SIZE + i];
    }

	barrier(CLK_LOCAL_MEM_FENCE);
    if (tx>i){
	for(j=0; j < i+1; j++)
        shadow[(i+1) * BLOCK_SIZE + tx] -= shadow[(i+1) * BLOCK_SIZE + j]*shadow[j * BLOCK_SIZE + tx];
    }
    
	barrier(CLK_LOCAL_MEM_FENCE);
    }
	  array_offset = (offset+1)*matrix_dim+offset;
      for(i=1; i < BLOCK_SIZE; i++){
      var_m=shadow[i * BLOCK_SIZE + tx];
      array_offset += matrix_dim;
      
    }
    write_channel_altera(c1, var_m);
    //printf("Kernel id after compute read=%d\n",tx);
}    

//__attribute__((num_simd_work_items(4)))
//__attribute__((reqd_work_group_size(64,1,1)))
//__attribute__((num_compute_units(CU)))
    
__kernel void lud_diagonal_wb(__global float *m, int matrix_dim, int offset){

	int tx = get_local_id(0);
	int array_offset = offset*matrix_dim+offset;
	//printf("Kernel id before write= %d\n", tx);
    m[array_offset+tx] = read_channel_altera(c1);
   // printf("Kernel id after write= %d\n", tx);
}

/*/////////////////Below Kernels Commented out//////////////////
__kernel void
lud_perimeter(__global float *m, 
			  __local  float *dia,
			  __local  float *peri_row,
			  __local  float *peri_col,
			  int matrix_dim, 
			  int offset)
{
    int i,j, array_offset;
    int idx;

    int  bx = get_group_id(0);	
    int  tx = get_local_id(0);

    if (tx < BLOCK_SIZE) {
      idx = tx;
      array_offset = offset*matrix_dim+offset;
      for (i=0; i < BLOCK_SIZE/2; i++){
      dia[i * BLOCK_SIZE + idx]=m[array_offset+idx];
      array_offset += matrix_dim;
      }
    
    array_offset = offset*matrix_dim+offset;
    for (i=0; i < BLOCK_SIZE; i++) {
      peri_row[i * BLOCK_SIZE+ idx]=m[array_offset+(bx+1)*BLOCK_SIZE+idx];
      array_offset += matrix_dim;
    }

    } else {
    idx = tx-BLOCK_SIZE;
    
    array_offset = (offset+BLOCK_SIZE/2)*matrix_dim+offset;
    for (i=BLOCK_SIZE/2; i < BLOCK_SIZE; i++){
      dia[i * BLOCK_SIZE + idx]=m[array_offset+idx];
      array_offset += matrix_dim;
    }
    
    array_offset = (offset+(bx+1)*BLOCK_SIZE)*matrix_dim+offset;
    for (i=0; i < BLOCK_SIZE; i++) {
      peri_col[i * BLOCK_SIZE + idx] = m[array_offset+idx];
      array_offset += matrix_dim;
    }
  
   }
    //barrier(CLK_LOCAL_MEM_FENCE);

    if (tx < BLOCK_SIZE) { //peri-row
     idx=tx;
      for(i=1; i < BLOCK_SIZE; i++){
      for (j=0; j < i; j++)
        peri_row[i * BLOCK_SIZE + idx]-=dia[i * BLOCK_SIZE+ j]*peri_row[j * BLOCK_SIZE + idx];
    }
    } else { //peri-col
     idx=tx - BLOCK_SIZE;
     for(i=0; i < BLOCK_SIZE; i++){
      for(j=0; j < i; j++)
        peri_col[idx * BLOCK_SIZE + i]-=peri_col[idx * BLOCK_SIZE+ j]*dia[j * BLOCK_SIZE + i];
      peri_col[idx * BLOCK_SIZE + i] /= dia[i * BLOCK_SIZE+ i];
     }
   }

	//barrier(CLK_LOCAL_MEM_FENCE);
    
  if (tx < BLOCK_SIZE) { //peri-row
    idx=tx;
    array_offset = (offset+1)*matrix_dim+offset;
    for(i=1; i < BLOCK_SIZE; i++){
      m[array_offset+(bx+1)*BLOCK_SIZE+idx] = peri_row[i*BLOCK_SIZE+idx];
      array_offset += matrix_dim;
    }
  } else { //peri-col
    idx=tx - BLOCK_SIZE;
    array_offset = (offset+(bx+1)*BLOCK_SIZE)*matrix_dim+offset;
    for(i=0; i < BLOCK_SIZE; i++){
      m[array_offset+idx] =  peri_col[i*BLOCK_SIZE+idx];
      array_offset += matrix_dim;
    }
  }
//printf("Kernel id after perimeter=%d\n",tx);
}


__kernel void
lud_internal(__global float *m, 
			 __local  float *peri_row,
			 __local  float *peri_col,
			int matrix_dim, 
			int offset)
{
  
  int  bx = get_group_id(0);	
  int  by = get_group_id(1);	
  
  int  tx = get_local_id(0);
  int  ty = get_local_id(1);

  int i;
  float sum;

  int global_row_id = offset + (by+1)*BLOCK_SIZE;
  int global_col_id = offset + (bx+1)*BLOCK_SIZE;

  peri_row[ty * BLOCK_SIZE + tx] = m[(offset+ty)*matrix_dim+global_col_id+tx];
  peri_col[ty * BLOCK_SIZE + tx] = m[(global_row_id+ty)*matrix_dim+offset+tx];

  //barrier(CLK_LOCAL_MEM_FENCE);

  sum = 0;
  for (i=0; i < BLOCK_SIZE; i++)
    sum += peri_col[ty * BLOCK_SIZE + i] * peri_row[i * BLOCK_SIZE + tx];
  m[(global_row_id+ty)*matrix_dim+global_col_id+tx] -= sum;

//printf("Kernel id after internal=%d\n",tx);
}
*/




