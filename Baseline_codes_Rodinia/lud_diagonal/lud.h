
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


	
#define REC_LENGTH 49 // size of a record in db

float *OpenCl_LUD(
	cl_context context,
	int sourcesize,
	int matrix_dim,float *m,int timing);
	
void lud_base(float *a, int size);

