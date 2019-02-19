include ../common/make.config

#C compiler
CC = g++
CC_FLAGS = -g2 -O3 -Wall

# Altera OpenCL
ALTERA_SDACCEL :=/root/intelFPGA/16.1/hld
DSA := de5net_a7
AOC := $(ALTERA_SDACCEL)/bin/aoc

#SRC_FPGA = lud_fpga.cpp clutils_fpga.cpp utils.cpp
#SRC_FPGA = lud_fpgacu1.cpp clutils_fpga.cpp utils.cpp
SRC_FPGA = lud_fpgacu2.cpp clutils_fpga.cpp utils.cpp
#SRC_FPGA = lud_fpgacu3.cpp clutils_fpga.cpp utils.cpp
#SRC_GPU = nearestNeighbor_gpu.cpp clutils_gpu.cpp utils.cpp
#SRC_FPGA = nearestNeighbor_fpga_old.cpp clutils_fpgabase.cpp utils.cpp

#CC_FLAGS = -g -O3
CC_FLAGS = -Wall

all: clean \
	 lud_fpga.exe \
#	 nn_gpu.exe \
     
#nn_gpu.exe:
#	$(CC) $(CC_FLAGS) -lOpenCL  -fopenmp $(SRC_GPU) -o $@ -I$(OPENCL_INC) -L$(OPENCL_LIB)
	
#nn_fpga.exe:nearestNeighbor_fpga.cpp
lud_fpga.exe:
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) -g -lOpenCL  -fopenmp $(SRC_FPGA) ./common.c -I../common -o $@  -I$(OPENCL_INC) -L$(OPENCL_LIB) -I$(ALTERA_OPENCL_INC) -L$(ALTERA_OPENCL_LIB)
	

	
	
#  nearestNeighbor_fpga.cpp ./clutils_gpu.cpp ./utils.cpp

clean: 
	rm -f *.exe 
