include ../common/make.config

#C compiler
CC = g++
CC_FLAGS = -g2 -O3 -Wall

# Altera OpenCL
ALTERA_SDACCEL :=/root/intelFPGA/16.1/hld
DSA := de5net_a7
AOC := $(ALTERA_SDACCEL)/bin/aoc

SRC_FPGA = lud_fpga.cpp clutils_fpga.cpp utils.cpp

all: clean \
     lud_fpga.exe \
     

lud_fpga.exe:
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) -g -lOpenCL  -fopenmp $(SRC_FPGA) ./common.c -I../common -o $@  -I$(OPENCL_INC) -L$(OPENCL_LIB) -I$(ALTERA_OPENCL_INC) -L$(ALTERA_OPENCL_LIB)
	

clean: 
	rm -f *.exe 
