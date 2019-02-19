include ../common/make.config

#C compiler
CC = g++

# Altera OpenCL
ALTERA_SDACCEL :=/root/intelFPGA/16.1/hld
DSA := de5net_a7
AOC := $(ALTERA_SDACCEL)/bin/aoc

SRC_FPGA = srad_fpga.cpp clutils_fpga.cpp utils.cpp util/graphics/resize.c util/graphics/graphics.c util/opencl/opencl.c


#CC_FLAGS = -g -O3
CC_FLAGS = -Wall

all: clean \
	 srad_fpga.exe \

     

srad_fpga.exe:
	$(CC) $(CC_FLAGS) -g -lOpenCL  -fopenmp $(SRC_FPGA) -o $@ -I$(OPENCL_INC) -L$(OPENCL_LIB) -I$(ALTERA_OPENCL_INC) -L$(ALTERA_OPENCL_LIB) -L/root/altera/16.0/hld/board/terasic/de5net/linux64/lib -L/root/altera/16.0/hld/host/linux64/lib -Wl,--no-as-needed -lalteracl -lalterahalmmd -lterasic_apb_16_0_mmd

clean: 
	rm -f *.exe 
