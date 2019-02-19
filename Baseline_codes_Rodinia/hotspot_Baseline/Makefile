include ../../common/make.config

CC = g++
CC_FLAGS = -g -O3 -Wall

GPU_SRC = hotspot_gpu.c hotspot_gpu.h OpenCL_helper_library.c OpenCL_helper_library.h

FPGA_SRC = hotspot_fpga.cpp hotspot_fpga.h timer.cc clutils_fpga.cpp utils.cpp ./util/opencl/opencl.o

all: hotspot_gpu.exe \
     hotspot_fpga.exe

hotspot_gpu.exe: $(GPU_SRC) 
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) $(GPU_SRC) -o $@ -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL

hotspot_fpga.exe: $(FPGA_SRC) 
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) $(FPGA_SRC) -o $@ -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL -I$(ALTERA_OPENCL_INC) -L$(ALTERA_OPENCL_LIB) -L/root/altera/16.0/hld/board/terasic/de5net/linux64/lib -L/root/altera/16.0/hld/host/linux64/lib -Wl,--no-as-needed -lalteracl -lalterahalmmd -lterasic_apb_16_0_mmd

clean:
	rm -f *.exe 
