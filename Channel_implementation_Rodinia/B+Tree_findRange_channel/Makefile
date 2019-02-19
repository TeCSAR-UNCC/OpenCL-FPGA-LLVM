include ../../common/make.config

C_C = g++
OCL_LIB = -lOpenCL

FPGA_SRC = b+tree_fpga.cpp clutils_fpga.cpp common.h utils.cpp timer.cc ./util/opencl/opencl.o

CC_FLAGS = -g -O3

all: b+tree_fpga.exe


b+tree_fpga.exe: $(FPGA_SRC)
	$(C_C) $(CC_FLAGS) -fopenmp $(FPGA_SRC) -o $@ -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL 

clean:
	rm -rf *.exe ./util/opencl/*.o output.txt
# ========================================================================================================================================================================================================200
#	END
# ========================================================================================================================================================================================================200
