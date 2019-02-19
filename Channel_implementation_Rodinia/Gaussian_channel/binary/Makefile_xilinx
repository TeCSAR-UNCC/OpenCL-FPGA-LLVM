XILINX_SDACCEL :=/opt/Xilinx/SDAccel/2016.1
XILINX_OPENCL := $(XILINX_SDACCEL)
DSA := xilinx:adm-pcie-7v3:1ddr:3.0
XOCC := $(XILINX_SDACCEL)/bin/xocc
CLFLAGS := -s --xdevice $(DSA)

all : gaussian_Fan1_default.xclbin\
      gaussian_Fan2_default.xclbin

.PHONY : all

gaussian_Fan1_default.xclbin: ./gaussianElim_kernels.cl
	$(XOCC) $(CLFLAGS) --nk Fan1:1 -k Fan1 $< -o $@ ;\

gaussian_Fan2_default.xclbin: ./gaussianElim_kernels.cl
	$(XOCC) $(CLFLAGS) --nk Fan2:1 -k Fan2 $< -o $@ ;\
