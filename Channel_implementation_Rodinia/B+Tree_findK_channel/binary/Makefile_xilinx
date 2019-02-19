XILINX_SDACCEL :=/opt/Xilinx/SDAccel/2016.1
XILINX_OPENCL := $(XILINX_SDACCEL)
DSA := xilinx:adm-pcie-7v3:1ddr:3.0
XOCC := $(XILINX_SDACCEL)/bin/xocc
CLFLAGS := -s --xdevice $(DSA)

all : b+tree_findK_default.xclbin \
      b+tree_findRangeK_default.xclbin \


.PHONY : all

b+tree_findK_default.xclbin: ./kernel_gpu_opencl.cl
	$(XOCC) $(CLFLAGS) --nk findK:1 -k findK $< -o $@ ;\

b+tree_findRangeK_default.xclbin: ./kernel_gpu_opencl_2.cl
	$(XOCC) $(CLFLAGS) --nk findRangeK:1 -k findRangeK $< -o $@ ;\
