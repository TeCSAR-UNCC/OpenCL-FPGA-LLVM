ALTERA_SDACCEL :=/root/intelFPGA/16.1/hld
DSA := de5net_a7
AOC := $(ALTERA_SDACCEL)/bin/aoc
CLFLAGS := --board $(DSA) --report --profile

all : srad_extract_kernel_default_2.aocx \
      
.PHONY : all

srad_extract_kernel_default_2.aocx: ./kernel_gpu_openclcu2.cl
	$(AOC) $(CLFLAGS) $< -o $@ ;\


