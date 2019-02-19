ALTERA_SDACCEL :=/root/intelFPGA/16.1/hld
DSA := de5net_a7
AOC := $(ALTERA_SDACCEL)/bin/aoc
CLFLAGS := --board $(DSA) --report --profile

all : srad_srad2_kernel_default_dp2.aocx \
      srad_compress_kernel_default_dp2.aocx \

.PHONY : all

srad_srad2_kernel_default_dp2.aocx: ./kernel_gpu_opencldp2.cl
	$(AOC) $(CLFLAGS) $< -o $@ ;\

srad_compress_kernel_default_dp2.aocx: ./kernel_gpu_opencldp2.cl
	$(AOC) $(CLFLAGS) $< -o $@ ;\


