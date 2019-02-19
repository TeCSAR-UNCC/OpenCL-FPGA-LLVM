ALTERA_SDACCEL :=/root/intelFPGA/16.1/hld
DSA := de5net_a7
AOC := $(ALTERA_SDACCEL)/bin/aoc
CLFLAGS := --board $(DSA) --report --profile

all : srad_extract_kernel_default_dp64.aocx \
      srad_prepare_kernel_default_dp64.aocx \
      srad_reduce_kernel_default_dp64.aocx \
      srad_srad_kernel_default_dp64.aocx \
      srad_srad2_kernel_default_dp64.aocx \
      srad_compress_kernel_default_dp64.aocx \

.PHONY : all

srad_extract_kernel_default_dp64.aocx: ./kernel_gpu_opencldp64.cl
	$(AOC) $(CLFLAGS) $< -o $@ ;\

srad_prepare_kernel_default_dp64.aocx: ./kernel_gpu_opencldp64.cl
	$(AOC) $(CLFLAGS) $< -o $@ ;\

srad_reduce_kernel_default_dp64.aocx: ./kernel_gpu_opencldp64.cl
	$(AOC) $(CLFLAGS) $< -o $@ ;\

srad_srad_kernel_default_dp64.aocx: ./kernel_gpu_opencldp64.cl
	$(AOC) $(CLFLAGS) $< -o $@ ;\

srad_srad2_kernel_default_dp64.aocx: ./kernel_gpu_opencldp64.cl
	$(AOC) $(CLFLAGS) $< -o $@ ;\

srad_compress_kernel_default_dp64.aocx: ./kernel_gpu_opencldp64.cl
	$(AOC) $(CLFLAGS) $< -o $@ ;\


