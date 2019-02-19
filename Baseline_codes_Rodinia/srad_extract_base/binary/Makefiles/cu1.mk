ALTERA_SDACCEL :=/root/intelFPGA/16.1/hld
DSA := de5net_a7
AOC := $(ALTERA_SDACCEL)/bin/aoc
CLFLAGS := --board $(DSA) --report --profile

all : srad_extract_kernel_default.aocx \
      srad_prepare_kernel_default.aocx \
      srad_reduce_kernel_default.aocx \
      srad_srad_kernel_default.aocx \
      srad_srad2_kernel_default.aocx \
      srad_compress_kernel_default.aocx \

.PHONY : all

srad_extract_kernel_default.aocx: ./kernel_gpu_opencl.cl
	$(AOC) $(CLFLAGS) $< -o $@ ;\

srad_prepare_kernel_default.aocx: ./kernel_gpu_opencl.cl
	$(AOC) $(CLFLAGS) $< -o $@ ;\

srad_reduce_kernel_default.aocx: ./kernel_gpu_opencl.cl
	$(AOC) $(CLFLAGS) $< -o $@ ;\

srad_srad_kernel_default.aocx: ./kernel_gpu_opencl.cl
	$(AOC) $(CLFLAGS) $< -o $@ ;\

srad_srad2_kernel_default.aocx: ./kernel_gpu_opencl.cl
	$(AOC) $(CLFLAGS) $< -o $@ ;\

srad_compress_kernel_default.aocx: ./kernel_gpu_opencl.cl
	$(AOC) $(CLFLAGS) $< -o $@ ;\

clean: $(SRC)
	rm -rf *.aoc* srad_compress_kernel_default srad_extract_kernel_default srad_prepare_kernel_default srad_reduce_kernel_default srad_srad2_kernel_default srad_srad_kernel_default result*
