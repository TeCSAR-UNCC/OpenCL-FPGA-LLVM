ALTERA_SDACCEL :=/root/intelFPGA/16.1/hld
DSA := de5net_a7
AOC := $(ALTERA_SDACCEL)/bin/aoc
CLFLAGS := --board $(DSA) --report --profile

all : b+tree_findK_default.aocx \


.PHONY : all

b+tree_findK_default.aocx: ./kernel_gpu_opencl.cl
	$(AOC) $(CLFLAGS)  $< -o $@ ;\

clean: $(SRC)
	rm -rf *.aoc* b+tree_findK_default* result*
