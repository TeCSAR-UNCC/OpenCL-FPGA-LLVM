ALTERA_SDACCEL :=/root/intelFPGA/16.1/hld
DSA := de5net_a7
AOC := $(ALTERA_SDACCEL)/bin/aoc
CLFLAGS := --board $(DSA) --report --profile

all : nn_kernel.aocx \

.PHONY : all

nn_kernel.aocx: ./nearestNeighbor_kernel.cl
	$(AOC) $(CLFLAGS) $< -o $@ ;\

clean: $(SRC)
	rm -rf *.aoc* nn_kernel result*
