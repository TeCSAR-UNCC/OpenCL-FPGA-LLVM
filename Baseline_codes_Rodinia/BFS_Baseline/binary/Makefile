ALTERA_SDACCEL :=/root/intelFPGA/16.1/hld
DSA := de5net_a7
AOC := $(ALTERA_SDACCEL)/bin/aoc
CLFLAGS := --board $(DSA) --report

all : BFS_1_default.aocx \

.PHONY : all


BFS_1_default.aocx: ./Kernels.cl
	$(AOC) $(CLFLAGS) $< -o $@ ;\

date
	
clean: $(SRC)
	rm -rf *.aoc* BFS_1_default* result*
	
