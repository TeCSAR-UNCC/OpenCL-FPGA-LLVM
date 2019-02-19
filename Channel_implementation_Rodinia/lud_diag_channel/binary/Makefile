ALTERA_SDACCEL :=/root/intelFPGA/16.1/hld
DSA := de5net_a7
AOC := $(ALTERA_SDACCEL)/bin/aoc
CLFLAGS := --board $(DSA) --report --profile

all : ludemu.aocx \

.PHONY : all

ludemu.aocx: ./lud_kernel.cl
	$(AOC) $(CLFLAGS) $< -o $@ ;\

clean: $(SRC)
	rm -rf *.aoc* lud_kernel result*
