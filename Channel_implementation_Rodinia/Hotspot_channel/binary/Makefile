ALTERA_SDACCEL :=/root/intelFPGA/16.1/hld
DSA := de5net_a7
AOC := $(ALTERA_SDACCEL)/bin/aoc
CLFLAGS := --board $(DSA) --report --profile

all : hotspot_default.aocx \

.PHONY : all

hotspot_default.aocx: ./Kernel.cl
	$(AOC) $(CLFLAGS) $< -o $@ ;\

clean: $(SRC)
	rm -rf *.aoc* hotspot_default* result*
