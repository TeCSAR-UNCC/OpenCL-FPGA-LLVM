/root/intelFPGA/16.1/hld/bin/aoc --board de5net_a7 -v --report --profile -D CU=2 binary/kernel_srad_cu.cl -o binary/kernel_srad_base_cu2.aocx

/root/intelFPGA/16.1/hld/bin/aoc --board de5net_a7 -v --report --profile -D CU=5 binary/kernel_srad_cu.cl -o binary/kernel_srad_base_cu5.aocx

/root/intelFPGA/16.1/hld/bin/aoc --board de5net_a7 -v --report --profile -D CU=10 binary/kernel_srad_cu.cl -o binary/kernel_srad_base_cu10.aocx

/root/intelFPGA/16.1/hld/bin/aoc --board de5net_a7 -v --report --profile -D CU=15 binary/kernel_srad_cu.cl -o binary/kernel_srad_base_cu15.aocx
