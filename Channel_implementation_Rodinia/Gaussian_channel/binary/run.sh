/root/intelFPGA/16.1/hld/bin/aoc --board de5net_a7 -D SIMD_WORK_ITEMS=1 -D WORK_GROUP_SIZE=256 --report --profile Fan1.cl -o gaussian_Fan1_default.aocx

/root/intelFPGA/16.1/hld/bin/aoc --board de5net_a7 -D SIMD_WORK_ITEMS=2 -D WORK_GROUP_SIZE=256 --report --profile Fan1.cl -o gaussian_Fan1_default_2.aocx

/root/intelFPGA/16.1/hld/bin/aoc --board de5net_a7 -D SIMD_WORK_ITEMS=4 -D WORK_GROUP_SIZE=256 --report --profile Fan1.cl -o gaussian_Fan1_default_4.aocx

/root/intelFPGA/16.1/hld/bin/aoc --board de5net_a7 -D SIMD_WORK_ITEMS=8 -D WORK_GROUP_SIZE=256 --report --profile Fan1.cl -o gaussian_Fan1_default_8.aocx

/root/intelFPGA/16.1/hld/bin/aoc --board de5net_a7 -D SIMD_WORK_ITEMS=16 -D WORK_GROUP_SIZE=256 --report --profile Fan1.cl -o gaussian_Fan1_default_16.aocx

/root/intelFPGA/16.1/hld/bin/aoc --board de5net_a7 -D SIMD_WORK_ITEMS=1 -D WORK_GROUP_SIZE=256 --report --profile Fan2.cl -o gaussian_Fan2_default.aocx

/root/intelFPGA/16.1/hld/bin/aoc --board de5net_a7 -D SIMD_WORK_ITEMS=2 -D WORK_GROUP_SIZE=256 --report --profile Fan2.cl -o gaussian_Fan2_default_2.aocx

/root/intelFPGA/16.1/hld/bin/aoc --board de5net_a7 -D SIMD_WORK_ITEMS=4 -D WORK_GROUP_SIZE=256 --report --profile Fan2.cl -o gaussian_Fan2_default_4.aocx

/root/intelFPGA/16.1/hld/bin/aoc --board de5net_a7 -D SIMD_WORK_ITEMS=8 -D WORK_GROUP_SIZE=256 --report --profile Fan2.cl -o gaussian_Fan2_default_8.aocx

/root/intelFPGA/16.1/hld/bin/aoc --board de5net_a7 -D SIMD_WORK_ITEMS=16 -D WORK_GROUP_SIZE=256 --report --profile Fan2.cl -o gaussian_Fan2_default_16.aocx
