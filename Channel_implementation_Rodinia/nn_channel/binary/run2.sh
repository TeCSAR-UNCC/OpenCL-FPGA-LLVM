/root/intelFPGA/16.1/hld/bin/aoc --board de5net_a7 --report -D WORK_ITEMS=1 -D WORKGROUP_SIZE=64 -D COMPUTE_UNITS=1 --profile nn_base.cl -o nn_scusdp.aocx

/root/intelFPGA/16.1/hld/bin/aoc --board de5net_a7 --report -D WORK_ITEMS=2 -D WORKGROUP_SIZE=64 -D COMPUTE_UNITS=1 --profile nn_base.cl -o nn_scumdp2.aocx

/root/intelFPGA/16.1/hld/bin/aoc --board de5net_a7 --report -D WORK_ITEMS=4 -D WORKGROUP_SIZE=64 -D COMPUTE_UNITS=1 --profile nn_base.cl -o nn_scumdp4.aocx

/root/intelFPGA/16.1/hld/bin/aoc --board de5net_a7 --report -D WORK_ITEMS=8 -D WORKGROUP_SIZE=64 -D COMPUTE_UNITS=1 --profile nn_base.cl -o nn_scumdp8.aocx

/root/intelFPGA/16.1/hld/bin/aoc --board de5net_a7 --report -D WORK_ITEMS=16 -D WORKGROUP_SIZE=64 -D COMPUTE_UNITS=1 --profile nn_base.cl -o nn_scumdp16.aocx

#/root/intelFPGA/16.1/hld/bin/aoc --board de5net_a7 --report -D WORK_ITEMS=2 -D WORKGROUP_SIZE=64 -D COMPUTE_UNITS=2 --profile nn_base.cl -o nn_mcu2mdp2.aocx

#/root/intelFPGA/16.1/hld/bin/aoc --board de5net_a7 --report -D WORK_ITEMS=4 -D WORKGROUP_SIZE=64 -D COMPUTE_UNITS=2 --profile nn_base.cl -o nn_mcu2mdp4.aocx

#/root/intelFPGA/16.1/hld/bin/aoc --board de5net_a7 --report -D WORK_ITEMS=8 -D WORKGROUP_SIZE=64 -D COMPUTE_UNITS=2 --profile nn_base.cl -o nn_mcu2mdp8.aocx

#/root/intelFPGA/16.1/hld/bin/aoc --board de5net_a7 --report -D WORK_ITEMS=16 -D WORKGROUP_SIZE=64 -D COMPUTE_UNITS=2 --profile nn_base.cl -o nn_mcu2mdp16.aocx

#/root/intelFPGA/16.1/hld/bin/aoc --board de5net_a7 --report -D WORK_ITEMS=2 -D WORKGROUP_SIZE=64 -D COMPUTE_UNITS=3 --profile nn_base.cl -o nn_mcu3mdp2.aocx

#/root/intelFPGA/16.1/hld/bin/aoc --board de5net_a7 --report -D WORK_ITEMS=4 -D WORKGROUP_SIZE=64 -D COMPUTE_UNITS=3 --profile nn_base.cl -o nn_mcu3mdp4.aocx

#/root/intelFPGA/16.1/hld/bin/aoc --board de5net_a7 --report -D WORK_ITEMS=8 -D WORKGROUP_SIZE=64 -D COMPUTE_UNITS=3 --profile nn_base.cl -o nn_mcu3mdp8.aocx
