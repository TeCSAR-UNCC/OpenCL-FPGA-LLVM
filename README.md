# OpenCL-FPGA-LLVM

This fork contains 3 parts:-
1. OpenCL codes of a subset of Rodinia benchmark suite, version 3.1. The fork features a clean-up of the code ported to run on the Altera Stratix V FPGA using the AOCL(Now Intel) Profiler version 16.0. For the original Rodinia benchmarks, refer to [the original home page](http://lava.cs.virginia.edu/wiki/rodinia).
2. Optimised version of the above codes using OpenCL channels run on the same FPGA platform.
3. LLVM parser used to automatically identify decouplable data from the collection of global variables.
