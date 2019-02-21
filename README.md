# OpenCL-FPGA-LLVM

This fork contains 3 parts:-
1. OpenCL codes of a subset of Rodinia benchmark suite, version 3.1. The fork features baseline versions of the Rodinia codes ported to run on the Altera Stratix V FPGA using the Altera(Now Intel) SDK and AOCL Profiler version 16.0. For the original Rodinia benchmarks, refer to [the original home page](http://lava.cs.virginia.edu/wiki/rodinia).
2. Optimised version of the above codes(bothe kernel and host code) using OpenCL channels run on the same FPGA platform.
3. LLVM parser used to automatically identify decouplable data from the collection of global variables.
