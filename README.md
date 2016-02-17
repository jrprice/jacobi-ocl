# jacobi-ocl
This project provides an OpenCL implementation of a Jacobi solver. The
OpenCL kernel is generated at runtime, and a variety of implementation
decisions can be made via a config file. This is primarily a research
tool for benchmarking different OpenCL-capable hardware and exploring
auto-tuning and performance portable code generation.

### Tuning implementation decisions
A JSON config file can be used to control how the OpenCL kernel should be implemented. The following options can be tuned:
- `wgsize`
A 2-element array describing the work-group size
- `unroll`
An integer indicating how many times to unroll the main loop
- `layout`
The memory layout of the matrix (`row-major` or `col-major`)
- `conditional`
How conditional code should be generated (`branch` or `mask`)
- `fmad`
Whether `a*b+c` should use regular arithmetic operators (`op`), `fma` builtin, or `mad` builtin
- `divide_A`
Whether division by the diagonal of the matrix should use a regular division operator (`normal`), `native_divide` builtin, or multiply by a precomputed reciprocal stored in either the global (`precompute-global`) or constant (`precompute-constant`) address spaces
- `addrspace_b`
Which address space the `b` vector should be stored in (`global` or `constant`)
- `addrspace_xold`
Which address space the `xold` vector should be stored in (`global` or `constant`)
- `integer`
Specifies whether integer variables should be signed or unsigned (`int` or `uint`)
- `relaxed_math`
A boolean specifying whether the `-cl-fast-relaxed-math` flag should be passed to the OpenCL kernel compiler
- `use_const`
A boolean specifying whether the `const` qualifer should be used for kernel arguments
- `use_restrict`
A boolean specifying whether the `restrict` qualifer should be used for kernel arguments
- `use_mad24`
A boolean specifying whether integer `a*b+c` operations should use the `mad24` builtin
- `const_norder`
A boolean specifying w
- `const_wgsize`
- `coalesce_cols`


### Example config file

    {
      // TODO
    }
