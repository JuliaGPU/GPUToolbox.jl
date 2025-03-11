# GPUToolbox.jl

*Common utilities shared between the various Julia GPU backends.*

| **Build Status**                                               | **Coverage**                    |
|:--------------------------------------------------------------:|:-------------------------------:|
| [![][gha-img]][gha-url] [![PkgEval][pkgeval-img]][pkgeval-url] | [![][codecov-img]][codecov-url] |

[gha-img]: https://github.com/JuliaGPU/GPUToolbox.jl/actions/workflows/Test.yml/badge.svg?branch=main
[gha-url]: https://github.com/JuliaGPU/GPUToolbox.jl/actions

[pkgeval-img]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/G/GPUToolbox.svg
[pkgeval-url]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/G/GPUToolbox.html

[codecov-img]: https://codecov.io/gh/JuliaGPU/GPUToolbox.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/JuliaGPU/GPUToolbox.jl

## Functionality

This package currently exports the following:
- `SimpleVersion`: a GPU-compatible version number
- `@sv_str`: constructs a SimpleVersion from a string
- `@checked`: Add to a function definition to generate an unchecked and a checked version.
- `@debug_ccall`: like `ccall` but prints the ccall, its arguments, and its return value
- `@gcsafe_ccall`: like `@ccall` but marking it safe for the GC to run.
-  Suffix literals like `1i8` (`i8`, `i16`, `i32`, `u8`, `u16`, `u32`) for constructing literals of a certain type.

For more details on a specific symbol, check out its docstring in the Julia REPL.
