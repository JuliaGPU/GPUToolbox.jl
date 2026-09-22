"""
    GPUToolbox.Overlays

Method tables with device overrides of Base functionality, shared by GPU back-ends.

Some of Base's methods are unsuited for GPUs, e.g., because they compute single-precision
results in double precision, which many GPUs don't support. Back-ends replace such methods
with overlay methods in their own method table. The tables in this module collect overrides
that aren't specific to a back-end, so that back-ends can share them by stacking them
underneath their own method table.

# Usage

A back-end uses a table by stacking it underneath its own method table when it creates the
method table view for a compiler job, e.g.:

```julia
function GPUCompiler.method_table_view(job::MyCompilerJob)
    if job.config.target.supports_fp64
        GPUCompiler.StackedMethodTable(job.world, method_table)
    else
        GPUCompiler.StackedMethodTable(job.world, method_table,
                                       GPUToolbox.Overlays.float64_overrides)
    end
end
```

As shown here, the tables to use can depend on the target device. The choice may only
depend on the job's target and parameters, though, as GPUCompiler shares inference results
between jobs with equal targets and parameters.

# Lookup order

The tables in a stack are searched from the top: the first table with a method that covers
the signature of a call provides the implementation, even if a table further down the
stack, or Base, has a more specific method. The stack should be ordered from the most to
the least device-specific knowledge:

1. the back-end's own method table, with implementations that use the device's hardware or
   vendor libraries;
2. tables shared by a family of back-ends, like the one from SPIRVIntrinsics.jl, used by the
   back-ends targeting OpenCL environments;
3. the tables in this module;
4. Base.

Because of that, the overrides here only act as fallbacks: a back-end replaces one by
defining its own method with the same signature. Calls made *by* an override are again
looked up from the top of the stack, so shared overrides use the back-end's implementations
of the functions they call.

# Rules

To keep the order of the stack from affecting correctness, overrides in this module:

- implement the Base method they replace completely and correctly for their signature, so
  that the back-end's methods only have to be better, not different;
- consist of generic Julia code, without calls to device-specific functionality;
- use concrete signatures, so as not to hide more specific methods further down the stack;
- replace a method in Base, or extend a Base function for a type defined in this module
  (checked by [`audit`](@ref));
- don't overlap with the overrides in other tables in this module, so that the relative
  order of these tables doesn't matter;
- document which functions they expect the stack to provide (e.g., `Float64`-free
  elementary functions).

Back-ends can check their stack with [`audit`](@ref), e.g., as part of their tests.

# Tables

- [`float64_overrides`](@ref): overrides of Base methods that use `Float64` to compute
  single- or half-precision results.
"""
module Overlays

include("overlays/float64.jl")


## checking

"""
    Overlays.audit(tables::Core.MethodTable...; world=Base.get_world_counter())

Check a stack of method tables, ordered from highest to lowest priority, and return a
vector of `(; table, method, issue, by)` named tuples describing problems with the methods
in these tables:

- `issue = :dead`: the method overrides a function from Base, but no method of that function
  matches its signature (e.g., because Base changed the function's signature). Methods of
  other functions, and methods for types defined in the table's module, aren't checked.
- `issue = :shadowed` or `issue = :partly_shadowed`: a method in table `by`, higher up the
  stack, covers all or part of the method's signature, so that (some) calls won't reach it.
  This is often intentional, e.g., when a back-end replaces a shared fallback.
"""
function audit(tables::Core.MethodTable...; world::UInt=Base.get_world_counter())
    issues = @NamedTuple{table::Core.MethodTable, method::Method, issue::Symbol,
                         by::Union{Nothing,Core.MethodTable}}[]
    for (i, table) in enumerate(tables)
        methods = Method[]
        Base.visit(m -> push!(methods, m), table)
        for method in methods
            sig = method.sig
            for higher in tables[1:i-1]
                matches = Base._methods_by_ftype(sig, higher, -1, world)
                (matches === nothing || isempty(matches)) && continue
                issue = any(m -> m.fully_covers, matches) ? :shadowed : :partly_shadowed
                push!(issues, (; table, method, issue, by=higher))
            end
            if replaces_base_function(sig, table) &&
               isempty(Base._methods_by_ftype(sig, nothing, -1, world))
                push!(issues, (; table, method, issue=:dead, by=nothing))
            end
        end
    end
    return issues
end

# does this signature override a function from Base, rather than extend it with methods
# for types owned by the overlay table's module?
function replaces_base_function(@nospecialize(sig), table::Core.MethodTable)
    params = Base.unwrap_unionall(sig).parameters
    ft = params[1]
    isdefined(ft, :instance) || return false
    Base.moduleroot(parentmodule(ft.instance)) in (Base, Core) || return false
    for T in params[2:end]
        T = Base.unwrap_unionall(T)
        T isa DataType && parentmodule(T) === table.module && return false
    end
    return true
end

end # module Overlays
