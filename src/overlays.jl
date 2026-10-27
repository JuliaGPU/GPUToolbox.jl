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

# Device-only functionality

Back-ends also define methods that only work on the device, in their own method table, with
the macros built on [`device_override`](@ref) (replacing a host method) and
[`device_function`](@ref) (a function that throws on the host). Their methods with device
code must stay out of the global method table: sysimage builds compile every method there
with a concrete signature for the host, and an image that refers to device symbols doesn't
load (nor does a package image on Windows). Back-ends can check that with
[`host_leaks`](@ref), e.g., as part of their tests.

# Tables

- [`float64_overrides`](@ref): overrides of Base methods that use `Float64` to compute
  single- or half-precision results.
- [`int128_overrides`](@ref): overrides of Base methods that use 128-bit integers to compute
  results for narrower ones.
"""
module Overlays

include("overlays/float64.jl")
include("overlays/int128.jl")
include("overlays/device.jl")


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

"""
    Overlays.host_leaks(mod::Module) -> Vector{@NamedTuple{method::Method, symbol::String}}

Find the methods of `mod` and its submodules that sysimage builds compile for the host, and
whose code may reach device-only functionality. Such methods should be device functions
(see [`device_function`](@ref)).

This considers the methods that Julia 1.12+ compiles when building a sysimage: those with a
compileable signature, or the compileable signatures obtained by splitting unions in their
arguments and type parameters, and that aren't `@generated`. It follows calls that aren't
inlined into other methods of `mod`, and reports the device symbol it finds first:

- a function that an `llvmcall` declares, other than an LLVM intrinsic: on the host, it only
  resolves if the dynamic linker happens to find a symbol with that name. Device code uses
  these for intrinsics that the GPU compiler lowers, like `julia.gpu.state_getter`, or that
  the device's runtime provides, like `__spirv_BuiltInGlobalInvocationId`;
- an LLVM intrinsic of a GPU target, like `llvm.nvvm.read.ptx.sreg.tid.x`.

The check is conservative: it reports declarations in `llvmcall` IR without checking that
the IR uses them. It doesn't follow calls into other packages, so it misses device code in a
dependency that is only compiled when called from `mod`, e.g., because the method that
contains it doesn't have a concrete signature (and thus is no root of the dependency's own
check).

On Julia 1.10 and 1.11, which have no global method table to search, only the methods of
functions that are bound in `mod` or its submodules are considered.
"""
function host_leaks(mod::Module)
    leaks = @NamedTuple{method::Method, symbol::String}[]
    owned(m::Module) = m === mod || (parentmodule(m) !== m && owned(parentmodule(m)))
    cache = Dict{Any,Union{Nothing,String}}()
    for method in owned_methods(owned, mod)
        for (atype, sparams) in compiled_signatures(method)
            visited = Set{Any}()
            symbol = find_device_symbol(owned, method, atype, sparams, cache, visited)
            if symbol !== nothing
                push!(leaks, (; method, symbol))
                break
            end
            # the search covered everything these methods reach
            for key in visited
                cache[key] = nothing
            end
        end
    end
    return leaks
end

# the methods of `mod` and its submodules, for which `owned` returns true
function owned_methods(owned, mod::Module)
    methods = Method[]
    if isdefined(Core, :methodtable)
        Base.visit(m -> owned(m.module) && push!(methods, m), Core.methodtable)
    else
        modules = Module[mod]
        i = 1
        while i <= length(modules)
            m = modules[i]
            i += 1
            for name in names(m; all=true)
                isdefined(m, name) || continue
                val = getfield(m, name)
                if val isa Module
                    if val !== m && parentmodule(val) === m && !(val in modules)
                        push!(modules, val)
                    end
                elseif val isa Function || val isa Type
                    for method in Base.methods(val)
                        if owned(method.module) && !(method in methods)
                            push!(methods, method)
                        end
                    end
                end
            end
        end
    end
    return methods
end

# the signatures, with their static parameters, that sysimage builds compile for `method`,
# following Julia's `infer_all_method_defs!`
function compiled_signatures(method::Method)
    sigs = Tuple{Any,Core.SimpleVector}[]
    isdefined(method, :source) || return sigs   # @generated
    startswith(String(method.name), "@") && return sigs
    compileable(sig, sparams) = Core.Compiler.isa_compileable_sig(sig, sparams, method)

    sig = method.sig
    if sig isa DataType && compileable(sig, Core.svec())
        push!(sigs, (sig, Core.svec()))
        return sigs
    end

    # split unions in the arguments, unless that results in more than 8 signatures
    params = Base.unwrap_unionall(sig).parameters
    for T in params
        if !(T isa Union) && T isa DataType && !isconcretetype(T) && !(T <: Type)
            return sigs     # splitting unions doesn't help
        end
    end
    choices = [T isa Union ? Base.uniontypes(T) : Any[T] for T in params]
    candidates = if 1 < prod(length, choices) <= 8
        [Base.rewrap_unionall(Tuple{types...}, sig) for types in Iterators.product(choices...)]
    else
        Any[sig]
    end

    # instantiate type variables bounded by a union or by a concrete type
    for candidate in candidates
        vars = TypeVar[]
        body = candidate
        while body isa UnionAll
            push!(vars, body.var)
            body = body.body
        end
        bounds = map(vars) do var
            ub = var.ub
            while ub isa TypeVar
                ub = ub.ub
            end
            ub isa Union ? Base.uniontypes(ub) :
                (ub isa Type && isabstracttype(ub) && !(ub <: Type)) ? nothing : Any[ub]
        end
        any(isnothing, bounds) && continue
        for values in Iterators.product(bounds...)
            atype = try
                foldl((T, value) -> T{value}, values; init=candidate)
            catch
                continue    # doesn't satisfy the other bounds
            end
            atype isa DataType || continue
            sparams = ccall(:jl_type_intersection_with_env, Any, (Any, Any),
                            atype, method.sig)[2]::Core.SimpleVector
            compileable(atype, sparams) && push!(sigs, (atype, sparams))
        end
    end
    return sigs
end

const device_intrinsics = ("llvm.nvvm.", "llvm.amdgcn.", "llvm.r600.", "llvm.spv.",
                           "llvm.air.", "llvm.genx.")
is_device_symbol(name) =
    !startswith(name, "llvm.") || any(prefix -> startswith(name, prefix), device_intrinsics)

# the first device symbol that code compiled for `method` specialized to `atype` refers to,
# if any. `visited` holds the methods this search has already entered, which it doesn't
# enter again: they're either still being searched further up, or finished without finding
# a symbol. That a method is clean is only known once the whole search finds nothing.
function find_device_symbol(owned, method::Method, @nospecialize(atype),
                            sparams::Core.SimpleVector,
                            cache::Dict{Any,Union{Nothing,String}}, visited::Set{Any})
    key = (method, atype)
    haskey(cache, key) && return cache[key]
    key in visited && return nothing
    push!(visited, key)
    symbol = search_device_symbol(owned, method, atype, sparams, cache, visited)
    symbol === nothing || (cache[key] = symbol)
    return symbol
end

function search_device_symbol(owned, method::Method, @nospecialize(atype),
                              sparams::Core.SimpleVector,
                              cache::Dict{Any,Union{Nothing,String}}, visited::Set{Any})
    interp = Core.Compiler.NativeInterpreter()
    ci = Core.Compiler.typeinf_code(interp, method, atype, sparams, true)
    ci isa Tuple && (ci = first(ci))    # Julia 1.10 and 1.11 also return the return type
    ci isa Core.CodeInfo || return nothing
    for stmt in ci.code
        stmt isa Expr || continue
        if stmt.head === :foreigncall && is_llvmcall_cc(stmt.args[5])
            name = foreigncall_name(stmt.args[1])
            if name !== nothing && is_device_symbol(name)
                return name
            end
        elseif stmt.head === :call && is_llvmcall(stmt.args[1])
            ir = llvmcall_ir(ci, stmt.args[2])
            ir === nothing && continue
            for m in eachmatch(r"declare[^@\n]*@\"?([^\s(\"]+)", ir)
                is_device_symbol(m[1]) && return String(m[1])
            end
        elseif stmt.head === :invoke
            mi = stmt.args[1]
            while !(mi isa Core.MethodInstance) && isdefined(mi, :def)
                mi = mi.def     # a CodeInstance, or an ABI override (1.12+)
            end
            mi isa Core.MethodInstance && mi.def isa Method || continue
            owned(mi.def.module) || continue
            symbol = find_device_symbol(owned, mi.def, mi.specTypes, mi.sparam_vals,
                                        cache, visited)
            symbol === nothing || return symbol
        end
    end
    return nothing
end

is_llvmcall_cc(cc) = cc isa QuoteNode ? is_llvmcall_cc(cc.value) :
                     cc isa Tuple ? is_llvmcall_cc(first(cc)) : cc === :llvmcall
is_llvmcall(f) = f === Core.Intrinsics.llvmcall || f == GlobalRef(Base, :llvmcall) ||
                 f == GlobalRef(Core.Intrinsics, :llvmcall)

# the value of a constant global, as used in the arguments of `llvmcall`
function constant(arg)
    if arg isa GlobalRef && isdefined(arg.mod, arg.name) && isconst(arg.mod, arg.name)
        return getfield(arg.mod, arg.name)
    end
    # Julia 1.15 refers to the binding partition that a global access resolves to
    if isdefined(Core, :BindingPartition) && arg isa Core.BindingPartition &&
       Base.is_defined_const_binding(Base.binding_kind(arg))
        return Base.partition_restriction(arg)
    end
    return arg
end

# the name of a function called with `ccall(name, llvmcall, ...)`
function foreigncall_name(name)
    name = constant(name)
    name isa QuoteNode && return foreigncall_name(name.value)
    name isa Tuple && return isempty(name) ? nothing : foreigncall_name(first(name))
    Meta.isexpr(name, :tuple) && return isempty(name.args) ? nothing : foreigncall_name(name.args[1])
    name isa Symbol && return String(name)
    name isa String || return nothing
    return startswith(name, "extern ") ? name[8:end] : name
end

# the IR of a call to `llvmcall`, a string or a tuple with the module and the entry point
function llvmcall_ir(ci::Core.CodeInfo, arg)
    arg = constant(arg)
    arg isa Core.SSAValue && return llvmcall_ir(ci, ci.code[arg.id])
    arg isa QuoteNode && return llvmcall_ir(ci, arg.value)
    arg isa String && return arg
    arg isa Tuple && return isempty(arg) ? nothing : llvmcall_ir(ci, first(arg))
    if Meta.isexpr(arg, :call) && length(arg.args) >= 2 &&
       (arg.args[1] === Core.tuple || arg.args[1] == GlobalRef(Core, :tuple))
        return llvmcall_ir(ci, arg.args[2])
    end
    return nothing
end

end # module Overlays
