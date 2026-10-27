# Defining device-only functionality

"""
    Overlays.device_override(mod::Module, table, ex) -> Expr

Rewrite the method definition `ex` into an overlay of an existing host method, in method
table `table`. This is the implementation of back-ends' `@device_override` macros:

```julia
macro device_override(ex)
    esc(GPUToolbox.Overlays.device_override(__module__, GlobalRef(MyBackend, :method_table), ex))
end
```

The overlay must be consistent with the host method it replaces: for the same arguments, it
returns an identical value, or throws when the host method throws. In return, inference can
keep constant-evaluating calls with the host method, which e.g. keyword-argument handling
relies on. Overrides that aren't consistent, like math functions that are less accurate than
Base's, should use `Base.Experimental.@overlay` instead.
"""
function device_override(mod::Module, table, ex)
    ex = macroexpand(mod, ex)
    if VERSION >= v"1.12.0-DEV.745" || v"1.11-rc1" <= VERSION < v"1.12-"
        return :(Base.Experimental.@consistent_overlay($table, $ex))
    else
        return :(Base.Experimental.@overlay($table, $ex))
    end
end

"""
    Overlays.device_function(mod::Module, table, ex) -> Expr

Rewrite the method definition `ex`, or every definition in the block `ex`, into a method
that throws when called on the host, and an overlay with the actual implementation in method
table `table`. This is the implementation of back-ends' `@device_function` macros:

```julia
macro device_function(ex)
    esc(GPUToolbox.Overlays.device_function(__module__, GlobalRef(MyBackend, :method_table), ex))
end
```

Use it for functions that only work on the device, i.e., that (indirectly) call
device-specific functionality, like intrinsics or the back-end's runtime. Code compiled for
the host, e.g., by sysimage builds or ahead-of-time compilers that compile every method,
then only contains the throwing method. See [`host_leaks`](@ref) to find functions that
should be device functions.

The overlay is not consistent with the host method (see [`device_override`](@ref)), or
inference could constant-fold calls to the host method's error. Docstrings document the
host method, i.e., the function. Definitions can be `@generated`. In a block, definitions
in conditionals are rewritten, but not those in other constructs, like loops.
"""
function device_function(mod::Module, table, ex)
    isdef(ex) = Meta.isexpr(ex, :function) ||
                (Meta.isexpr(ex, :(=)) && Meta.isexpr(ex.args[1], (:call, :where, :(::))))
    isdoc(m) = m === GlobalRef(Core, Symbol("@doc")) || m === Symbol("@doc") ||
               (Meta.isexpr(m, :.) && m.args[end] == QuoteNode(Symbol("@doc")))

    # rewrite definitions before expanding macros where possible: the shape of the
    # expansion of `@doc` has changed repeatedly (e.g., Julia 1.13 wraps the definition in
    # `if true; val = ...; end`), and a definition that escapes the rewrite ends up as a
    # host method that contains device code
    function rewrite(ex)
        if isdef(ex)
            msg = "This function is not intended for use on the CPU"
            stub = Expr(ex.head, ex.args[1], Expr(:call, GlobalRef(Base, :error), msg))
            return quote
                Base.@__doc__ $stub
                Base.Experimental.@overlay($table, $ex)
            end
        elseif Meta.isexpr(ex, :macrocall) && isdoc(ex.args[1])
            return Expr(:macrocall, ex.args[1:end-1]..., rewrite(ex.args[end]))
        elseif Meta.isexpr(ex, :macrocall)
            # other macros, e.g., `@inline` or `@generated`
            return rewrite(macroexpand(mod, ex; recursive=false))
        elseif Meta.isexpr(ex, (:block, :if, :elseif))
            return Expr(ex.head, map(rewrite, ex.args)...)
        elseif Meta.isexpr(ex, :(=)) && ex.args[1] isa Symbol
            # assignment of a definition to a temporary, from the expansion of `@doc`
            return Expr(:(=), ex.args[1], rewrite(ex.args[2]))
        else
            return ex
        end
    end
    return rewrite(ex)
end
