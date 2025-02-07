# utilities for calling foreign functionality more conveniently

export @checked, @debug_ccall, @gcsafe_ccall


## function wrapper for checking the return value of a function

"""
    @checked function foo(...)
        rv = ...
        return rv
    end

Macro for wrapping a function definition returning a status code. Two versions of the
function will be generated: `foo`, with the function execution wrapped by an invocation of
the `check` function (to be implemented by the caller of this macro), and `unchecked_foo`
where no such invocation is present and the status code is returned to the caller.
"""
macro checked(ex)
    # parse the function definition
    @assert Meta.isexpr(ex, :function)
    sig = ex.args[1]
    @assert Meta.isexpr(sig, :call)
    body = ex.args[2]
    @assert Meta.isexpr(body, :block)

    # make sure these functions are inlined
    pushfirst!(body.args, Expr(:meta, :inline))

    # generate a "safe" version that performs a check
    safe_body = quote
        @inline
        check() do
            $body
        end
    end
    safe_sig = Expr(:call, sig.args[1], sig.args[2:end]...)
    safe_def = Expr(:function, safe_sig, safe_body)

    # generate a "unchecked" version that returns the error code instead
    unchecked_sig = Expr(:call, Symbol("unchecked_", sig.args[1]), sig.args[2:end]...)
    unchecked_def = Expr(:function, unchecked_sig, body)

    return esc(:($safe_def, $unchecked_def))
end

## version of ccall that prints the ccall, its arguments and its return value

macro debug_ccall(ex)
    @assert Meta.isexpr(ex, :(::))
    call, ret = ex.args
    @assert Meta.isexpr(call, :call)
    target, argexprs... = call.args
    args = map(argexprs) do argexpr
        @assert Meta.isexpr(argexpr, :(::))
        argexpr.args[1]
    end

    ex = Expr(:macrocall, Symbol("@ccall"), __source__, ex)

    # avoid task switches
    io = :(Core.stdout)

    return quote
        print($io, $(string(target)), '(')
        for (i, arg) in enumerate(($(map(esc, args)...),))
            i > 1 && print($io, ", ")
            render_arg($io, arg)
        end
        print($io, ')')

        rv = $(esc(ex))

        println($io, " = ", rv)
        for (i, arg) in enumerate(($(map(esc, args)...),))
            if arg isa Base.RefValue
                println($io, " $i: ", arg[])
            end
        end
        rv
    end
end

render_arg(io, arg) = print(io, arg)
render_arg(io, arg::AbstractArray) = summary(io, arg)
render_arg(io, arg::Base.RefValue{T}) where {T} = print(io, "Ref{", T, "}")


## version of ccall that calls jl_gc_safe_enter|leave around the inner ccall

# TODO: replace with JuliaLang/julia#49933 once merged

function ccall_macro_lower(func, rettype, types, args, nreq)
    # instead of re-using ccall or Expr(:foreigncall) to perform argument conversion,
    # we need to do so ourselves in order to insert a jl_gc_safe_enter|leave
    # just around the inner ccall

    cconvert_exprs = []
    cconvert_args = []
    for (typ, arg) in zip(types, args)
        var = gensym("$(func)_cconvert")
        push!(cconvert_args, var)
        push!(cconvert_exprs, :($var = Base.cconvert($(esc(typ)), $(esc(arg)))))
    end

    unsafe_convert_exprs = []
    unsafe_convert_args = []
    for (typ, arg) in zip(types, cconvert_args)
        var = gensym("$(func)_unsafe_convert")
        push!(unsafe_convert_args, var)
        push!(unsafe_convert_exprs, :($var = Base.unsafe_convert($(esc(typ)), $arg)))
    end

    call = quote
        $(unsafe_convert_exprs...)

        gc_state = @ccall(jl_gc_safe_enter()::Int8)
        ret = ccall(
            $(esc(func)), $(esc(rettype)), $(Expr(:tuple, map(esc, types)...)),
            $(unsafe_convert_args...)
        )
        @ccall(jl_gc_safe_leave(gc_state::Int8)::Cvoid)
        ret
    end

    return quote
        @inline
        $(cconvert_exprs...)
        GC.@preserve $(cconvert_args...) $(call)
    end
end

"""
    @gcsafe_ccall ...

Call a foreign function just like `@ccall`, but marking it safe for the GC to run. This is
useful for functions that may block, so that the GC isn't blocked from running, but may also
be required to prevent deadlocks (see JuliaGPU/CUDA.jl#2261).
"""
macro gcsafe_ccall(expr)
    return ccall_macro_lower(Base.ccall_macro_parse(expr)...)
end
