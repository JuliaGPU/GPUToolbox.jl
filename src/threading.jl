export LazyInitialized

@inline generating_output() = ccall(:jl_generating_output, Cint, ()) != 0

const HAS_FIELD_REPLACE = isdefined(Base, :OncePerProcess)

@inline function wipe_on_serialize!(@nospecialize(obj), field::Symbol, @nospecialize(val))
    @static if HAS_FIELD_REPLACE
        ccall(:jl_set_precompile_field_replace, Cvoid, (Any, Any, Any), obj, field, val)
    end
    return
end

"""
    LazyInitialized{T}()

A thread-safe, lazily-initialized wrapper for a value of type `T`. Initialize and fetch the
value by calling `get!`. The constructor is ensured to only be called once.

This type is intended for lazy initialization of e.g. global structures, without using
`__init__`. It is similar to protecting accesses using a lock, but is much cheaper.

Any value of type `T`, including `nothing` when allowed by `T`, may be stored.

"""
mutable struct LazyInitialized{T}
    @atomic value::Union{Nothing,Some{T}}
    const lock::ReentrantLock
end

LazyInitialized{T}() where {T} = LazyInitialized{T}(nothing, ReentrantLock())

@inline function Base.get!(constructor::Base.Callable, x::LazyInitialized{T}) where {T}
    val = @atomic :acquire x.value
    val !== nothing && return Base.something(val)::T
    return slow_init!(constructor, x)::T
end

@noinline function slow_init!(constructor, x::LazyInitialized{T}) where {T}
    if generating_output() && !HAS_FIELD_REPLACE
        return constructor()::T
    end

    @lock x.lock begin
        val = @atomic :acquire x.value
        val !== nothing && return Base.something(val)::T
        result = constructor()::T
        generating_output() && wipe_on_serialize!(x, :value, nothing)
        @atomic :release x.value = Some{T}(result)
        return result
    end
end
