export @memoize

@static if VERSION >= v"1.11"
    const _SlotArray{T} = AtomicMemory{Union{Nothing,Some{T}}}
else
    const _SlotArray{T} = Vector{Union{Nothing,Some{T}}}
end

mutable struct FixedMemo{T}
    @atomic data::Union{Nothing,_SlotArray{T}}
    const lock::ReentrantLock
end
FixedMemo{T}() where {T} = FixedMemo{T}(nothing, ReentrantLock())

mutable struct DictMemo{K,V}
    dict::Dict{K,V}
    const lock::ReentrantLock
end
DictMemo{K,V}() where {K,V} = DictMemo{K,V}(Dict{K,V}(), ReentrantLock())

@inline memoize_should_publish() = !generating_output() || HAS_FIELD_REPLACE

Base.@propagate_inbounds @inline function memoize_slot_load(data::_SlotArray{T}, key) where {T}
    @boundscheck checkbounds(data, key)
    @static if VERSION >= v"1.12"
        return @inbounds @atomic :acquire data[key]
    elseif VERSION >= v"1.11"
        # the `@atomic` macro only gained `AtomicMemory` element access on 1.12;
        # use the `Core.memoryref*` intrinsics as the bridge on 1.11.
        ref = Core.memoryrefnew(Core.memoryref(data), Int(key), false)
        return Core.memoryrefget(ref, :acquire, false)
    else
        return @inbounds data[key]
    end
end

Base.@propagate_inbounds @inline function memoize_slot_store!(
    data::_SlotArray{T}, key, val::Union{Nothing,Some{T}}
) where {T}
    @boundscheck checkbounds(data, key)
    @static if VERSION >= v"1.12"
        @inbounds @atomic :release data[key] = val
    elseif VERSION >= v"1.11"
        ref = Core.memoryrefnew(Core.memoryref(data), Int(key), false)
        Core.memoryrefset!(ref, val, :release, false)
    else
        @inbounds data[key] = val
    end
    return val
end

@inline memoize_grow_len(oldlen::Int, key::Int) = max(key, 2 * oldlen)

function memoize_grow_slots(::Type{T}, old::Union{Nothing,_SlotArray{T}}, newlen) where {T}
    newlen = Int(newlen)
    data = _SlotArray{T}(undef, newlen)
    n = old === nothing ? 0 : length(old)
    @inbounds for i in 1:n
        memoize_slot_store!(data, i, memoize_slot_load(old, i))
    end
    @inbounds for i in (n + 1):newlen
        memoize_slot_store!(data, i, nothing)
    end
    return data
end

@noinline function memoize_grow_slow!(constructor, m::FixedMemo{T}, key) where {T}
    key = Int(key)
    key >= 1 || throw(BoundsError())
    memoize_should_publish() || return constructor()::T

    @lock m.lock begin
        data = @atomic :acquire m.data
        if data === nothing
            data = memoize_grow_slots(T, nothing, memoize_grow_len(0, key))
            generating_output() && wipe_on_serialize!(m, :data, nothing)
            @atomic :release m.data = data
        elseif key > length(data)
            oldlen = length(data)
            newlen = memoize_grow_len(oldlen, key)
            @static if VERSION >= v"1.11"
                data = memoize_grow_slots(T, data, newlen)
                generating_output() && wipe_on_serialize!(m, :data, nothing)
                @atomic :release m.data = data
            else
                generating_output() && wipe_on_serialize!(m, :data, nothing)
                resize!(data, newlen)
                @inbounds for i in (oldlen + 1):newlen
                    memoize_slot_store!(data, i, nothing)
                end
            end
        end

        cached = memoize_slot_load(data, key)
        cached !== nothing && return Base.something(cached)::T

        val = constructor()::T
        generating_output() && wipe_on_serialize!(m, :data, nothing)
        memoize_slot_store!(data, key, Some{T}(val))
        return val
    end
end

"""
    @memoize [key=expr::K | index=expr] begin
        # expensive computation
    end::T

Low-level, no-frills memoization macro that stores values in a process-local, typed cache.
The types of the caches are derived from the syntactical type assertions.
All return values, including `nothing`, are memoized correctly.

With no leading argument, a single value is memoized. Use `key=expr::K` for dictionary
memoization keyed by values of type `K`. Use `index=expr` for array-backed memoization
with small, dense, positive integer indices; sparse or large keys should use dictionary
mode instead. Index mode grows the backing array on demand. On Julia 1.11+, index-mode
slots use atomic per-element access. Atomic slot access is lock-free for
pointer-representable values; other values may use Julia's internal per-element atomic
fallback.
"""
macro memoize(ex...)
    code = ex[end]
    args = ex[1:end-1]

    # decode the code body
    Meta.isexpr(code, :(::)) ||
        throw(ArgumentError("@memoize requires the body to end in `end::T`"))
    rettyp = code.args[2]
    code = code.args[1]

    # decode the arguments
    mode = :single
    key = nothing
    index = nothing
    for arg in args
        if !Meta.isexpr(arg, :(=))
            throw(ArgumentError(
                "@memoize positional keys are no longer supported; use `key=expr::K` " *
                "for dictionary mode or `index=expr` for array mode",
            ))
        end

        name, val = arg.args
        if name === :key
            mode === :single ||
                throw(ArgumentError("@memoize accepts only one of `key=` or `index=`"))
            Meta.isexpr(val, :(::)) ||
                throw(ArgumentError("@memoize `key=` requires a type assertion, e.g. `key=x::Int`"))
            key = (val=val.args[1], typ=val.args[2])
            mode = :dict
        elseif name === :index
            mode === :single ||
                throw(ArgumentError("@memoize accepts only one of `key=` or `index=`"))
            Meta.isexpr(val, :(::)) &&
                throw(ArgumentError("@memoize `index=` does not take a type assertion; use `index=x`"))
            index = val
            mode = :array
        elseif name === :maxlen
            throw(ArgumentError(
                "@memoize `maxlen=` is no longer supported; use `index=expr` for " *
                "dense integer indices or `key=expr::K` for dictionary mode",
            ))
        else
            throw(ArgumentError(
                "@memoize unknown option `$name`; expected `key=expr::K` or `index=expr`",
            ))
        end
    end

    @gensym global_cache
    mod = @__MODULE__
    lazy_ty = GlobalRef(mod, :LazyInitialized)
    fixed_ty = GlobalRef(mod, :FixedMemo)
    dict_ty = GlobalRef(mod, :DictMemo)
    slot_load = GlobalRef(mod, :memoize_slot_load)
    grow_slow = GlobalRef(mod, :memoize_grow_slow!)
    should_publish = GlobalRef(mod, :memoize_should_publish)
    generating = GlobalRef(mod, :generating_output)
    wipe = GlobalRef(mod, :wipe_on_serialize!)

    rettyp_esc = esc(rettyp)
    code_esc = esc(code)

    if mode === :single
        @eval __module__ begin
            const $global_cache = $lazy_ty{$rettyp}()
        end

        ex = quote
            let cache = $(esc(global_cache))
                val = @atomic :acquire cache.value
                if val !== nothing
                    Base.something(val)::$rettyp_esc
                else
                    Base.get!(cache) do
                        $code_esc::$rettyp_esc
                    end
                end
            end
        end
    elseif mode === :array
        @eval __module__ begin
            const $global_cache = $fixed_ty{$rettyp}()
        end

        index_esc = esc(index)
        ex = quote
            let cache = $(esc(global_cache)), key = Int($index_esc)
                @static if VERSION >= v"1.11"
                    data = @atomic :acquire cache.data
                    if data !== nothing && 1 <= key <= length(data)
                        cached_value = $slot_load(data, key)
                        if cached_value !== nothing
                            Base.something(cached_value)::$rettyp_esc
                        else
                            $grow_slow(cache, key) do
                                $code_esc::$rettyp_esc
                            end
                        end
                    else
                        $grow_slow(cache, key) do
                            $code_esc::$rettyp_esc
                        end
                    end
                else
                    $grow_slow(cache, key) do
                        $code_esc::$rettyp_esc
                    end
                end
            end
        end
    elseif mode === :dict
        @eval __module__ begin
            const $global_cache = $dict_ty{$(key.typ),$rettyp}()
        end

        key_val_esc = esc(key.val)
        ex = quote
            let cache = $(esc(global_cache)), key = $key_val_esc
                @lock cache.lock begin
                    dict = cache.dict
                    if haskey(dict, key)
                        dict[key]::$rettyp_esc
                    else
                        new_value = $code_esc::$rettyp_esc
                        if $should_publish()
                            $generating() && $wipe(cache, :dict, empty(dict))
                            dict[key] = new_value
                        end
                        new_value
                    end
                end
            end
        end
    end

    quote
        $ex
    end
end
