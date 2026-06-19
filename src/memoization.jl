export @memoize

@static if VERSION >= v"1.11"
    const _SlotArray{T} = AtomicMemory{Union{Nothing,T}}
else
    const _SlotArray{T} = Vector{Union{Nothing,T}}
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
    data::_SlotArray{T}, key, val::Union{Nothing,T}
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

function memoize_new_slots(::Type{T}, len) where {T}
    data = _SlotArray{T}(undef, len)
    for i in eachindex(data)
        memoize_slot_store!(data, i, nothing)
    end
    return data
end

@noinline function memoize_fixed_data!(m::FixedMemo{T}, len) where {T}
    data = @atomic :acquire m.data
    data !== nothing && return data
    memoize_should_publish() || return nothing

    @lock m.lock begin
        data = @atomic :acquire m.data
        if data === nothing
            data = memoize_new_slots(T, len)
            generating_output() && wipe_on_serialize!(m, :data, nothing)
            @atomic :release m.data = data
        end
        return data
    end
end

@noinline function memoize_slow!(constructor, m::FixedMemo{T}, key, len) where {T}
    @static if VERSION >= v"1.11"
        data = memoize_fixed_data!(m, len)
        if data === nothing
            checkbounds(1:len, key)
            return constructor()::T
        end

        cached = memoize_slot_load(data, key)
        cached !== nothing && return cached::T

        val = constructor()::T
        if memoize_should_publish()
            generating_output() && wipe_on_serialize!(m, :data, nothing)
            memoize_slot_store!(data, key, val)
        end
        return val
    else
        @lock m.lock begin
            data = @atomic :acquire m.data
            if data === nothing
                if !memoize_should_publish()
                    checkbounds(1:len, key)
                    return constructor()::T
                end
                data = memoize_new_slots(T, len)
                generating_output() && wipe_on_serialize!(m, :data, nothing)
                @atomic :release m.data = data
            end

            cached = memoize_slot_load(data, key)
            cached !== nothing && return cached::T

            val = constructor()::T
            if memoize_should_publish()
                generating_output() && wipe_on_serialize!(m, :data, nothing)
                memoize_slot_store!(data, key, val)
            end
            return val
        end
    end
end

"""
    @memoize [key::T] [maxlen=...] begin
        # expensive computation
    end::T

Low-level, no-frills memoization macro that stores values in a process-local, typed cache.
The types of the caches are derived from the syntactical type assertions.

If the `maxlen` option is specified, the `key` is assumed to be an integer, and the
cache will be a fixed-size array with length `maxlen`. Otherwise, a dictionary is used.
On Julia 1.11+, fixed-size slots use atomic per-element access. Atomic slot access is
lock-free for pointer-representable values; other values may use Julia's internal
per-element atomic fallback.
"""
macro memoize(ex...)
    code = ex[end]
    args = ex[1:end-1]

    # decode the code body
    @assert Meta.isexpr(code, :(::))
    rettyp = code.args[2]
    code = code.args[1]

    # decode the arguments
    key = nothing
    if length(args) >= 1
        arg = args[1]
        @assert Meta.isexpr(arg, :(::))
        key = (val=arg.args[1], typ=arg.args[2])
    end
    options = Dict()
    for arg in args[2:end]
        @assert Meta.isexpr(arg, :(=))
        options[arg.args[1]] = arg.args[2]
    end

    @gensym global_cache
    mod = @__MODULE__
    lazy_ty = GlobalRef(mod, :LazyInitialized)
    fixed_ty = GlobalRef(mod, :FixedMemo)
    dict_ty = GlobalRef(mod, :DictMemo)
    slot_load = GlobalRef(mod, :memoize_slot_load)
    slot_store = GlobalRef(mod, :memoize_slot_store!)
    new_slots = GlobalRef(mod, :memoize_new_slots)
    fixed_slow = GlobalRef(mod, :memoize_slow!)
    should_publish = GlobalRef(mod, :memoize_should_publish)
    generating = GlobalRef(mod, :generating_output)
    wipe = GlobalRef(mod, :wipe_on_serialize!)

    rettyp_esc = esc(rettyp)
    code_esc = esc(code)

    if key === nothing
        @eval __module__ begin
            const $global_cache = $lazy_ty{$rettyp}()
        end

        ex = quote
            let cache = $(esc(global_cache))
                val = @atomic :acquire cache.value
                if val !== nothing
                    val::$rettyp_esc
                else
                    Base.get!(cache) do
                        $code_esc::$rettyp_esc
                    end
                end
            end
        end
    elseif haskey(options, :maxlen)
        @eval __module__ begin
            const $global_cache = $fixed_ty{$rettyp}()
        end

        key_val_esc = esc(key.val)
        maxlen_esc = esc(options[:maxlen])
        ex = quote
            let cache = $(esc(global_cache)), key = $key_val_esc
                @static if VERSION >= v"1.11"
                    data = @atomic :acquire cache.data
                    if data !== nothing
                        cached_value = $slot_load(data, key)
                        if cached_value !== nothing
                            cached_value::$rettyp_esc
                        else
                            $fixed_slow(cache, key, $maxlen_esc) do
                                $code_esc::$rettyp_esc
                            end
                        end
                    else
                        $fixed_slow(cache, key, $maxlen_esc) do
                            $code_esc::$rettyp_esc
                        end
                    end
                else
                    @lock cache.lock begin
                        len = $maxlen_esc
                        publish = $should_publish()
                        data = @atomic :acquire cache.data
                        if data === nothing
                            if !publish
                                checkbounds(1:len, key)
                                $code_esc::$rettyp_esc
                            else
                                data = $new_slots($rettyp_esc, len)
                                $generating() && $wipe(cache, :data, nothing)
                                @atomic :release cache.data = data

                                cached_value = $slot_load(data, key)
                                if cached_value !== nothing
                                    cached_value::$rettyp_esc
                                else
                                    new_value = $code_esc::$rettyp_esc
                                    $generating() && $wipe(cache, :data, nothing)
                                    $slot_store(data, key, new_value)
                                    new_value
                                end
                            end
                        else
                            cached_value = $slot_load(data, key)
                            if cached_value !== nothing
                                cached_value::$rettyp_esc
                            else
                                new_value = $code_esc::$rettyp_esc
                                if publish
                                    $generating() && $wipe(cache, :data, nothing)
                                    $slot_store(data, key, new_value)
                                end
                                new_value
                            end
                        end
                    end
                end
            end
        end
    else
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
