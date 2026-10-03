export @enum_without_prefix

using Republic: @public

## redeclare enum values without a prefix

# this is useful when enum values from an underlying C library, typically prefixed for the
# lack of namespacing in C, are to be used in Julia where we do have module namespacing.
macro enum_without_prefix(ex...)
    # destructure keyword arguments
    call = ex[end-1:end]
    kwargs = map(ex[1:end-2]) do kwarg
        if kwarg isa Symbol
            :($kwarg = $kwarg)
        elseif Meta.isexpr(kwarg, :(=))
            kwarg
        else
            throw(ArgumentError("Invalid keyword argument '$kwarg'"))
        end
    end

    visibility = nothing
    for kwarg in kwargs
        key, val = kwarg.args
        if key == :visibility
            val isa QuoteNode && (val = val.value)
            visibility = val::Symbol
        else
            throw(ArgumentError("Invalid keyword argument '$key'"))
        end
    end

    enum, prefix = call
    if isa(enum, Symbol)
        mod = __module__
    elseif Meta.isexpr(enum, :(.))
        mod = getfield(__module__, enum.args[1])
        enum = enum.args[2].value
    else
        error("Do not know how to refer to $enum")
    end
    enum = getfield(mod, enum)
    prefix = String(prefix)

    ex = quote end
    for instance in instances(enum)
        name = String(Symbol(instance))
        @assert startswith(name, prefix)
        short = Symbol(name[length(prefix)+1:end])
        push!(ex.args, :(const $short = $(mod).$(Symbol(name))))
        if visibility == :export
            push!(ex.args, :(export $short))
        elseif visibility == :public
            push!(ex.args, :($(@__MODULE__).@public $short))
        end
    end

    return esc(ex)
end
