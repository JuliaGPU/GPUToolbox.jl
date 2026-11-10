# Overrides of Base methods that throw exceptions with messages built at run time

Base.Experimental.@MethodTable(throw_overrides)

"""
    GPUToolbox.Overlays.throw_overrides

Method table with overrides of Base's helper functions that throw an exception with a
message built at run time, e.g., by interpolating values into a string. GPU compilers can't
compile that code, so these overrides throw the same exception with a constant message
instead, using [`device_throw`](@ref).

This currently covers two helpers that `Rational` arithmetic calls on Julia 1.10 and 1.11,
whose messages include the integer type. Julia 1.11 removed one of them, and Julia 1.12
builds the message of the other lazily, so the table is empty on later versions.

See [`GPUToolbox.Overlays`](@ref) for how to use this table.
"""
throw_overrides

macro throw_override(ex)
    esc(:(Base.Experimental.@overlay($throw_overrides, $ex)))
end

# rational.jl: Julia 1.10 interpolates the type into the message (the helper was removed in
# Julia 1.11)
@static if isdefined(Base, :__throw_rational_argerror_typemin)
    @throw_override @noinline Base.__throw_rational_argerror_typemin(::Type{<:Integer}) =
        @device_throw ArgumentError("invalid rational: denominator can't be typemin")
end

# rational.jl: Julia 1.12 builds this message with a `LazyString`
@static if VERSION < v"1.12-"
    @throw_override @noinline Base.__throw_rational_numerator_typemin(::Type{<:Integer}) =
        @device_throw OverflowError("rational numerator is typemin")
end
