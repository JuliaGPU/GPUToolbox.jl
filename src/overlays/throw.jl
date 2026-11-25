# Overrides of Base methods that throw exceptions with messages built at run time

Base.Experimental.@MethodTable(throw_overrides)

"""
    GPUToolbox.Overlays.throw_overrides

Method table with overrides of Base's helper functions that throw an exception with a
message built at run time, e.g., by interpolating values into a string. GPU compilers can't
compile that code, so these overrides throw the same exception with a constant message
instead, using [`device_throw`](@ref).

This covers `Base.throw_domerr_powbysq`, which integer exponentiation calls for a negative
power, and two helpers that `Rational` arithmetic calls on Julia 1.10 and 1.11, whose
messages include the integer type (Julia 1.11 removed one of them, and Julia 1.12 builds the
message of the other lazily).

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

# intfuncs.jl: a single override for Base's three methods (for `::Any`, `::Integer` and
# `::AbstractMatrix` bases), whose messages differ only in their advice. Overrides of one
# table shouldn't shadow one another: when Julia 1.12 or 1.13 loads a package image, it only
# trusts a kernel's edge to an overlay method that no more specific method of the same table
# intersects, and verifies other edges against the global method table, which doesn't have
# the overlay, so the kernel is dropped and compiled again in every session
# (JuliaLang/julia#64727).
@throw_override @noinline Base.throw_domerr_powbysq(::Any, p) =
    @device_throw DomainError("Cannot raise an integer to a negative power")
