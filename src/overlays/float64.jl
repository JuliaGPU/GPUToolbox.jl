# Overrides of Base methods that use Float64 to compute single- and half-precision results

Base.Experimental.@MethodTable(float64_overrides)

"""
    GPUToolbox.Overlays.float64_overrides

Method table with overrides of Base methods that use `Float64` to compute single- or
half-precision results, for devices that don't support `Float64` or where it is slow.

The overrides only replace Base's generic code, and call functions like `sin`, `cos`,
`sinpi`, `cospi` and `^(::Float32, ::Float32)`, which Base also implements using `Float64`.
Back-ends stacking this table should provide `Float64`-free implementations of those (e.g.,
using their hardware's math library) in a table higher up the stack.

See [`GPUToolbox.Overlays`](@ref) for how to use this table.
"""
float64_overrides

macro float64_override(ex)
    esc(:(Base.Experimental.@overlay($float64_overrides, $ex)))
end

# float.jl: comparisons of Float32 and 32-bit integers are performed in Float64. Widening
# the integer instead reuses the exact Float32/Int64 comparisons.
for op in (:(==), :<, :<=)
    @eval begin
        @float64_override Base.$op(x::Float32, y::Union{Int32,UInt32}) = $op(x, Int64(y))
        @float64_override Base.$op(x::Union{Int32,UInt32}, y::Float32) = $op(Int64(x), y)
        @float64_override Base.$op(x::Float16, y::Union{Int32,UInt32,Int64,UInt64}) =
            $op(Float32(x), y)
        @float64_override Base.$op(x::Union{Int32,UInt32,Int64,UInt64}, y::Float16) =
            $op(x, Float32(y))
    end
end

# div.jl: Julia 1.12 and 1.13 divide Float32 values in Float64 (JuliaLang/julia#49637).
# Use the generic implementation that replaced it (JuliaLang/julia#60497), which is exact
# as long as `eps(x/y) <= 1`, but keep Base's results for non-finite operands and zero
# quotients. `rem` does not support rounding ties away from zero or up, so handle those
# by rounding ties to even first.
@static if v"1.12-" <= VERSION < v"1.14-"
    @float64_override function Base.div(x::Float32, y::Float32,
                                        r::Union{RoundingMode{:ToZero}, RoundingMode{:Down},
                                                 RoundingMode{:Up}, RoundingMode{:Nearest},
                                                 RoundingMode{:FromZero}})
        q = x / y
        (isfinite(x) & isfinite(y) & !iszero(y)) || return round(q, r)
        d = round(q - rem(x, y, r) / y)
        return iszero(d) ? copysign(d, q) : d
    end
    @float64_override function Base.div(x::Float32, y::Float32,
                                        r::Union{RoundingMode{:NearestTiesAway},
                                                 RoundingMode{:NearestTiesUp}})
        (isfinite(x) & isfinite(y) & !iszero(y)) || return round(x / y, r)
        d = div(x, y, RoundNearest)
        m = rem(x, y, RoundNearest)
        2 * abs(m) == abs(y) || return d
        # the quotient is halfway between `d` and this other integer
        e = signbit(m) == signbit(y) ? d + 1f0 : d - 1f0
        return r === RoundNearestTiesUp ? max(d, e) : ifelse(abs(e) > abs(d), e, d)
    end
end

# math.jl: the power is computed by squaring in Float64. Use the float power instead,
# splitting exponents that aren't exactly representable as Float32 in two parts, and
# restoring the sign of the result for odd exponents.
@float64_override function Base.:(^)(x::Float32, n::Integer)
    n == -2 && return (i = inv(x); i * i)
    n == -1 && return inv(x)
    n == 0 && return one(x)
    n == 1 && return x
    n == 2 && return x * x
    n == 3 && return x * x * x
    # literal bounds: integer powers don't constant-fold in device code
    y = if -16777216 <= n <= 16777216
        abs(x)^Float32(n)
    else
        lo = rem(n, 65536)
        abs(x)^Float32(n - lo) * abs(x)^Float32(lo)
    end
    return isodd(n) ? copysign(y, x) : y
end

# math.jl: Float32 `hypot` is computed in Float64. Use the generic implementation.
@float64_override Base.Math._hypot(x::Float32, y::Float32) =
    invoke(Base.Math._hypot, Tuple{Any,Any}, x, y)

# special/trig.jl: `sind` and friends reduce their argument to [-45°, 45°] in Float32, but
# convert it to radians and evaluate the sine and cosine kernels in Float64. Keep the
# reduction, and wrap the converted argument so that the kernels use `sin` and `cos`.
struct Radians32
    x::Float32
end
# π/180 split into a Float32 value and the remainder
@float64_override Base.Math.deg2rad_ext(x::Float32) =
    Radians32(muladd(x, 0.017453292f0, x * 1.3519961f-10))
@float64_override Base.Math.sin_kernel(y::Radians32) = sin(y.x)
@float64_override Base.Math.cos_kernel(y::Radians32) = cos(y.x)

# special/trig.jl: `sincospi` uses Float64 kernels (as do `sinpi` and `cospi`, which the
# back-end is expected to override).
@float64_override Base.sincospi(x::Float32) = (sinpi(x), cospi(x))

# complex.jl: single-precision complex numbers are divided and inverted in double precision.
# Divide with the robust algorithm Base uses for ComplexF64 instead (Baudin & Smith, 2012),
# but scale large divisors further so that the reciprocal of their magnitude isn't subnormal
# (which GPUs may flush to zero). Dividing by a divisor with a positive dominant component,
# and computing the imaginary part without negating it, gives zeros the signs of Base's
# ComplexF32 results.
const TWO_M8 = Float32(0x1p-8)
const TWO_47 = Float32(0x1p47)
const TWO_M47 = Float32(0x1p-47)
@inline function robust_cdiv2(a::Float32, b::Float32, c::Float32, d::Float32, r::Float32,
                              t::Float32)
    if r != 0
        br = b * r
        return br != 0 ? (a + br) * t : a * t + (b * t) * r
    else
        # `b / c` can overflow when `d` is zero; the term is then a signed zero (`c > 0`)
        return (a + (iszero(d) ? flipsign(d, b) : d * (b / c))) * t
    end
end
@float64_override function Base.:(/)(z::ComplexF32, w::ComplexF32)
    a, b = reim(z)
    c, d = reim(w)
    if (isinf(c) | isinf(d))
        isfinite(z) && return complex(0f0 * sign(a) * sign(c), -0f0 * sign(b) * sign(d))
        return complex(NaN32, NaN32)
    end
    absa, absb, absc, absd = abs(a), abs(b), abs(c), abs(d)
    ab = absa >= absb ? absa : absb
    cd = absc >= absd ? absc : absd
    if signbit(absd <= absc ? c : d)
        a, b, c, d = -a, -b, -c, -d
    end

    s = 1f0
    if ab >= floatmax(Float32) / 2
        a *= 0.5f0; b *= 0.5f0; s *= 2f0
    elseif ab <= 2floatmin(Float32) / eps(Float32)
        a *= TWO_47; b *= TWO_47; s *= TWO_M47
    end
    if cd >= floatmax(Float32) * TWO_M8
        c *= TWO_M8; d *= TWO_M8; s *= TWO_M8
    elseif cd <= 2floatmin(Float32) / eps(Float32)
        c *= TWO_47; d *= TWO_47; s *= TWO_47
    end

    if absd <= absc
        r = d / c
        t = 1f0 / (c + d * r)
        p, q = robust_cdiv2(a, b, c, d, r, t), robust_cdiv2(b, -a, c, d, r, t)
    else
        r = c / d
        t = 1f0 / (d + c * r)
        p, q = robust_cdiv2(b, a, d, c, r, t), robust_cdiv2(-a, b, d, c, r, t)
    end
    return Complex(p * s, q * s)
end
# Invert with Smith's algorithm, after scaling by a power of two so that the reciprocal
# neither overflows nor becomes subnormal.
const TWO_64 = Float32(0x1p64)
const TWO_M64 = Float32(0x1p-64)
@inline scale_factor(m::Float32) = m >= TWO_64 ? TWO_M64 : m <= TWO_M64 ? TWO_64 : 1f0
@float64_override function Base.inv(w::ComplexF32)
    c, d = reim(w)
    (isinf(c) | isinf(d)) && return complex(copysign(0f0, c), flipsign(-0f0, d))
    s = scale_factor(max(abs(c), abs(d)))
    c, d = c * s, d * s
    if abs(d) <= abs(c)
        r = d / c
        t = inv(muladd(d, r, c))
        return Complex(t * s, -r * t * s)
    else
        r = c / d
        t = inv(muladd(c, r, d))
        return Complex(r * t * s, -t * s)
    end
end
