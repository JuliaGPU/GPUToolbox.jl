# Overrides of Base methods that use 128-bit integers to compute results for narrower ones

Base.Experimental.@MethodTable(int128_overrides)

"""
    GPUToolbox.Overlays.int128_overrides

Method table with overrides of Base methods that use `Int128` or `UInt128` to compute results
for narrower integers, for devices that don't support 128-bit integers.

This currently covers the comparisons of `Rational`s with each other, with integers and with
floating-point numbers, which Base performs on the 128-bit products of their 64-bit numerators
and denominators.

See [`GPUToolbox.Overlays`](@ref) for how to use this table.
"""
int128_overrides

macro int128_override(ex)
    esc(:(Base.Experimental.@overlay($int128_overrides, $ex)))
end

const BitInteger64 = Base.BitInteger64

# the exact product of two integers, as whether it is negative and the high and low words of
# its magnitude. the magnitudes are multiplied in 32-bit halves, like Base's `widemul` does on
# 32-bit platforms.
@inline function wide_product(x::BitInteger64, y::BitInteger64)
    a = unsigned(abs(x)) % UInt64   # also for `typemin(x)`, which `abs` returns as is
    b = unsigned(abs(y)) % UInt64
    a0, a1 = a & 0xffffffff, a >>> 32
    b0, b1 = b & 0xffffffff, b >>> 32
    w0 = a0 * b0
    t = a1 * b0 + (w0 >>> 32)
    w1 = a0 * b1 + (t & 0xffffffff)
    hi = a1 * b1 + (t >>> 32) + (w1 >>> 32)
    lo = (w0 & 0xffffffff) | (w1 << 32)
    negative = (signbit(x) ⊻ signbit(y)) & !(iszero(hi) & iszero(lo))
    return negative, hi, lo
end

# compare two 128-bit magnitudes with `<`, `<=` or `cmp`
@inline wide_compare(op, xhi, xlo, yhi, ylo) = xhi == yhi ? op(xlo, ylo) : op(xhi, yhi)

# compare the products `a * b` and `c * d` with `<` or `<=`, as `op(widemul(a, b), widemul(c, d))`
@inline function compare_products(op, a::BitInteger64, b::BitInteger64,
                                  c::BitInteger64, d::BitInteger64)
    # the products of 32-bit integers fit in 64 bits
    max(sizeof(a), sizeof(b), sizeof(c), sizeof(d)) <= 4 && return op(widemul(a, b), widemul(c, d))

    xneg, xhi, xlo = wide_product(a, b)
    yneg, yhi, ylo = wide_product(c, d)
    xneg == yneg || return xneg
    return xneg ? wide_compare(op, yhi, ylo, xhi, xlo) : wide_compare(op, xhi, xlo, yhi, ylo)
end

const Rational64 = Rational{<:BitInteger64}

# rational.jl: comparisons with rationals and integers multiply out the denominators
for op in (:<, :<=)
    @eval begin
        @int128_override Base.$op(x::Rational64, y::Rational64) =
            x.den == y.den ? $op(x.num, y.num) :
                             compare_products($op, x.num, y.den, x.den, y.num)
        @int128_override Base.$op(x::Rational64, y::BitInteger64) =
            compare_products($op, x.num, one(x.num), x.den, y)
        @int128_override Base.$op(x::BitInteger64, y::Rational64) =
            compare_products($op, x, y.den, y.num, one(y.num))
    end
end

# the number of bits in a 128-bit magnitude
@inline wide_bits(hi::UInt64, lo::UInt64) =
    iszero(hi) ? 64 - leading_zeros(lo) : 128 - leading_zeros(hi)

# shift a 128-bit magnitude left by `0 <= k < 128` bits
@inline wide_shl(hi::UInt64, lo::UInt64, k::Int) =
    k >= 64 ? (lo << (k - 64), zero(lo)) : ((hi << k) | (lo >>> (64 - k)), lo << k)

# rational.jl: comparisons with floating-point numbers multiply the significand of each by
# the denominator of the other. this is a port of Base's implementation, using the
# (sign, high word, low word) representation of the products.
for rel in (:<, :<=, :cmp), (Tx, Ty) in ((Rational64, Base.IEEEFloat),
                                          (Base.IEEEFloat, Rational64))
    @eval @int128_override function Base.$rel(x::$Tx, y::$Ty)
        if isnan(x)
            $(rel === :cmp ? :(return isnan(y) ? 0 : 1) : :(return false))
        end
        if isnan(y)
            $(rel === :cmp ? :(return -1) : :(return false))
        end

        xn, xp, xd = Base.decompose(x)
        yn, yp, yd = Base.decompose(y)

        if xd < 0
            xn = -xn
            xd = -xd
        end
        if yd < 0
            yn = -yn
            yd = -yd
        end

        xneg, xhi, xlo = wide_product(xn, yd)
        yneg, yhi, ylo = wide_product(yn, xd)
        xs = xneg ? -1 : Int(!(iszero(xhi) & iszero(xlo)))
        ys = yneg ? -1 : Int(!(iszero(yhi) & iszero(ylo)))

        if xs != ys
            return $rel(xs, ys)
        elseif xs == 0
            # both are zero or ±Inf
            return $rel(xn, yn)
        end

        xb, yb = wide_bits(xhi, xlo) + xp, wide_bits(yhi, ylo) + yp

        if xb == yb
            if xp > yp
                xhi, xlo = wide_shl(xhi, xlo, xp - yp)
            else
                yhi, ylo = wide_shl(yhi, ylo, yp - xp)
            end
            return xs > 0 ? wide_compare($rel, xhi, xlo, yhi, ylo) :
                            wide_compare($rel, yhi, ylo, xhi, xlo)
        else
            return xs > 0 ? $rel(xb, yb) : $rel(yb, xb)
        end
    end
end
