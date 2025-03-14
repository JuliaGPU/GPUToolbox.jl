export SimpleVersion, @sv_str

"""
    SimpleVersion(major, [minor])
    SimpleVersion(v::AbstractString)

A GPU-compatible version number.
"""
struct SimpleVersion
    major::UInt32
    minor::UInt32

    SimpleVersion(major, minor = 0) = new(major, minor)
end

function Base.tryparse(::Type{SimpleVersion}, v::AbstractString)
    parts = split(v, ".")
    1 <= length(parts) <= 2 || return nothing

    int_parts = map(parts) do part
        tryparse(Int, part)
    end
    any(isnothing, int_parts) && return nothing

    return SimpleVersion(int_parts...)
end

function Base.parse(::Type{SimpleVersion}, v::AbstractString)
    ver = tryparse(SimpleVersion, v)
    ver === nothing && throw(ArgumentError("invalid SimpleVersion string: '$v'"))
    return ver
end

SimpleVersion(v::AbstractString) = parse(SimpleVersion, v)

@inline function Base.isless(a::SimpleVersion, b::SimpleVersion)
    (a.major < b.major) && return true
    (a.major > b.major) && return false
    (a.minor < b.minor) && return true
    (a.minor > b.minor) && return false
    return false
end

"""
    sv"str"

Construct a `SimpleVersion` from `str`.
"""
macro sv_str(str)
    return SimpleVersion(str)
end
