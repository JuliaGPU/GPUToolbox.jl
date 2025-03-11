export i8, i16, i32, u8, u16, u32
# helper type for writing smaller than Int64 literals

"""
    Literal{T}

Construct an object that can be used to convert literals to other types.
One can use the object in suffix form `1*i8` or `1i8` to perform the conversion.

## Exported constants
- `i8`: Convert to `Int8`
- `i16`: Convert to `Int16`
- `i32`: Convert to `Int32`
- `u8`: Convert to `UInt8`
- `u16`: Convert to `UInt16`
- `u32`: Convert to `UInt32`
"""
struct Literal{T} end
Base.:(*)(x::Number, ::Type{Literal{T}}) where {T} = T(x)

const i8 = Literal{Int8}
const i16 = Literal{Int16}
const i32 = Literal{Int32}
const u8 = Literal{UInt8}
const u16 = Literal{UInt16}
const u32 = Literal{UInt32}
