export i8, i16, i32, u8, u16, u32
# helper type for writing smaller than Int64 literals
struct Literal{T} end
Base.:(*)(x::Number, ::Type{Literal{T}}) where {T} = T(x)
const i8 = Literal{Int8}
const i16 = Literal{Int16}
const i32 = Literal{Int32}
const u8 = Literal{Int8}
const u16 = Literal{Int16}
const u32 = Literal{Int32}
