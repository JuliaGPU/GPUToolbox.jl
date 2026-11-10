# Throwing exceptions from shared overrides

"""
    Overlays.device_throw(::Type{E}, ::Val{reason}) where {E<:Exception, reason}

Throw an exception of type `E` with the message `String(reason)`. The overrides in this
module throw their exceptions with this function (as `@device_throw ArgumentError("...")`),
so that back-ends can report them like the exceptions they throw themselves: GPU compilers
drop the exception object, so a plain `throw(E(msg))` is only reported as an exception of
unknown type.

This method throws `E(String(reason))`, which assumes that `E` has a constructor that takes
the message (as `ArgumentError`, `OverflowError` and `ErrorException` do, but not, e.g.,
`DomainError`, which would print the message as the offending value).

Back-ends replace it in their own method table. Both arguments are constants, so a
back-end's method can pass them to a reporting function that expects compile-time constants,
e.g.:

```julia
@device_override @noinline function GPUToolbox.Overlays.device_throw(::Type{E}, ::Val{reason}) where {E, reason}
    record_exception(Val(nameof(E)), Val(reason))
    throw(nothing)
end
```

or, when the reporting is done by a macro that requires string literals, using a generated
function:

```julia
@device_override @generated function GPUToolbox.Overlays.device_throw(::Type{E}, ::Val{reason}) where {E, reason}
    :(@gputhrow \$(String(nameof(E))) \$(String(reason)))
end
```

A back-end's method must throw, like this one does.
"""
@noinline device_throw(::Type{E}, reason::Val) where {E<:Exception} =
    throw(E(reason_string(reason)))

# the reason as a constant string: converting a Symbol at run time requires a call into the
# runtime, which GPU compilers can't compile
@generated reason_string(::Val{reason}) where {reason} = String(reason)

# throw an exception through `device_throw`, as `@device_throw ArgumentError("message")`
macro device_throw(ex)
    if !(Meta.isexpr(ex, :call, 2) && ex.args[2] isa String)
        throw(ArgumentError("`@device_throw` expects an exception type applied to a string " *
                            "literal, e.g., `ArgumentError(\"message\")`"))
    end
    E, reason = ex.args
    return esc(:($device_throw($E, $(Val(Symbol(reason))))))
end
