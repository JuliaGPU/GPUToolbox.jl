using Test
using GPUToolbox
using InteractiveUtils
using IOCapture

@testset "GPUToolbox.jl" begin
    @testset "SimpleVersion" begin
        # Construct
        sv1 = SimpleVersion(42)
        @test sv1.major == 42
        @test sv1.minor == 0

        sv2 = SimpleVersion(42, 5)
        @test sv2.major == 42
        @test sv2.minor == 5

        sv3 = SimpleVersion("41")
        @test sv3.major == 41
        @test sv3.minor == 0

        sv4 = SimpleVersion("41.6")
        @test sv4.major == 41
        @test sv4.minor == 6

        sv5 = sv"43"
        @test sv5.major == 43
        @test sv5.minor == 0

        sv6 = sv"43.7"
        @test sv6.major == 43
        @test sv6.minor == 7

        @test_throws "invalid SimpleVersion string" SimpleVersion("42.5.0")
        @test_throws "invalid SimpleVersion string" SimpleVersion("bsrg")
        @test_throws "invalid SimpleVersion string" eval(:(sv"42.5.0"))
        @test_throws "invalid SimpleVersion string" eval(:(sv"bsrg"))

        # Comparison
        @test   sv3 < sv1  # (a.major < b.major)
        @test !(sv1 < sv3) # (a.major > b.major)
        @test   sv1 < sv2  # (a.minor < b.minor)
        @test !(sv4 < sv3) # (a.minor > b.minor)
        @test !(sv1 < sv1) # Default
        @test !(sv2 > sv2) # Default
    end

    @testset "Literals" begin
        @test 1i8 === Int8(1)
        @test 1i16 === Int16(1)
        @test 1i32 === Int32(1)
        @test_throws InexactError 128i8

        @test 1u8 === UInt8(1)
        @test 1u16 === UInt16(1)
        @test 1u32 === UInt32(1)
        @test_throws InexactError 256u8
    end

    @testset "gcsafe_ccall" begin
        function gc_safe_ccall()
            # jl_rand is marked as JL_NOTSAFEPOINT
            @gcsafe_ccall jl_rand()::UInt64
        end

        let llvm = sprint(code_llvm, gc_safe_ccall, ())
            # check that the call works
            @test gc_safe_ccall() isa UInt64
            # v1.10 is hard to test since ccall are just raw runtime pointers
            if VERSION >= v"1.11"
                if !GPUToolbox.HAS_CCALL_GCSAFE
                    # check for the gc_safe store
                    @test occursin("jl_gc_safe_enter", llvm)
                    @test occursin("jl_gc_safe_leave", llvm)
                else
                    @test occursin("store atomic i8 2", llvm)
                end
            end
        end
    end

    @testset "@enum_without_prefix" begin
        mod = @eval module $(gensym())
            using GPUToolbox
            @enum MY_ENUM MY_ENUM_VALUE
            @enum_without_prefix MY_ENUM MY_
        end

        @test mod.ENUM_VALUE == mod.MY_ENUM_VALUE
    end

    @testset "LazyInitialized" begin
        # Basic functionality
        lazy = LazyInitialized{Int}()
        @test get!(lazy) do
            42
        end == 42

        # Should return same value on subsequent calls
        @test get!(lazy) do
            error("Should not be called")
        end == 42

        # Test with validator
        valid = Ref(true)
        lazy_with_validator = LazyInitialized{Int}() do val
            valid[]
        end
        @test get!(lazy_with_validator) do
            1
        end == 1
        @test get!(lazy_with_validator) do
            2
        end == 1
        valid[] = false
        @test get!(lazy_with_validator) do
            3
        end == 3
    end

    @testset "@memoize" begin
        # Test basic memoization without key
        call_count = Ref(0)
        function test_basic_memo()
            @memoize begin
                call_count[] += 1
                42
            end::Int
        end

        @test test_basic_memo() == 42
        @test call_count[] == 1
        @test test_basic_memo() == 42
        @test call_count[] == 1  # Should not increment

        # Test memoization with key (dictionary)
        dict_call_count = Ref(0)
        function test_dict_memo(x)
            @memoize x::Int begin
                dict_call_count[] += 1
                x * 2
            end::Int
        end

        @test test_dict_memo(5) == 10
        @test dict_call_count[] == 1
        @test test_dict_memo(5) == 10
        @test dict_call_count[] == 1  # Should not increment
        @test test_dict_memo(3) == 6
        @test dict_call_count[] == 2  # Should increment for new key

        # Test memoization with maxlen (vector)
        vec_call_count = Ref(0)
        function test_vec_memo(x)
            @memoize x::Int maxlen=10 begin
                vec_call_count[] += 1
                x * 3
            end::Int
        end

        @test test_vec_memo(1) == 3
        @test vec_call_count[] == 1
        @test test_vec_memo(1) == 3
        @test vec_call_count[] == 1  # Should not increment
        @test test_vec_memo(2) == 6
        @test vec_call_count[] == 2  # Should increment for new index
    end

    @testset "@checked" begin
        # Test checked function generation
        check_called = Ref(false)
        check_result = Ref{Any}(nothing)

        check(f) = begin
            check_called[] = true
            result = f()
            check_result[] = result
            result == 0 ? nothing : error("Check failed with code $result")
        end

        @checked function test_checked_func(should_fail::Bool)
            should_fail ? 1 : 0
        end

        # Test successful case
        check_called[] = false
        @test test_checked_func(false) === nothing
        @test check_called[]
        @test check_result[] == 0

        # Test failure case
        check_called[] = false
        @test_throws "Check failed with code 1" test_checked_func(true)
        @test check_called[]
        @test check_result[] == 1

        # Test unchecked version
        @test unchecked_test_checked_func(false) == 0
        @test unchecked_test_checked_func(true) == 1
    end

    @testset "@debug_ccall" begin
        # Test that debug_ccall works and captures output
        c = IOCapture.capture() do
            @debug_ccall time()::Cint
        end

        @test c.value isa Cint
        @test occursin("time()", c.output)
        @test occursin("=", c.output)
    end
end
