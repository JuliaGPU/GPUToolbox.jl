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

        # test with visibility=:export
        mod_export = @eval module $(gensym())
            using GPUToolbox
            @enum MY_ENUM MY_ENUM_A MY_ENUM_B
            @enum_without_prefix visibility=:export MY_ENUM MY_ENUM_
        end
        @test mod_export.A == mod_export.MY_ENUM_A
        @test mod_export.B == mod_export.MY_ENUM_B
        exported = names(mod_export)
        @test :A in exported
        @test :B in exported

        # test with visibility=:public
        @static if VERSION >= v"1.11"
            mod_public = @eval module $(gensym())
                using GPUToolbox
                @enum MY_ENUM MY_ENUM_P MY_ENUM_Q
                @enum_without_prefix visibility=:public MY_ENUM MY_ENUM_
            end
            @test mod_public.P == mod_public.MY_ENUM_P
            @test mod_public.Q == mod_public.MY_ENUM_Q
            # public but not exported
            @test :P in names(mod_public)
            @test !Base.isexported(mod_public, :P)
        end
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

        nothing_count = Ref(0)
        lazy_nothing = LazyInitialized{Union{Nothing,Int}}()
        @test get!(lazy_nothing) do
            nothing_count[] += 1
            nothing
        end === nothing
        @test nothing_count[] == 1
        @test get!(lazy_nothing) do
            error("Should not be called")
        end === nothing
        @test nothing_count[] == 1

    end

    @testset "LazyInitialized concurrency" begin
        init_count = Threads.Atomic{Int}(0)
        lazy = LazyInitialized{NTuple{4,Int}}()

        function build_tuple()
            id = Threads.atomic_add!(init_count, 1) + 1
            yield()
            return ntuple(Returns(id), 4)
        end

        tasks = [Threads.@spawn get!(build_tuple, lazy) for _ in 1:128]
        results = fetch.(tasks)
        @test init_count[] == 1
        @test all(==((1, 1, 1, 1)), results)
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

        nothing_call_count = Ref(0)
        function test_no_key_nothing_memo()
            @memoize begin
                nothing_call_count[] += 1
                nothing
            end::Union{Nothing,Int}
        end

        @test test_no_key_nothing_memo() === nothing
        @test nothing_call_count[] == 1
        @test test_no_key_nothing_memo() === nothing
        @test nothing_call_count[] == 1

        # Test memoization with key (dictionary)
        dict_call_count = Ref(0)
        function test_dict_memo(x)
            @memoize key=x::Int begin
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

        dict_nothing_call_count = Ref(0)
        function test_dict_nothing_memo(x)
            @memoize key=x::Int begin
                dict_nothing_call_count[] += 1
                nothing
            end::Union{Nothing,Int}
        end

        @test test_dict_nothing_memo(5) === nothing
        @test dict_nothing_call_count[] == 1
        @test test_dict_nothing_memo(5) === nothing
        @test dict_nothing_call_count[] == 1
        @test test_dict_nothing_memo(3) === nothing
        @test dict_nothing_call_count[] == 2

        # Test memoization with index (auto-growing array)
        index_call_count = Ref(0)
        function test_index_memo(x)
            @memoize index=x begin
                index_call_count[] += 1
                x * 3
            end::Int
        end

        @test test_index_memo(1) == 3
        @test index_call_count[] == 1
        @test test_index_memo(1) == 3
        @test index_call_count[] == 1  # Should not increment
        @test test_index_memo(2) == 6
        @test index_call_count[] == 2  # Should increment for new index

        index_nothing_call_count = Ref(0)
        function test_index_nothing_memo(x)
            @memoize index=x begin
                index_nothing_call_count[] += 1
                nothing
            end::Union{Nothing,Int}
        end

        @test test_index_nothing_memo(1) === nothing
        @test index_nothing_call_count[] == 1
        @test test_index_nothing_memo(1) === nothing
        @test index_nothing_call_count[] == 1
        @test test_index_nothing_memo(2) === nothing
        @test index_nothing_call_count[] == 2

        # `index` may come from APIs that return smaller integer types.
        int32_index_call_count = Ref(0)
        function test_index_memo_int32_index(x)
            @memoize index=x begin
                int32_index_call_count[] += 1
                Int(x) * 4
            end::Int
        end

        @test test_index_memo_int32_index(Int32(1)) == 4
        @test int32_index_call_count[] == 1
        @test test_index_memo_int32_index(Int32(1)) == 4
        @test int32_index_call_count[] == 1

        grow_counts = zeros(Int, 12)
        function test_index_grow(x)
            @memoize index=x begin
                grow_counts[x] += 1
                x + 100
            end::Int
        end

        @test [test_index_grow(i) for i in 1:12] == collect(101:112)
        @test [test_index_grow(i) for i in 1:12] == collect(101:112)
        @test all(==(1), grow_counts)

        bad_index_call_count = Ref(0)
        function test_bad_index_memo(x)
            @memoize index=x begin
                bad_index_call_count[] += 1
                x
            end::Int
        end

        @test_throws BoundsError test_bad_index_memo(0)
        @test_throws BoundsError test_bad_index_memo(-1)
        @test bad_index_call_count[] == 0

        @test_throws ArgumentError macroexpand(
            @__MODULE__, :(@memoize x::Int begin x end::Int)
        )
        @test_throws ArgumentError macroexpand(
            @__MODULE__, :(@memoize maxlen=2 begin 1 end::Int)
        )
    end

    @testset "@memoize concurrency" begin
        no_key_count = Threads.Atomic{Int}(0)
        function memo_no_key_stress()
            @memoize begin
                Threads.atomic_add!(no_key_count, 1)
                yield()
                1234
            end::Int
        end

        tasks = [Threads.@spawn memo_no_key_stress() for _ in 1:128]
        @test all(==(1234), fetch.(tasks))
        @test no_key_count[] == 1

        dict_count = Threads.Atomic{Int}(0)
        function memo_dict_stress(x)
            @memoize key=x::Int begin
                Threads.atomic_add!(dict_count, 1)
                yield()
                x * 10
            end::Int
        end

        keys = [mod1(i, 16) for i in 1:256]
        tasks = [Threads.@spawn memo_dict_stress(key) for key in keys]
        @test fetch.(tasks) == keys .* 10
        @test dict_count[] == 16

        index_count = Threads.Atomic{Int}(0)
        function memo_index_stress(x)
            @memoize index=x begin
                Threads.atomic_add!(index_count, 1)
                yield()
                x * 20
            end::Int
        end

        tasks = [Threads.@spawn memo_index_stress(key) for key in keys]
        @test fetch.(tasks) == keys .* 20
        @test index_count[] == 16

        index_count_after_warmup = index_count[]
        tasks = [Threads.@spawn memo_index_stress(key) for key in keys]
        @test fetch.(tasks) == keys .* 20
        @test index_count[] == index_count_after_warmup

        grow_race_count = Threads.Atomic{Int}(0)
        function memo_index_grow_race(x)
            @memoize index=x begin
                Threads.atomic_add!(grow_race_count, 1)
                yield()
                x * 30
            end::Int
        end

        @test memo_index_grow_race(1) == 30
        @test grow_race_count[] == 1
        tasks = [Threads.@spawn memo_index_grow_race(64) for _ in 1:128]
        @test all(==(64 * 30), fetch.(tasks))
        @test grow_race_count[] == 2
    end

    @testset "Session-safe caches" begin
        mktempdir() do dir
            probe_dir = joinpath(dir, "MemoSessionProbe")
            src_dir = joinpath(probe_dir, "src")
            mkpath(src_dir)

            write(joinpath(probe_dir, "Project.toml"), """
            name = "MemoSessionProbe"
            uuid = "5c47f73b-ac9c-4df8-9db3-37f1345bcb51"
            version = "0.1.0"

            [deps]
            Pkg = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
            """)

            write(joinpath(src_dir, "MemoSessionProbe.jl"), """
            module MemoSessionProbe

            using GPUToolbox

            const PRECOMPILE = Ref((0, 0, 0, 0))
            const LAZY = LazyInitialized{Int}()

            lazy_pid() = get!(() -> Int(getpid()), LAZY)

            function memo_no_key_pid()
                @memoize begin
                    Int(getpid())
                end::Int
            end

            function memo_index_pid(key)
                @memoize index=key begin
                    Int(getpid())
                end::Int
            end

            function memo_dict_pid(key)
                @memoize key=key::Int begin
                    Int(getpid())
                end::Int
            end

            precompile_values() =
                (lazy_pid(), memo_no_key_pid(), memo_index_pid(1), memo_dict_pid(1))

            runtime_values() =
                (lazy_pid(), memo_no_key_pid(), memo_index_pid(1), memo_index_pid(4), memo_dict_pid(1))

            if ccall(:jl_generating_output, Cint, ()) != 0
                PRECOMPILE[] = precompile_values()
            end

            end
            """)

            julia = joinpath(Sys.BINDIR, Base.julia_exename())
            toolbox_path = pkgdir(GPUToolbox)
            setup = """
            using Pkg
            Pkg.develop(PackageSpec(path=$(repr(toolbox_path))))
            Pkg.instantiate()
            using MemoSessionProbe
            """
            run(Cmd([julia, "--startup-file=no", "--project=$probe_dir", "-e", setup]))

            check = """
            using MemoSessionProbe
            pre = MemoSessionProbe.PRECOMPILE[]
            vals = MemoSessionProbe.runtime_values()
            println("RESULT ", join((pre..., vals..., Int(getpid())), ","))
            """
            out = read(Cmd([julia, "--startup-file=no", "--project=$probe_dir", "-e", check]), String)
            line = only(filter(startswith("RESULT "), split(out, '\n')))
            nums = parse.(Int, split(chop(line; head=7, tail=0), ","))
            precompile_vals = nums[1:4]
            runtime_vals = nums[5:9]
            runtime_pid = nums[10]

            @test all(!=(0), precompile_vals)
            @test all(==(runtime_pid), runtime_vals)
            @test all(!=(runtime_pid), precompile_vals)
        end
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
            @debug_ccall time(C_NULL::Ptr{Cvoid})::Cint
        end

        @test c.value isa Cint
        @test occursin("time(Ptr{Nothing}", c.output)
        @test occursin("=", c.output)
    end

    @testset "overlay audit" begin
        Overlays = GPUToolbox.Overlays
        # every override in the shared tables replaces a Base method, and they don't overlap
        @test isempty(Overlays.audit(Overlays.float64_overrides))

        @eval module AuditTest
            Base.Experimental.@MethodTable(high)
            Base.Experimental.@MethodTable(low)
            struct Radians end
            Base.Experimental.@overlay high Base.sin(x::Float32) = x
            Base.Experimental.@overlay high Base.cos(x::Float32) = x
            Base.Experimental.@overlay low Base.sin(x::Float32) = x
            Base.Experimental.@overlay low Base.cos(x::Real) = x
            Base.Experimental.@overlay low Base.Checked.throw_overflowerr_negation(op, x, y) = x
            Base.Experimental.@overlay low Base.sin(x::Radians) = x
        end
        issues = Overlays.audit(AuditTest.high, AuditTest.low)
        found(issue, sig) = any(i -> i.issue === issue && i.method.sig == sig, issues)
        @test found(:shadowed, Tuple{typeof(sin), Float32})
        @test found(:partly_shadowed, Tuple{typeof(cos), Real})
        @test found(:dead, Tuple{typeof(Base.Checked.throw_overflowerr_negation), Any, Any, Any})
        @test length(issues) == 3
    end

    # device overrides can be called on the CPU by invoking their method directly, but that
    # requires Julia 1.12. Calls within the overrides then use Base's (host) methods.
    VERSION >= v"1.12" && @testset "device overrides" begin
        function override(f, args...)
            sig = Tuple{typeof(f), map(typeof, args)...}
            matches = Base._methods_by_ftype(sig, GPUToolbox.Overlays.float64_overrides, -1,
                                             Base.get_world_counter())
            invoke(f, only(matches).method, args...)
        end

        # error in ulps, compared to a more precise result
        ulps(x::T, ref) where {T} = isequal(x, T(ref)) ? zero(T) : abs(x - T(ref)) / eps(T(ref))

        @testset "comparisons" begin
            xs = Float32[0, -0.0, 1, -1, 2^24, 2^24 + 2, 2^31, -2^31, 2^32, NaN, Inf, -Inf, 0.5]
            ys = Any[Int32(0), Int32(1), Int32(-1), Int32(2^24 + 1), typemax(Int32),
                     typemin(Int32), UInt32(2^24 + 1), typemax(UInt32)]
            for op in (==, <, <=), x in xs, y in ys
                @test override(op, x, y) == op(x, y)
                @test override(op, y, x) == op(y, x)
            end
            for op in (==, <, <=), x in Float16[0, 1, 2048, 65504, Inf, NaN],
                y in (Int32(2049), typemax(Int64), UInt64(1))
                @test override(op, x, y) == op(x, y)
                @test override(op, y, x) == op(y, x)
            end
        end

        # matches Base for finite quotients with eps(x/y) <= 1, and for special values
        v"1.12-" <= VERSION < v"1.14-" && @testset "div" begin
            xs = Float32[1, 6, 3, -1, 1f7, 7, -7, 0.5, -0.5, 0, -0.0, Inf, -Inf, NaN]
            ys = Float32[0.1, 0.1, 0.3, 0.1, 3, 2, -2, 1, 1, 1, -1, 1, 0, Inf, -Inf, NaN]
            for x in [xs; 5f0; -5f0; 2.5f0; -1.5f0; floatmax(Float32); -floatmax(Float32)], y in [ys; 2f0], r in (RoundToZero, RoundDown, RoundUp,
                RoundNearest, RoundFromZero, RoundNearestTiesAway, RoundNearestTiesUp)
                @test isequal(override(div, x, y, r), div(x, y, r))
            end
        end

        @testset "integer powers" begin
            for x in (0.7f0, -1.3f0, 2f0, -0f0, 1.000001f0), n in (-7, -2, -1, 0, 1, 2, 3, 5, 12, 16777217)
                @test ulps(override(^, x, n), big(x)^n) <= 2
            end
            @test override(^, -1f0, 16777217) == -1
            @test override(^, -1f0, Int32(16777216)) == 1
            @test override(^, -0f0, -16777217) == -Inf
            @test override(^, 0.5f0, typemax(Int64)) == 0
        end

        @testset "hypot" begin
            @test override(Base.Math._hypot, 3f0, 4f0) == 5
            @test override(Base.Math._hypot, 3f30, 4f30) == 5f30
            @test override(Base.Math._hypot, 3f-30, 4f-30) ≈ 5f-30
        end

        # the kernels used by `sind` and friends, on the converted argument
        @testset "degree trigonometry" begin
            for x in (0.001f0, 1f0, 30f0, 44.99f0, -45f0)
                y = override(Base.Math.deg2rad_ext, x)
                @test ulps(override(Base.Math.sin_kernel, y), sind(big(x))) <= 1
                @test ulps(override(Base.Math.cos_kernel, y), cosd(big(x))) <= 1
            end
        end

        @test override(sincospi, 0.25f0) == (sinpi(0.25f0), cospi(0.25f0))

        @testset "complex division" begin
            zs = ComplexF32[1 + 2im, -3f20 + 1f-20im, 2f38 + 2f38im, 1f-30 - 3f-30im, 1f-20 + 1im,
                            0x1p64, complex(0x1p-64, 0x1p-64), 1f-10]
            for a in zs, b in zs
                @test override(/, a, b) ≈ ComplexF32(ComplexF64(a) / ComplexF64(b))
            end
            for b in zs
                @test override(inv, b) ≈ ComplexF32(inv(ComplexF64(b)))
            end
            @test override(/, 2f38 + 2f38im, 2f38 + 2f38im) == 1
            # normal components shouldn't be lost when subnormals are flushed to zero
            ftz = get_zero_subnormals()
            if set_zero_subnormals(true)
                try
                    @test override(/, 1f20 + 1f-20im, 1f-10 + 0im) ≈ 1f30 + 1f-10im
                finally
                    set_zero_subnormals(ftz)
                end
            end
            @test isequal(override(inv, complex(Inf32, 1f0)), inv(complex(Inf32, 1f0)))
            @test isequal(override(inv, complex(1f0, -0f0)), inv(complex(1f0, -0f0)))
            @test isequal(override(/, 1f0 + 1f0im, complex(1f0, -Inf32)), (1f0 + 1f0im) / complex(1f0, -Inf32))
            @test all(isnan, reim(override(inv, 0f0im)))
            @test all(isnan, reim(override(/, 1f0 + 0im, 0f0im)))

            # zero components have the same signs as Base's, which matters for branch cuts
            vals = Float32[0, -0.0, 1, -1, 2, -2, 0.5, 3, -3]
            exact(x, y) = all(((u, v),) -> iszero(v) || isinteger(v) ? isequal(u, v) : u ≈ v,
                              zip(reim(x), reim(y)))
            @test all(exact(override(/, complex(a, b), complex(c, d)), complex(a, b) / complex(c, d))
                      for a in vals, b in vals, c in vals, d in vals if !iszero(c) || !iszero(d))
            @test all(exact(override(inv, complex(c, d)), inv(complex(c, d)))
                      for c in vals, d in vals if !iszero(c) || !iszero(d))
            @test isequal(override(/, 1f0 + 0im, complex(-1f0, -0f0)), complex(-1f0, 0f0))

            # a normal component shouldn't be lost when intermediate ratios underflow
            q = override(/, complex(1f38, 0f0), complex(1f20, 1f-30))
            @test real(q) ≈ 1f18 && imag(q) ≈ -1f-32

            # normwise accuracy over the whole range
            rnd() = Float32(randn() * 10.0^rand(-37:37))
            for _ in 1:10_000
                a, b = complex(rnd(), rnd()), complex(rnd(), rnd())
                ref = ComplexF64(a) / ComplexF64(b)
                floatmin(Float32) <= abs(ref) <= floatmax(Float32) / 4 || continue
                @test abs(ComplexF64(override(/, a, b)) - ref) <= 2eps(Float32) * abs(ref)
            end
        end
    end
end
