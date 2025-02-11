using Test
using GPUToolbox

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

    # TODO: @debug_ccall and @gcsafe_ccall tests
end
