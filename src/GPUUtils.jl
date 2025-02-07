module GPUUtils

include("simpleversion.jl") # exports SimpleVersion, @sv_str
include("ccalls.jl") # exports @checked, @debug_ccall, @gcsafe_ccall

end # module GPUUtils
