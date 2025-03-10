using GPUToolbox
using Documenter
import Documenter.Remotes: GitHub

DocMeta.setdocmeta!(GPUToolbox, :DocTestSetup, :(using GPUToolbox); recursive = true)

makedocs(;
    modules = [GPUToolbox],
    authors = "",
    repo = GitHub("JuliaGPU", "GPUToolbox.jl"),
    sitename = "GPUToolbox.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://juliagpu.com/GPUToolbox.jl",
        mathengine = MathJax3(),
    ),
    pages = [
        "Home" => "index.md",
    ],
    doctest = true,
    linkcheck = true,
)

deploydocs(;
    repo = "github.com/JuliaGPU/GPUToolbox.jl.git",
    devbranch = "main",
    push_preview = true,
)
