using GPUToolbox
using Documenter
import Documenter.Remotes: GitHub
using DocumenterCodeBlocks: CodeBlocks

DocMeta.setdocmeta!(GPUToolbox, :DocTestSetup, :(using GPUToolbox); recursive = true)

makedocs(;
    modules = [GPUToolbox],
    authors = "",
    repo = GitHub("JuliaGPU", "GPUToolbox.jl"),
    sitename = "GPUToolbox.jl",
    plugins = [
        CodeBlocks(; line_counter = :named),
    ],
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://juliagpu.com/GPUToolbox.jl",
        mathengine = MathJax3(),
    ),
    pages = [
        "Home" => "index.md",
        "Overlays" => "overlays.md",
    ],
    doctest = true,
    linkcheck = true,
)

deploydocs(;
    repo = "github.com/JuliaGPU/GPUToolbox.jl.git",
    devbranch = "main",
    push_preview = true,
)
