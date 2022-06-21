using BulkLMM
using Documenter

DocMeta.setdocmeta!(BulkLMM, :DocTestSetup, :(using BulkLMM); recursive=true)

makedocs(;
    modules=[BulkLMM],
    authors="Saunak Sen <sen@uthsc.edu> and contributors",
    repo="https://github.com/sens/BulkLMM.jl/blob/{commit}{path}#{line}",
    sitename="BulkLMM.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
