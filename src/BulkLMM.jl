module BulkLMM

    # packages we need to work
    using DelimitedFiles, DataFrames, CSV, Missings, LinearAlgebra, Statistics, Optim, Random, Distributions, LoopVectorization

    include("util.jl");
    include("kinship.jl");

    include("readData.jl");

    # code for (wls) weighted least squares
    include("wls.jl");
    export wls

    # code for rorateData and flmm
    include("lmm.jl");
    # data type we are exporting
    export LMMEstimates

    include("scan.jl");
    export scan
    export scan_perms
    export scan_perms_lite

    include("bulkscan.jl");
    export scan_lite_multivar

    include("transform_helpers.jl");
    export transform_rotation

end # module
