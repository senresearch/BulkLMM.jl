module BulkLMM

# packages we need to work
using DelimitedFiles
using DataFrames
using CSV
using Missings
using LinearAlgebra
using Statistics
using Optim
using Random
using Distributions
using LoopVectorization


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

include("util.jl");

include("transform_helpers.jl");
export transform_rotation

include("readData.jl");

end # module
