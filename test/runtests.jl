using BulkLMM
using CSV
using Test
using Statistics
using Random
using Distributions
using LinearAlgebra
using DataFrames

include("testHelpers.jl");

@testset "BulkLMM" begin
    include("util_test.jl");
    include("wls_basic_test.jl");
    include("wls_results_test.jl");

    include("generate_test_bxdData.jl"); # read in real BXD data for the remaining tests:
    include("lmm_test.jl");
    include("transform_helpers_test.jl");
    include("scan_test_lmmlite.jl");
#     include("bulkscan_test.jl");
end;
