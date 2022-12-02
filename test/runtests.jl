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
    include("util-test.jl");
    include("wls-basic-test.jl");
    include("wls-results-test.jl");

    include("generate_test_bxdData.jl"); # read in real BXD data for the remaining tests:
    include("lmm-test.jl");
    include("transform-helpers-test.jl");
    include("scan-test-lmmlite.jl");
end;