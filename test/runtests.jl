using BulkLMM
using Test

@testset "BulkLMM" begin
    include("testHelpers.jl");
    include("generate_test_bxdData.jl");

    include("util-test.jl");
    include("wls-basic-test.jl");
    include("wls-results-test.jl");
    include("lmm-test.jl");
    include("transform-helpers-test.jl");
    include("scan-test-lmmlite.jl");
end;