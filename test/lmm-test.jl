# Test fitlmm functions:

## Note: make sure pwd() is "BulkLMM.jl/test"

## Loading required packages:
using Test
using BenchmarkTools

## load the BXD data for testing:
include("BXDdata_for_test.jl");

## load the helper functions to be tested
include("../src/lmm.jl");
include("../src/wls.jl");

pheno_y = reshape(pheno[:, 1126], :, 1);

##########################################################################################################
## TEST: makeweights()
##########################################################################################################

## check for edge cases:
test_makeweights1 = quote
    try 
        makeweights(1.0, [0.0]);
    catch e
        @test e.msg == "Exists heritability of 1 which is not allowed for modeling.";
    end
end;

test_makeweights2 = quote
    try 
        makeweights(2.0, [1.0]);
    catch e
        @test e.msg == "Resulting non-positive environmental variance which is not reasonable; check input values.";
    end
end;

## check results:


##########################################################################################################
## TEST: runtests
##########################################################################################################

@testset "Test lmm.jl" begin
    eval(test_makeweights1);
    eval(test_makeweights2);
end;
