# Test fitlmm functions:

## Note: make sure pwd() is "BulkLMM.jl/test"

pheno_y = reshape(pheno[:, 1126], :, 1);

##########################################################################################################
## TEST: makeweights()
##########################################################################################################

## check for edge cases:
test_makeweights1 = quote
    try 
        BulkLMM.makeweights(1.0, [0.0]);
    catch e
        @test e.msg == "Heritability of 1 is not allowed.";
    end
end;

test_makeweights2 = quote
    try 
        BulkLMM.makeweights(2.0, [1.0]);
    catch e
        @test e.msg == "Non-positive environmental variance is not allowed; check input values.";
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
