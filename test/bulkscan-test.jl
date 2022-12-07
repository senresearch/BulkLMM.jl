# Test basic helper functions:

## Note: make sure pwd() is "BulkLMM.jl/test"

##########################################################################################################
## TEST:  r2lod()
##########################################################################################################

function lod2r(lod::Float64, n::Int64)
    rsq = 1-10^(-2/n*lod);
    r = sqrt(rsq);
    return r;
end

test_r2lod = quote
    r = lod2r(3.0, 79);
    tol = 1e-7;
    @test abs(BulkLMM.r2lod(r, 79)-3.0) <= tol;
end

##########################################################################################################
## TEST: computeR_LMM
##########################################################################################################

test_computeR_LMM1 = quote
    rng = MersenneTwister(1234);
    x = randn(rng, 100);
    ϵ = rand(Normal(0.0, 1.0));
    y = 3x.+ϵ;

    x = reshape(x, :, 1);
    y = reshape(y, :, 1);
    BulkLMM.colStandardize!(x);
    BulkLMM.colStandardize!(y);

    intercept = reshape(ones(100).*1.0, :, 1);

    @test BulkLMM.computeR_LMM(y, x, intercept)[1, 1] - cor(y, x)[1, 1] <= 1e-7;
end;

test_computeR_LMM2 = quote

    rng = MersenneTwister(1234);
    X = randn(100, 100);
    Y = randn(100, 100);
    BulkLMM.colStandardize!(x);
    BulkLMM.colStandardize!(y);

    intercept = reshape(ones(100).*1.0, :, 1);

    R = BulkLMM.computeR_LMM(Y, X, intercept);

    @test sumSqDiff(R, cor(X, Y)) <= 1e-8;
end;

##########################################################################################################
## TEST: scan_lite_multivar
##########################################################################################################

test_scan_multivar = quote
    stand_pheno = BulkLMM.colStandardize(pheno[:, 705:1112]);
    stand_geno = BulkLMM.colStandardize(geno);

    test_multivar = BulkLMM.scan_lite_multivar(stand_pheno, stand_geno, kinship, 4;
                                               prior_variance = 1.0, 
                                               prior_sample_size = 0.1);

    y_705 = reshape(pheno[:, 705], :, 1);
    test_null_705 = BulkLMM.scan(y_705, geno, kinship; 
                            prior_variance = var(y_705), prior_sample_size = 0.1).lod;

    y_1112 = reshape(pheno[:, 1112], :, 1);
    test_null_1112 = BulkLMM.scan(y_1112, geno, kinship; 
                            prior_variance = var(y_1112), prior_sample_size = 0.1).lod;

    @test sum((test_null_705 .- test_multivar[:, 1]).^2) <= 1e-7;
    @test sum((test_null_1112 .- test_multivar[:, end]).^2) <= 1e-7;

end;





##########################################################################################################
## TEST: run all tests
##########################################################################################################
@testset "Multiple Trait Scan Tests" begin

    eval(test_r2lod);
    eval(test_computeR_LMM1);
    eval(test_computeR_LMM2);
    eval(test_scan_multivar);

end;