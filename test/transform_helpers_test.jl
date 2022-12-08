# Test basic helper functions:

## Note: make sure pwd() is "BulkLMM.jl/test"

pheno_y = reshape(pheno[:, 1126], :, 1);

##########################################################################################################
## TEST: transform_rotation()
##########################################################################################################

## check dimensions:
test_rotation1 = quote
    try
        BulkLMM.transform_rotation(pheno_y[2:end, :], geno, kinship);
    catch e
        @test e.msg == "Dimension mismatch.";
    end
end;

## check add intercept
test_rotation2 = quote
    X_intercept = BulkLMM.transform_rotation(pheno_y, geno, kinship)[2];
    X_nointercept = BulkLMM.transform_rotation(pheno_y, geno, kinship; addIntercept = false)[2];
    @test size(X_intercept, 2) == p+1;
    @test size(X_nointercept, 2) == p;
end;

## check if the kinship (covariance) matrix is semi-positive definite
test_rotation3 = quote
    M_notSPD = diagm(ones(n));
    M_notSPD[22, 22] = -1.0;
    try 
        BulkLMM.transform_rotation(pheno_y, geno, M_notSPD);
    catch e
        @test e.msg == "Negative eigenvalues exist. The kinship matrix supplied may not be SPD.";
    end
end;

## check if final results are as desired:
test_rotation4 = quote
    results = BulkLMM.transform_rotation(pheno_y, geno, kinship; addIntercept = false);

    EF = eigen(kinship);

    Ut = EF.vectors';

    @test sumSqDiff(Ut*pheno_y, results[1]) < 1e-10;
    @test sumSqDiff(Ut*geno, results[2]) < 1e-10;
    @test mean(EF.values .== results[3]) == 1;

end;

##########################################################################################################
## TEST: transform_reweight()
##########################################################################################################

(y0, X0, lambda0) = BulkLMM.transform_rotation(pheno_y, geno, kinship);
X0_inter = reshape(X0[:, 1], :, 1);
X0_covar = X0[:, 2:end];

## Note: nested functions `fitlmm`, `makeweights` `rowMultiply!`, `resid` have been tested in other testing files

## check if the function modifies the inputs:
test_reweight1 = quote
    c_y0 = copy(y0);
    c_X0 = copy(X0);

    BulkLMM.transform_reweight(y0, X0, lambda0);

    @test sumSqDiff(c_y0, y0) < 1e-10;
    @test sumSqDiff(c_X0, X0) < 1e-10;
end;

## check if the dimensions of the output are as desired (especially for the covariate matrix)
test_reweight2 = quote
    results = BulkLMM.transform_reweight(y0, X0, lambda0);

    @test size(results[1], 1) == n;
    # check if the output covariate matrix excludes the intercept
    @test mean(size(results[2]) .== [n, p]) == 1;
end;

## check results, by comparing with equivalent wls results
test_reweight3 = quote
    results = BulkLMM.transform_reweight(y0, X0, lambda0);
    prior = zeros(2);

    vc = BulkLMM.fitlmm(y0, X0_inter, lambda0, prior);
    sqrtw = sqrt.(BulkLMM.makeweights(vc.h2, lambda0));

    wlsEsts = wls(X0_covar, X0_inter, sqrtw, prior);

    res_wls = y0 - X0_inter*vc.b;
    res_wls = mapslices(x -> x .* sqrt.(sqrtw), res_wls; dims = 1);

    covar_wls = X0_covar .- X0_inter*wlsEsts.b;
    covar_wls = mapslices(x -> x .* sqrt.(sqrtw), covar_wls; dims = 1);

    @test sumSqDiff(res_wls, results[1]) < 1e-10;
    @test sumSqDiff(covar_wls, results[2]) < 1e-10;

end;

##########################################################################################################
## TEST: transform_permute()
##########################################################################################################

(r0, X00, sigma2_e) = BulkLMM.transform_reweight(y0, X0, lambda0);

## Note: nested function `shuffleVector` has been tested in another testing file

## check is the output has the desired size
test_permute1 = quote
    result_noOri = BulkLMM.transform_permute(r0; nperms = 1000, original = false);
    result = BulkLMM.transform_permute(r0; nperms = 1000);

    @test mean(size(result_noOri) .== [n, 1000]) == 1;
    @test mean(size(result) .== [n, 1001]) == 1;

end;

## check if the original is kept;
test_permute2 = quote
    result = BulkLMM.transform_permute(r0; nperms = 1000);

    @test mean(r0 .== result[:, 1]) == 1;

end;

##########################################################################################################
## TEST: Run all tests
##########################################################################################################
@testset "testTransformHelpers" begin

    eval(test_rotation1);
    eval(test_rotation2);
    eval(test_rotation3);
    eval(test_rotation4);

    eval(test_reweight1);
    eval(test_reweight2);
    eval(test_reweight3);

    eval(test_permute1);
    eval(test_permute2);

end;