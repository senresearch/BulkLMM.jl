# This test file will test univariate scan functions by running on BXD data and comparing results with R lmmlite package

## Consider the 7919-th trait
pheno_y = reshape(pheno[:, 7919], :, 1);

## Run BulkLMM code to get results:
reml_results = BulkLMM.scan(pheno_y, geno, kinship; reml = true); # by default uses scan_null;
ml_results = BulkLMM.scan(pheno_y, geno, kinship; reml = false);

lods_BulkLMM_reml = reshape(reml_results.lod, :, 1);
lods_BulkLMM_ml = reshape(ml_results.lod, :, 1);


## Compare with lmmlite results:
## Read in lmmlite results:
file_reml_path = string(@__DIR__, "/run-lmmlite_R/output/result.lmmlite_REML.csv") 
file_ml_path = string(@__DIR__, "/run-lmmlite_R/output/result.lmmlite_ML.csv")
reml_results_lmmlite = CSV.read(file_reml_path, DataFrame);
ml_results_lmmlite = CSV.read(file_ml_path, DataFrame);

lods_lmmlite_reml = parse.(Float64, reshape(reml_results_lmmlite[2:end, 5], :, 1));
lods_lmmlite_ml = parse.(Float64, reshape(ml_results_lmmlite[2:end, 5], :, 1));

## Testings:
println("Scan functions test: ")
@testset "lmmlite_results_tests" begin
    tol = 1e-9;
    @test sumSqDiff(lods_lmmlite_reml, lods_BulkLMM_reml) <= sqrt(tol);
    @test sumSqDiff(lods_lmmlite_ml, lods_BulkLMM_ml) <= sqrt(tol);
    @test maxSqDiff(lods_lmmlite_reml, lods_BulkLMM_reml) <= tol;
    @test maxSqDiff(lods_lmmlite_ml, lods_BulkLMM_ml) <= tol;
end
