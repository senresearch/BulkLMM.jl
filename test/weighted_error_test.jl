# Tests of the modeling weighted errors feature:
## will mainly test two cases:
## - when W is an identical matrix (same errors)
## - when W is not an identical (not identical errors for every response)

using Random

n = size(pheno, 1);
weights_identical = ones(n);
weights_at_random = rand(Uniform(0, 1), n); # create weights at random
pseudo_covars = pheno[:, 1:3];
tol = 1e-3;

## Pre-process weighted data for comparison:
pheno_weighted = BulkLMM.rowMultiply(pheno, weights_at_random);
pheno_y_weighted = reshape(pheno_weighted[:, pheno_id], :, 1);
geno_weighted = BulkLMM.rowMultiply(geno, weights_at_random);
intercept_weighted = BulkLMM.rowMultiply(ones(n, 1), weights_at_random);
covars_weighted = BulkLMM.rowMultiply(pseudo_covars, weights_at_random);

kinship_weighted = BulkLMM.rowMultiply(kinship, weights_at_random);
kinship_weighted = BulkLMM.rowMultiply(permutedims(kinship_weighted), weights_at_random);


## scan functions:
results_no_weights = scan(pheno_y, geno, kinship);
results_identical = scan(pheno_y, geno, kinship; weights = weights_identical);
results_no_weights_covar = scan(pheno_y, geno, pseudo_covars, kinship);
results_identical_covar = scan(pheno_y, geno, pseudo_covars, kinship; weights = weights_identical);

results_w_weights = scan(pheno_y_weighted, geno_weighted, intercept_weighted, kinship_weighted; addIntercept = false);
results_random_weights = scan(pheno_y, geno, kinship; weights = weights_at_random);
results_w_weights_covar = scan(pheno_y_weighted, geno_weighted, [intercept_weighted covars_weighted], kinship_weighted; addIntercept = false);
results_random_weights_covar = scan(pheno_y, geno, pseudo_covars, kinship; weights = weights_at_random);

### additional: from permutation testing results...
results_random_weights_perms = scan(pheno_y, geno, kinship; weights = weights_at_random, permutation_test = true, original = true, nperms = 10)[:, 1];
results_random_weights_covar_perms = scan(pheno_y, geno, pseudo_covars, kinship; weights = weights_at_random, permutation_test = true, original = true, nperms = 10)[:, 1];

test_scan_caseI = quote
    @test sum(abs.(results_identical.lod .- results_no_weights.lod)) <= tol
end

test_scan_caseI2 = quote
    @test sum(abs.(results_identical_covar.lod .- results_no_weights_covar.lod)) <= tol
end

test_scan_caseW = quote
    @test sum(abs.(results_random_weights.lod .- results_w_weights.lod)) <= tol
end

test_scan_caseW2 = quote
    @test sum(abs.(results_random_weights_covar.lod .- results_w_weights_covar.lod)) <= tol
end

test_scan_caseW3 = quote
    @test sum(abs.(results_random_weights_perms .- results_w_weights.lod)) <= tol
end

test_scan_caseW4 = quote
    @test sum(abs.(results_random_weights_covar_perms .- results_w_weights_covar.lod)) <= tol
end

@testset "Tests for weighted errors feature" begin
    eval(test_scan_caseI);
    eval(test_scan_caseI2);
    eval(test_scan_caseW);
    eval(test_scan_caseW2);
    eval(test_scan_caseW3);
    eval(test_scan_caseW4);
end

