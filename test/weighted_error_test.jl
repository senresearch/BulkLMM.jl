# Tests of the modeling weighted errors feature:
## will mainly test two cases:
## - when W is an identical matrix (same errors)
## - when W is not an identical (not identical errors for every response)

using Random

n = size(pheno, 1);
weights_identical = ones(n);
weights_at_random = rand(Uniform(0, 1), n); # create weights at random
pseudo_covars = pheno[:, 1:3];
prior = [1.0, 0.1];
tol = 1e-3;

## Pre-process weighted data for comparison:
pheno_weighted = BulkLMM.rowMultiply(pheno, weights_at_random);
# pheno_weighted = colStandardize(pheno_weighted);
pheno_y_weighted = reshape(pheno_weighted[:, pheno_id], :, 1);
geno_weighted = BulkLMM.rowMultiply(geno, weights_at_random);
intercept_weighted = BulkLMM.rowMultiply(ones(n, 1), weights_at_random);
covars_weighted = BulkLMM.rowMultiply(pseudo_covars, weights_at_random);

kinship_weighted = BulkLMM.rowMultiply(kinship, weights_at_random);
kinship_weighted = BulkLMM.rowMultiply(permutedims(kinship_weighted), weights_at_random);


## scan functions:
results_no_weights = scan(pheno_y, geno, kinship; optim_interval = 10, prior_variance = prior[1], prior_sample_size = prior[2]);
results_identical = scan(pheno_y, geno, kinship; weights = weights_identical, optim_interval = 10, prior_variance = prior[1], prior_sample_size = prior[2]);
results_no_weights_covar = scan(pheno_y, geno, pseudo_covars, kinship; optim_interval = 10, prior_variance = prior[1], prior_sample_size = prior[2]);
results_identical_covar = scan(pheno_y, geno, pseudo_covars, kinship; weights = weights_identical, optim_interval = 10, prior_variance = prior[1], prior_sample_size = prior[2]);

results_w_weights = scan(pheno_y_weighted, geno_weighted, intercept_weighted, kinship_weighted; addIntercept = false, optim_interval = 10, prior_variance = prior[1], prior_sample_size = prior[2]);
results_random_weights = scan(pheno_y, geno, kinship; weights = weights_at_random, optim_interval = 10, prior_variance = prior[1], prior_sample_size = prior[2]);
results_w_weights_covar = scan(pheno_y_weighted, geno_weighted, [intercept_weighted covars_weighted], kinship_weighted; addIntercept = false, optim_interval = 10, prior_variance = prior[1], prior_sample_size = prior[2]);
results_random_weights_covar = scan(pheno_y, geno, pseudo_covars, kinship; weights = weights_at_random, optim_interval = 10, prior_variance = prior[1], prior_sample_size = prior[2]);

### additional: from permutation testing results...
results_random_weights_perms = scan(pheno_y, geno, kinship; weights = weights_at_random, permutation_test = true, original = true, nperms = 10, optim_interval = 10, prior_variance = prior[1], prior_sample_size = prior[2])[:, 1];
results_random_weights_covar_perms = scan(pheno_y, geno, pseudo_covars, kinship; weights = weights_at_random, permutation_test = true, original = true, nperms = 10, optim_interval = 10, prior_variance = prior[1], prior_sample_size = prior[2])[:, 1];

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


## bulkscan (multiple-trait scans) functions:

### bulkscan_null (exact)
pheno_Y = pheno[:, 7919:7923]; # pick a smaller number of traits for fast testing
pheno_Y_weighted = pheno_weighted[:, 7919:7923];

### Identical weights:
resultsY_no_weights = bulkscan_null(pheno_Y, geno, kinship; optim_interval = 10, prior_variance = prior[1], prior_sample_size = prior[2]);
resultsY_identical = bulkscan_null(pheno_Y, geno, kinship; weights = weights_identical, optim_interval = 10, prior_variance = prior[1], prior_sample_size = prior[2]);
resultsY_no_weights_covar = bulkscan_null(pheno_Y, geno, pseudo_covars, kinship; optim_interval = 10, prior_variance = prior[1], prior_sample_size = prior[2]);
resultsY_identical_covar = bulkscan_null(pheno_Y, geno, pseudo_covars, kinship; weights = weights_identical, optim_interval = 10, prior_variance = prior[1], prior_sample_size = prior[2]);

### Non-identical weights:
resultsY_w_weights = bulkscan_null(pheno_Y_weighted, geno_weighted, intercept_weighted, kinship_weighted; 
                                   addIntercept = false, optim_interval = 10, prior_variance = prior[1], prior_sample_size = prior[2]);
resultsY_random_weights = bulkscan_null(pheno_Y, geno, kinship; 
                                   weights = weights_at_random, optim_interval = 10, prior_variance = prior[1], prior_sample_size = prior[2]);

resultsY_w_weights_covar = bulkscan_null(pheno_Y_weighted, geno_weighted, [intercept_weighted covars_weighted], kinship_weighted; 
                                         addIntercept = false, optim_interval = 10, prior_variance = prior[1], prior_sample_size = prior[2]);
resultsY_random_weights_covar = bulkscan_null(pheno_Y, geno, pseudo_covars, kinship; 
                                         weights = weights_at_random, optim_interval = 10, prior_variance = prior[1], prior_sample_size = prior[2]);


test_bulkscan_caseI = quote
    @test sum(abs.(resultsY_identical.L[:, 1] .- results_identical.lod)) <= tol
    @test sumSqDiff(resultsY_no_weights.L, resultsY_identical.L) <= tol;
    @test sumSqDiff(resultsY_no_weights_covar.L, resultsY_identical_covar.L) <= tol;
end

test_bulkscan_caseW = quote
    @test sum(abs.(resultsY_random_weights.L[:, 1] .- results_random_weights.lod)) <= tol
    @test sumSqDiff(resultsY_w_weights.L, resultsY_random_weights.L) <= tol;
    @test sumSqDiff(resultsY_w_weights_covar.L, resultsY_random_weights_covar.L) <= tol;
end

### bulkscan_null_grid (approximation using grid-search)
grid_h2 = collect(0.00:0.001:0.999);
grid_h2 = vcat(grid_h2, resultsY_random_weights.h2_null_list, resultsY_random_weights_covar.h2_null_list)
looser_tol = 0.1;


resultsY_identical_grid = bulkscan_null_grid(pheno_Y, geno, kinship, grid_h2; 
                                             weights = weights_identical, prior_variance = prior[1], prior_sample_size = prior[2]);
resultsY_identical_covar_grid = bulkscan_null_grid(pheno_Y, geno, pseudo_covars, kinship, grid_h2; 
                                                   weights = weights_identical, prior_variance = prior[1], prior_sample_size = prior[2]);

resultsY_random_weights_grid = bulkscan_null_grid(pheno_Y, geno, kinship, grid_h2; 
                                                  weights = weights_at_random, prior_variance = prior[1], prior_sample_size = prior[2]);
resultsY_random_weights_covar_grid = bulkscan_null_grid(pheno_Y, geno, pseudo_covars, kinship, grid_h2;
                                                        weights = weights_at_random, prior_variance = prior[1], prior_sample_size = prior[2]);

test_bulkscan_grid_caseI = quote
    @test maxSqDiff(resultsY_identical_grid.L, resultsY_identical.L) <= looser_tol;
    @test maxSqDiff(resultsY_identical_covar_grid.L, resultsY_identical_covar.L) <= looser_tol;
end

test_bulkscan_grid_caseW = quote
    @test maxSqDiff(resultsY_random_weights_grid.L, resultsY_random_weights.L) <= looser_tol;
    @test maxSqDiff(resultsY_random_weights_covar_grid.L, resultsY_random_weights_covar.L) <= looser_tol;
end



@testset "Tests for weighted errors feature" begin
    eval(test_scan_caseI);
    eval(test_scan_caseI2);
    eval(test_scan_caseW);
    eval(test_scan_caseW2);
    eval(test_scan_caseW3);
    eval(test_scan_caseW4);
    eval(test_bulkscan_caseI);
    eval(test_bulkscan_caseW);
    eval(test_bulkscan_grid_caseI);
    eval(test_bulkscan_grid_caseW); # bulkscan_null_grid may have some issues...
end

