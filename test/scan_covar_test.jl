# Tests of single trait scan function scan() interface with adding covariates:

## Consider the 705-th trait
pheno_y = reshape(pheno[:, 705], :, 1);
## Take measurements of the first three traits as pseudo-covariates:
pseudo_covar = pheno[:, 1:3];


## For short wait-time reason, test by comparing single trait with covariates scan with the 
## multiple trait null_grid algorithm function bulkscan_null_grid()
test_scan_covar = scan(pheno_y, geno, pseudo_covar, kinship);
test_grid_covar = bulkscan_null_grid(pheno, geno, pseudo_covar, kinship, 
                                     vcat(collect(0.0:0.05:0.95), test_scan_covar.h2_null)).L;

tol = 1e-8;    

# Test that the error case will be successfully detected:
try 
    scan(pheno_y, geno, kinship; addIntercept = false)
catch e
    @test e.msg == "Intercept has to be added when no other covariate is given."
end

@test mean(abs.(test_scan_covar.lod .- test_grid_covar[:, 705])) <= tol

