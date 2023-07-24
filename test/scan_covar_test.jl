# Tests of single trait scan function scan() interface with adding covariates:

## Consider the 7919-th trait
## Take measurements of the first three traits as pseudo-covariates:
# pseudo_covar = pheno[:, 1:3];


## For short wait-time reason, test by comparing single trait with covariates scan with the 
## multiple trait null_grid algorithm function bulkscan_null_grid()
test_scan_covar = scan(pheno_y, geno, pseudo_covars, kinship);
test_scan_covar_svd = scan(pheno_y, geno, pseudo_covars, kinship; decomp_scheme = "svd");
test_grid_covar = bulkscan_null_grid(pheno[:, 4:end], geno, pseudo_covars, kinship, 
                                     vcat(collect(0.0:0.05:0.95), test_scan_covar.h2_null)).L;
test_grid_covar_svd = bulkscan_null_grid(pheno[:, 4:end], geno, pseudo_covars, kinship, 
                                     vcat(collect(0.0:0.05:0.95), test_scan_covar.h2_null);
                                     decomp_scheme = "svd").L;                                    

tol = 1e-8;    

# Test that the error case will be successfully detected:
try 
    scan(pheno_y, geno, kinship; addIntercept = false)
catch e
    @test e.msg == "Intercept has to be added when no other covariate is given."
end

println("Scan with covariates functions test: ", 
@test mean(abs.(test_scan_covar.lod .- test_grid_covar[:, (pheno_id-3)])) <= tol
)

println("Scan with covariates functions test (SVD): ", 
@test mean(abs.(test_scan_covar.lod .- test_scan_covar_svd.lod)) <= tol
)

println("Scan with covariates functions test (SVD2): ", 
@test mean(abs.(test_scan_covar.lod .- test_grid_covar_svd[:, (pheno_id-3)])) <= tol
)

