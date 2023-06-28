## Include the helper functions for writing tests:

## Read in BXD data:
pheno_file = string(@__DIR__, "/../data/bxdData/spleen-pheno-nomissing.csv")
pheno_raw = readdlm(pheno_file, ',', header = false);
pheno = pheno_raw[2:end, 2:(end-1)].*1.0;
pheno_id = 7919
pheno_y = reshape(pheno[:, pheno_id], :, 1);

geno_file = string(@__DIR__, "/../data/bxdData/spleen-bxd-genoprob.csv")
geno_raw = readdlm(geno_file, ',', header = false);
geno = geno_raw[2:end, 1:2:end] .* 1.0;

kinship = calcKinship(geno) |> x -> round.(x, digits = 12); # calculate k

# Write to CSV files for running lmmlite in R for testing:
# writedlm("test/run-lmmlite_R/processed_bxdData/BXDpheno.csv",  pheno, ',');
# writedlm("test/run-lmmlite_R/processed_bxdData/BXDgeno.csv", geno, ',');
# writedlm("test/run-lmmlite_R/processed_bxdData/BXDkinship.csv", kinship, ',');

nperms = 1024; # number of permutated copies required;
n = size(pheno, 1);
m = size(pheno, 2);
p = size(geno, 2); # number of markers
