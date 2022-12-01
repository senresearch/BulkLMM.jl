## Include the helper functions for writing tests:
include("../src/kinship.jl");
include("../src/readData.jl");
include("testHelpers.jl");


## Read in BXD data:
pheno_file = "../data/bxdData/BXDtraits.csv"
pheno = readBXDpheno(pheno_file);
geno_file = "../data/bxdData/BXDgeno_prob.csv"
geno = readGenoProb_ExcludeComplements(geno_file);

kinship = calcKinship(geno); # calculate k

nperms = 1024; # number of permutated copies required;
n = size(pheno, 1);
m = size(pheno, 2);
p = size(geno, 2); # number of markers