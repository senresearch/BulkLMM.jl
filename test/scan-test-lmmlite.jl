# This test file will test univariate scan functions by running on BXD data and comparing results with R lmmlite package

## Note: make sure pwd() is "BulkLMM.jl/test"

## Load required packages:
using DelimitedFiles
using LinearAlgebra
using Optim
using Distributions
using Random
using CSV
using DataFrames
### For writing tests:
using Test
using BenchmarkTools

## Include the source code of BulkLMM to be tested:
include("../src/scan.jl");
include("../src/transform_helpers.jl");
include("../src/lmm.jl");
include("../src/wls.jl");
include("../src/util.jl");
include("../src/kinship.jl");
include("../src/readData.jl");

## Also include the helper functions for writing tests:
include("testHelpers.jl");

## Read in BXD data:
pheno_file = "../data/bxdData/BXDtraits.csv"
pheno = readBXDpheno(pheno_file);
geno_file = "../data/bxdData/BXDgeno_prob.csv"
geno = readGenoProb_ExcludeComplements(geno_file);

kinship = calcKinship(geno); # calculate kinship matrix from genotype data

## Consider the 7919-th trait
pheno_y = reshape(pheno[:, 7919], :, 1);

## Run BulkLMM code to get results:
reml_results = scan(pheno_y, geno, kinship; reml = true); # by default uses scan_null;
ml_results = scan(pheno_y, geno, kinship; reml = false);

lods_BulkLMM_reml = reshape(reml_results.lod, :, 1);
lods_BulkLMM_ml = reshape(ml_results.lod, :, 1);


## Compare with lmmlite results:
## Read in lmmlite results:
reml_results_lmmlite = CSV.read("run-lmmlite_R/output/result.lmmlite_REML.csv", DataFrame);
ml_results_lmmlite = CSV.read("run-lmmlite_R/output/result.lmmlite_ML.csv", DataFrame);

lods_lmmlite_reml = parse.(Float64, reshape(reml_results_lmmlite[2:end, 5], :, 1));
lods_lmmlite_ml = parse.(Float64, reshape(ml_results_lmmlite[2:end, 5], :, 1));

## Testings:

@testset "lmmlite_results_tests" begin
    tol = 1e-9;
    @test sumSqDiff(lods_lmmlite_reml, lods_BulkLMM_reml) <= sqrt(tol);
    @test sumSqDiff(lods_lmmlite_ml, lods_BulkLMM_ml) <= sqrt(tol);
    @test maxSqDiff(lods_lmmlite_reml, lods_BulkLMM_reml) <= tol;
    @test maxSqDiff(lods_lmmlite_ml, lods_BulkLMM_ml) <= tol;
end;
