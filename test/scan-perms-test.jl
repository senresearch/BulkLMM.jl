# Scan-perms Tests - Tests for genome scan of univariate trait and its permutated copies


## Loading required libraries
using DelimitedFiles
using LinearAlgebra
using Optim
using Distributions
using CSV
using DataFrames
using Random

using Test
using BenchmarkTools

## Loading the code to be tested and supporting functions for testing:
include("../src/scan.jl")
include("../src/kinship.jl")
include("../src/lmm.jl")
include("../src/util.jl")
include("../src/wls.jl")
include("../src/readData.jl");

##########################################################################################################
## Read-in Data (BXD data; kinship matrix data is by running kinship.jl on BXD genotype probability data)
##########################################################################################################
pheno_file = "BulkLMM.jl/data/bxdData/BXDtraits.csv"
pheno = readBXDpheno(pheno_file);
geno_file = "BulkLMM.jl/data/bxdData/BXDgeno_prob.csv"
geno = readGenoProb_ExcludeComplements(geno_file);
kinship = CSV.File("BulkLMM.jl/data/bxdData/BXDkinship.csv") |> DataFrame |> Matrix;

## load functions to help testing
include("../src/scan_for_tests.jl");

## Helper functions for comparing results:
function maxSqDiff(a::Array{Float64, 2}, b::Array{Float64, 2})

    return maximum((a .- b) .^2)

end

function sumSqDiff(a::Array{Float64, 2}, b::Array{Float64, 2})

    return sum((a .- b) .^2)

end

##########################################################################################################
## TEST: Compare results of scans for a (small) random number of permutations
##########################################################################################################


## For testing purposes, we consider the case when we need to perform scans for the second trait in the BXD traits data:
pheno_2 = reshape(pheno[:, 2], :, 1);


function toCompare(y::Array{Float64, 2}, g::Array{Float64, 2}, K::Array{Float64, 2}, nperms::Int64)

    (n, p) = size(g);

    ## Perform data transformations such that the resulting trait data is i.i.d standard normal distributed under the null:
    (y_star_perms, X_star) = permuteHelper(y, g, K; nperms = nperms, reml = true);
    m = size(y_star_perms, 2); # number of permuted copies (may include the original)

    function one_trait_at_a_time(y_star_i::Array{Float64, 2}, X_star::Array{Float64, 2})

        lod_for_one = zeros(1, p);

        rss0 = sum(y_star_i .^2);
    
        ## loop over markers
        for i in 1:p

            ## alternative rss
            rss1 = rss(y_star_i, reshape(X_star[:, i], :, 1))[1, 1]

            # println(size(rss1))

            ## calculate LOD score and assign
            lod_for_one[1, i] = (n/2)*(log10(rss0) - log10(rss1))

        end

        return lod_for_one

    end

    lod_for_all = zeros(m, p)

    for i in 1:m

        y_star_i = reshape(y_star_perms[:, i], :, 1);
        lod_for_all[i, :] = one_trait_at_a_time(y_star_i, X_star)
    
    end

    return lod_for_all

end


nperms = rand(1:10);

## Get LOD results from running the scan_perms implementation:
results_perms = scan_perms(pheno_2, geno, kinship; nperms = nperms);

results_toCompare = toCompare(pheno_2, geno, kinship, nperms);


tol = 1e-6

##########################################################################################################
## TEST: Compare the LODs of scans on the original with the LODs from directly applying scan_null on the original
##########################################################################################################

null_results = scan(pheno_2, geno, kinship; reml = false, method = "null");
null_LODs = reshape(null_results[3], :, 1);


##########################################################################################################
## TEST: Check if wls() on the unweighted data and ls() on the weighted data gives the same coefficients
## and residuals...
##########################################################################################################


## residuals by re-weight first, and then perform OLS (the approach in original implementation)
function transformX_1(X::Array{Float64, 2}, weights::Vector{Float64})

    # rescale by weights; now these have the same mean/variance and are independent
    rowDivide!(X, sqrt.(weights))
    X_star = resid(X[:, 2:end], reshape(X[:, 1], :, 1)) # after re-weighting X, calling resid on re-weighted X is the same as doing wls too.

    return X_star

end

## residuals by WLS
function transformX_2(X::Array{Float64, 2}, weights::Vector{Float64})

    ests = wls(X[:, 2:end], reshape(X[:, 1], :, 1), weights)
    X_star = X[:, 2:end] - X[:, 1]*ests.b;
    rowDivide!(X_star, sqrt.(weights));
    
    return X_star

end

# n - the sample sizes
n = size(geno, 1);

# make intercept
intercept = ones(n, 1);
# rotate data so errors are uncorrelated
(y0, X0, lambda0) = rotateData(pheno_2, [intercept geno], kinship);
dc = deepcopy(X0);

## Note: estimate once the variance components from the null model and use for all marker scans
# fit lmm
vc = fitlmm(y0, reshape(X0[:, 1], :, 1), lambda0; reml = false); # vc.b is estimated through weighted least square

# weights proportional to the variances
# wts = makeweights(vc.h2, lambda0);
wts = 1.0 ./ collect(1:79) # for testing

X_star1 = transformX_1(X0, wts);
X0 = dc;
X_star2 = transformX_2(X0, wts);


@btime transformX_1(X0, wts);

X0 = dc;

@btime transformX_2(X0, wts);


(y0, X0, lambda0) = rotateData(pheno_2, [intercept geno], kinship);
dc = deepcopy(X0);

b1 = wls(X0[:, 2:end], reshape(X0[:, 1], :, 1), wts);

resid_wls = X0[:, 2:end] .- X0[:, 1]*b1.b;
resid_wls = mapslices(x -> x .* sqrt.(wts), resid_wls; dims = 1);

X0 = dc;
step_alt = quote

    rowDivide!(X0, 1.0 ./ sqrt.(wts));
    b2 = ls(X0[:, 2:end], reshape(X0[:, 1], :, 1));

end
eval(step_alt)

resid_ls = resid(X0[:, 2:end], reshape(X0[:, 1], :, 1));

##########################################################################################################
## TEST: Run Tests
##########################################################################################################

@testset "testScanPerms" begin

    # pre-algorithm check if the function modifies the original inputs:

    y_copy = deepcopy(pheno_2);
    scan_perms(pheno_2, geno, kinship; nperms = 1);
    @test maxSqDiff(y_copy, pheno_2) <= tol;    

    # tests for comparing one scan for one permutation at a time v.s. perform scans for all permutations by operations on matrices
    @test maxSqDiff(results_perms, results_toCompare) <= tol;
    @test sumSqDiff(results_perms, results_toCompare) <= sqrt(tol);

    # tests for comparing the first row of results from permutation testings (on original y) v.s. applying scan_null on the original data
    @test maxSqDiff(reshape(results_perms[1, :], :, 1), null_LODs) <= tol;
    @test sumSqDiff(reshape(results_perms[1, :], :, 1), null_LODs) <= sqrt(tol);

    # tests to verify the estimates given by two mathematically equivalent transformations are the same
    @test sumSqDiff(b1.b, b2.b) <= tol;
    @test sumSqDiff(resid_wls, resid_ls) <= tol

end



##########################################################################################################
## BENCHMARK:
##########################################################################################################

@btime scan_perms(pheno_2, geno, kinship; nperms = 50);
@btime toCompare(pheno_2, geno, kinship, 50);