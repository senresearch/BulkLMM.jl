# Scan-perms Tests - Tests for genome scan of univariate trait and its permutated copies

## Loading required libraries
using DelimitedFiles
using LinearAlgebra
using Optim
using Distributions
using Distributed
using Test
using BenchmarkTools
using CSV
using DataFrames
using Random

## Loading functions to be tested:
include("../src/scan.jl")
include("../src/kinship.jl")
include("../src/lmm.jl")
include("../src/util.jl")
include("../src/wls.jl")
include("../src/readData.jl");

##########################################################################################################
## Read-in Data (BXD data)
##########################################################################################################
pheno_file = "../data/bxdData/traits.csv"
pheno = readBXDpheno(pheno_file);
geno_file = "../data/bxdData/geno_prob.csv"
geno = readGenoProb_ExcludeComplements(geno_file);
kinship = CSV.File("../../data/BXDkinship.csv") |> DataFrame |> Matrix;




##########################################################################################################
## TEST: resid()
##########################################################################################################