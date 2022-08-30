# WLS Functions Tests - by comparing final outputs running on simulated heteroskedestic data

## Loading required libraries
using Random
using Distributions
using LinearAlgebra
using Statistics
using Test
using BenchmarkTools

## Loading functions to test
include("../src/wls.jl")
include("../src/util.jl")


##########################################################################################################
## Simulation Settings:
##########################################################################################################
## Generate heteroscedestic data:
function generateErrors(V::AbstractArray{Float64, 2})

    return MvNormal(V)

end

function generateErrors(v::AbstractArray{Float64, 1})

    V = diagm(v)

    return MvNormal(V)

end

N = 100; # number of observations
p = 1; # 1-df test
beta = [1.0, 0.5]; # true effects plus intercept

# Construct means
group = repeat([0.0, 1.0], inner = Int64(N/2));
mean = beta[1] .+ group .* beta[2];

# Construct heteroscedestic errors
vars = rand(Uniform(0, 0.1), 100);
errors_dist = generateErrors(vars);
errors = rand(errors_dist, 1);

# Finally, construct heteroscedestic dependent variables
y = mean .+ errors;

X = hcat(repeat([1.0], inner = N), group) # design matrix
weights = 1.0 ./ sqrt.(vars); # construct weights by standard deviations


##########################################################################################################
## TEST:
##########################################################################################################

# Get results from wls implementation
result_wls = wls(y, X, weights; reml = false, loglik = true, method = "cholesky");

# Get weighted LS results by manually weighting and then perform OLS:
W = weights .* (1.0*Matrix(I, N, N));
Wy = sqrt.(W) * y;
WX = sqrt.(W) * X;

result_ls = ls(Wy, WX; reml = false, loglik = true)

# Function to compare two solutions by returning the sum of squared differences
function biasSquared(est::AbstractArray{Float64, 2}, truth::AbstractArray{Float64, 2})
    
    bias = est - truth
    return(sum(bias.^2))
    
end

tol = 1e-2;
truth = reshape([1.0 0.5], 2, 1);

@testset "simple test WLS" begin
    @test biasSquared(result_wls.b, result_ls.b) <= tol^4
    @test biasSquared(result_wls.b, truth) <= tol
    @test biasSquared(result_ls.b, truth) <= tol
end;

##########################################################################################################
## BENCHMARKING:
##########################################################################################################

@btime wls(y, X, weights; reml = false, loglik = true, method = "qr");

alternative = quote
    W = weights .* (1.0*Matrix(I, N, N));
    Wy = sqrt.(W) * y;
    WX = sqrt.(W) * X;

    ls(Wy, WX; reml = false, loglik = true);
end

@btime eval(alternative);