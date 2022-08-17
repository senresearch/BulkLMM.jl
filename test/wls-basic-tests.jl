# WLS Functions Tests - basic tests for resid, rss functions

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


## Simulate multivariate traits data
N = 100;
m = rand([2, 5]); # randomly generate the number of rows
p = rand([m-1, 5]);
    
Y = rand([0.0, 1.0], N, m); # e.g., m Quantitative Traits
X = rand([0.0, 1.0], N, p);
X[:, 1] = ones(N) # adding the intercept;

model_resids = resid(Y, X);

@test size(model_resids) == (N, m)

test_resids = zeros(Float64, N, m)

for t in 1:m
    
    y = Y[:, t];
    
    b = X\y;
    yhat = X*b;
    curr_res = y .- yhat;
    
    test_resids[:, t] = curr_res
    
end 