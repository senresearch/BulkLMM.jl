# WLS Functions Tests - basic tests for resid, rss functions

## Loading required libraries
using Random
using LinearAlgebra
using Statistics
using Test
using BenchmarkTools

## Loading functions to test
include("../src/wls.jl")
include("../src/util.jl")

function ls(y::Array{Float64, 2}, X::Array{Float64, 2};
    reml::Bool = false, loglik = true)

    # number of individuals
    n = size(y,1)
    # number of covariates
    p = size(X,2)

    b = X\y # uses QR decomposition
    yhat = X*b
    rss0 = sum((y-yhat).^2)

    if( reml )
    sigma2 = rss0/(n-p)
    else
    sigma2 = rss0/n
    end

    if(loglik) 
    if ( reml )
        logdetSigma = (n-p)*log(sigma2)
    else
        logdetSigma = n*log(sigma2)
    end

    ell = -0.5 * ( logdetSigma + rss0/sigma2 )
    else
    ell = missing
    end

    return LSEstimates(b, sigma2, ell)

end

## Simulate multivariate traits data
N = 100;
m = 3; # number of traits to perform scan on
p = rand(collect(2:5));
    
Y = rand(N, m); # e.g., m Quantitative Traits
X = rand(N, p);
X[:, 1] = ones(N) # adding the intercept;



##########################################################################################################
## TEST: resid()
##########################################################################################################

### A helper function to calculate the squared difference between two arrays
function biasSquared(est::AbstractArray{Float64, 2}, truth::AbstractArray{Float64, 2})
    
    bias = est - truth
    return(sum(bias.^2))
    
end

tol = 1e-8;

model_resids = resid(Y, X);

## Test1: simply check the output dimensions

test1_resids = quote
    @test size(model_resids) == (N, m)
end

### Test2: compare results uisng both methods (cholesky v.s. qr)
test2_resids = quote
    @test biasSquared(resid(Y, X), resid(Y, X, "qr")) <= tol
end

## Test3: compare with manually computed residuals
test_resids = zeros(Float64, N, m)

for t in 1:m
    y = Y[:, t];
    
    # perform OLS for each trait
    b = X\y;
    yhat = X*b;
    curr_res = y .- yhat;
    
    test_resids[:, t] = curr_res
end

test3_resids = quote
    @test biasSquared(test_resids, model_resids) <= tol 
end


##########################################################################################################
## TEST: rss()
##########################################################################################################

model_rss = rss(Y, X);
test_rss = reshape([sum(test_resids[:, 1].^2), 
                    sum(test_resids[:, 2].^2),
                    sum(test_resids[:, 3].^2)], 1, :);


tests_rss = quote
    @test biasSquared(test_rss, model_rss) <= tol
end



##########################################################################################################
## RUN TESTS:
##########################################################################################################
@testset "Basic wls() Tests" begin
    eval(test1_resids)
    eval(test2_resids)
    eval(test3_resids)
    eval(tests_rss)
end

##########################################################################################################
## BENCHMARKING:
##########################################################################################################

function toCompare!(Y::Array{Float64, 2}, X::Array{Float64, 2}, m::Int64, holder::Array{Float64, 2})

    for t in 1:m
    
        y = Y[:, t];
        
        b = X\y;
        yhat = X*b;
        curr_res = y .- yhat;
        
        holder[:, t] = curr_res
        
    end

end

@btime model_resids = resid(Y, X)
@btime toCompare!(Y, X, m, test_resids)

 