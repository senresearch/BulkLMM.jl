# Implementations of scan functions for testing purposes:



function permuteHelper(y::Array{Float64, 2}, g::Array{Float64, 2}, K::Array{Float64, 2}; 
                       nperms::Int64 = 1024, rndseed::Int64 = 0, reml::Bool = false)

    # n - the sample sizes
    n = size(g, 1)

    # make intercept
    intercept = ones(n, 1)
    # rotate data so errors are uncorrelated
    (y0, X0, lambda0) = rotateData(y, [intercept g], K)

    ## Note: estimate once the variance components from the null model and use for all marker scans
    # fit lmm
    vc = fitlmm(y0, reshape(X0[:,1], :, 1), lambda0; reml = reml) # vc.b is estimated through weighted least square
    r0 = y0 - X0[:,1]*vc.b

    # for testing
    #=     println("y: ")
    println(r0[1:6, 1]) =# 


    # weights inversely-proportional to the variances
    wts = makeweights(vc.h2, lambda0)

    #=     println("weights: ")
    println(wts[1:6]) =#

    # rescale by weights; now these have the same mean/variance and are independent
    ## NOTE: although rowDivide! makes in-place changes to the inputs, it only modifies the rotated data which are returned outputs
    rowMultiply!(r0, sqrt.(wts))
    rowMultiply!(X0, sqrt.(wts))
    X00 = resid(X0[:, 2:end], reshape(X0[:, 1], :, 1)) # after re-weighting X, calling resid on re-weighted X is the same as doing wls too.

    #=     println("X: ")
    println(X00[1:6, 2]) =#

    ## random permutations; the first column is the original data
    rng = MersenneTwister(rndseed);
    r0perm = shuffleVector(rng, r0[:, 1], nperms; original = true) # permutation on r0 which have iid standard normal distribution under null

    return (r0perm, X00)
end