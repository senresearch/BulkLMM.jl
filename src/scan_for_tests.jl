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

function calcLod_perms(rss0::Array{Float64, 2}, y_perms::Array{Float64, 2}, X_i::Array{Float64, 1})

    n = size(y_perms, 1); # number of observations for that trait
    rss1_i = rss(y_perms, reshape(X_i, :, 1)); # a matrix, each column is the rss after regressing each permuted y_star on the current marker
    lod_i = (n/2)*(log10.(rss0) .- log10.(rss1_i));
    return lod_i

end

function scan_perms_toCompare(y::Array{Float64,2}, g::Array{Float64,2}, K::Array{Float64,2};

    nperms::Int64 = 1024, rndseed::Int64 = 0, 
    reml::Bool = true, original::Bool = true)

    # check the number of traits as this function only works for permutation testing of univariate trait
    if(size(y,2) != 1)
        error("Can only handle one trait.")
    end

    # n - the sample size
    # m - the number of markers
    (n, m) = size(g)

    # make intercept
    intercept = ones(n, 1)

    # rotate data so errors are uncorrelated
    (y0, X0, lambda0) = rotateData(y, [intercept g], K)


    ## Note: estimate once the variance components from the null model and use for all marker scans
    # fit lmm

    # X0_intercept = @view X0[:, 1] # to compare
    vc = fitlmm(y0, reshape(X0[:, 1], :, 1), lambda0; reml = reml) # vc.b is estimated through weighted least square
    r0 = y0 - X0[:, 1]*vc.b

    # weights proportional to the variances
    sqrtw = sqrt.(makeweights(vc.h2, lambda0))

    # compared runtime of the following with "wls(X0[:, 2:end], X0[:, 1], wts)" ?
    # rescale by weights; now these have the same mean/variance and are independent
    # rowDivide!(r0, 1.0./sqrt.(wts))
    # rowDivide!(X0, 1.0./sqrt.(wts))
    rowMultiply!(r0, sqrtw);
    rowMultiply!(X0, sqrtw);


    # after re-weighting X, calling resid on re-weighted X is the same as doing wls on the X after rotation.
    X00 = resid(X0[:, 2:end], reshape(X0[:, 1], :, 1)) # consider not using sub-array, consider @view; in-place changes

    ## random permutations; the first column is the original trait (after transformation)
    rng = MersenneTwister(rndseed);
    ## permute r0 (which is an iid, standard normal distributed N-vector under the null)
    r0perm = shuffleVector(rng, r0[:, 1], nperms; original = original)

    ## Null RSS:
    # rss0 = rss(r0perm, reshape(X0[:, 1], n, 1)) original implementation; questionable and can result in negative LOD scores
    # Instead, as by null hypothesis, mean is 0. RSS just becomes the sum of squares of the residuals (r0perm's)
    # (For theoretical derivation of the results, see notebook)
    rss0 = mapslices(x -> sum(x .^2), r0perm; dims = 1);

    lod = mapslices(x -> calcLod_perms(rss0, r0perm, x), X00; dims = 1)

    return lod

end