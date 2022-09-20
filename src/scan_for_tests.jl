# Implementations of scan functions for testing purposes:

function permuteHelper(y::Array{Float64, 2}, g::Array{Float64, 2}, K::Array{Float64, 2}; 
                       nperms::Int64 = 1024, rndseed::Int64 = 0, 
                       reml::Bool = false, original::Bool = true)

    # n - the sample sizes
    n = size(g, 1)

    # make intercept
    intercept = ones(n, 1)
    # rotate data so errors are uncorrelated
    (y0, X0, lambda0) = rotateData(y, [intercept g], K)

    ## Note: estimate once the variance components from the null model and use for all marker scans
    # fit lmm
    vc = fitlmm(y0, reshape(X0[:, 1], :, 1), lambda0; reml = reml)
    # println(vc) # vc.b is estimated through weighted least square
    r0 = y0 - X0[:, 1]*vc.b

    # weights inversely-proportional to the variances
    sqrtw = sqrt.(makeweights(vc.h2, lambda0))
    # println(sqrtw) 
    
    # rescale by weights; now these have the same mean/variance and are independent
    ## NOTE: although rowDivide! makes in-place changes to the inputs, it only modifies the rotated data which are returned outputs
    rowMultiply!(r0, sqrtw)
    rowMultiply!(X0, sqrtw)
    X00 = resid(X0[:, 2:end], reshape(X0[:, 1], :, 1)) # after re-weighting X, calling resid on re-weighted X is the same as doing wls too.

    ## random permutations; the first column is the original data
    rng = MersenneTwister(rndseed);
    r0perm = shuffleVector(rng, r0[:, 1], nperms; original = original) # permutation on r0 which have iid standard normal distribution under null

    return (r0perm, X00, sqrtw, y0, lambda0)
end

function calcLod_perms(y_perms::Array{Float64, 2}, X_i::Array{Float64, 1})

    n = size(y_perms, 1); # number of observations for that trait
    
    rss0 = permutedims(mapslices(x -> sum(x .^2), y_perms; dims = 1));
    rss1_i = permutedims(rss(y_perms, reshape(X_i, :, 1))); # a matrix, each column is the rss after regressing each permuted y_star on the current marker
    lod_i = (n/2)*(log10.(rss0) .- log10.(rss1_i));
    return lod_i

end

function calcLod_perms2(y_perms::Array{Float64, 2}, X_i::Array{Float64, 1})

    (n, np) = size(y_perms); # number of observations for that trait; number of permutations of that trait
    
    rss0 = repeat([sum(y_perms[:, 1].^2)], inner = np);
    rss1_i = permutedims(rss(y_perms, reshape(X_i, :, 1))); # a matrix, each column is the rss after regressing each permuted y_star on the current marker
    lod_i = (n/2)*(log10.(rss0) .- log10.(rss1_i));
    return lod_i

end

function calcLod_perms3(y_perms::Array{Float64, 2}, X_i::Array{Float64, 1})

    n = size(y_perms, 1); # number of observations for that trait; number of permutations of that trait
    
    rss0 = sum(y_perms[:, 1].^2);
    rss1_i = permutedims(rss(y_perms, reshape(X_i, :, 1))); # a matrix, each column is the rss after regressing each permuted y_star on the current marker
    lod_i = (-n/2)*(log10.(rss1_i) .- log10(rss0))
    return lod_i

end

function scan_perms_toCompare(y::Array{Float64, 2}, g::Array{Float64,2}, K::Array{Float64,2}; 
    choice::Int64 = 1,
    nperms::Int64 = 1024, rndseed::Int64 = 0, 
    reml::Bool = false, original::Bool = true)

    # check the number of traits as this function only works for permutation testing of univariate trait
    if(size(y, 2) != 1)
        error("Can only handle one trait.")
    end

    (r0perm, X00) = permuteHelper(y, g, K; nperms = nperms, rndseed = rndseed, reml = reml);

    if choice == 1
        lod = map(eachcol(X00)) do col
            calcLod_perms(r0perm, Array{Float64, 1}(col))
        end
    elseif choice == 2
        lod = map(eachcol(X00)) do col
            calcLod_perms2(r0perm, Array{Float64, 1}(col))
        end
    else
        lod = map(eachcol(X00)) do col
            calcLod_perms3(r0perm, Array{Float64, 1}(col))
        end
    end

    return lod

end

function scan_perms2(y::Array{Float64,2}, g::Array{Float64,2}, K::Array{Float64,2};
              nperms::Int64 = 1024, rndseed::Int64 = 0, 
              reml::Bool = false, original::Bool = true)

    ## random permutations; the first column is the original trait (after transformation)
    rng = MersenneTwister(rndseed);

    # check the number of traits as this function only works for permutation testing of univariate trait
    if(size(y, 2) != 1)
        error("Can only handle one trait.")
    end

    # n - the sample size
    # m - the number of markers
    (n, m) = size(g)

    # make intercept
    intercept = ones(n, 1)

    # rotate data so errors are uncorrelated
    (y0, X0, lambda0) = rotateData(y, [intercept g], K)
    # lambda0 = round.(lambda0; digits = 5)

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
    # rng = MersenneTwister(rndseed);
    ## permute r0 (which is an iid, standard normal distributed N-vector under the null)
    r0perm = shuffleVector(rng, r0[:, 1], nperms; original = original)

    ## Null RSS:
    # rss0 = rss(r0perm, reshape(X0[:, 1], n, 1)) original implementation; questionable and can result in negative LOD scores
    # Instead, as by null hypothesis, mean is 0. RSS just becomes the sum of squares of the residuals (r0perm's)
    # (For theoretical derivation of the results, see notebook)
    rss0 = sum(r0perm[:, 1].^2) # a scalar; bc rss0 for every permuted trait is the same under the null (zero mean);
    
    ## make array to hold Alternative RSS's for each permutated trait
    if original
        rss1 = Array{Float64, 2}(undef, nperms+1, m)
    else
        rss1 = Array{Float64, 2}(undef, nperms, m)
    end
    
    ## loop over markers
    for i = 1:m

        ## alternative rss
        @inbounds rss1[:, i] = rss(r0perm, @view X00[:, i]);
        
    end

    lod = (-n/2)*(log10.(rss1) .- log10(rss0))

    return lod

end


function sim_for_test(r0perm::Array{Float64, 2}, X00::Array{Float64, 2}; 
                        original::Bool = false)


    (n, m) = size(X00);
    nperms = size(r0perm, 2);

    rss0 = sum(r0perm[:, 1].^2) # a scalar; bc rss0 for every permuted trait is the same under the null (zero mean);
    
    ## make array to hold Alternative RSS's for each permutated trait
    if original
        rss1 = Array{Float64, 2}(undef, nperms+1, m)
    else
        rss1 = Array{Float64, 2}(undef, nperms, m)
    end
    
    ## loop over markers
    for i = 1:m

        ## alternative rss
        @inbounds rss1[:, i] = rss(r0perm, @view X00[:, i]);
        
    end

    lod = (-n/2)*(log10.(rss1) .- log10(rss0))

    return lod

end

    