function transform1(y::Array{Float64, 2}, g::Array{Float64, 2}, K::Array{Float64, 2})
        
    # n - the sample sizes
    n = size(g, 1);

    # make intercept
    intercept = ones(n, 1);
    # rotate data so errors are uncorrelated
    (y0, X0, lambda0) = rotateData(y, [intercept g], K);

    return (y0, X0, lambda0);

end

# Take the rotated data
function transform2(y0::Array{Float64, 2}, X0::Array{Float64, 2}, lambda0::Array{Float64, 1}; 
                    nperms::Int64 = 1024, rndseed::Int64 = 0, 
                    reml::Bool = false, original::Bool = true)

        ## Note: estimate once the variance components from the null model and use for all marker scans
        # fit lmm
        vc = fitlmm(y0, reshape(X0[:, 1], :, 1), lambda0; reml = reml)
        # println(vc) # vc.b is estimated through weighted least square
        r0 = y0 - X0[:, 1]*vc.b

        # weights inversely-proportional to the variances
        sqrtw = sqrt.(makeweights(vc.h2, lambda0))

        # rescale by weights; now these have the same mean/variance and are independent
        ## NOTE: although rowDivide! makes in-place changes to the inputs, it only modifies the rotated data which are returned outputs
        rowMultiply!(r0, sqrtw)
        rowMultiply!(X0, sqrtw) 
        X00 = resid(X0[:, 2:end], reshape(X0[:, 1], :, 1)) # after re-weighting X, calling resid on re-weighted X is the same as doing wls too.

        ## random permutations; the first column is the original data
        rng = MersenneTwister(rndseed);
        r0perm = shuffleVector(rng, r0[:, 1], nperms; original = original) # permutation on r0 which have iid standard normal distribution under null

        return (r0perm, X00)

end

function scan_distributed(y0::Array{Float64, 2}, X0::Array{Float64, 2}, lambda0::Array{Float64, 1}; 
                    nperms::Int64 = 1024, rndseed::Int64 = 0, 
                    reml::Bool = false, original::Bool = true)

        (r0perm, X00) = transform2(y0, X0, lambda0; 
                                    nperms = nperms, rndseed = rndseed, reml = reml, original = original);

        (n, m) = size(X00);

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
            rss1[:, i] = rss(r0perm, @view X00[:, i]);
        
        end

        lod = (-n/2)*(log10.(rss1) .- log10(rss0))

        return lod
        
end