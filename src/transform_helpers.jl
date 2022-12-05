function transform_rotation(y::Array{Float64, 2}, g::Array{Float64, 2}, K::Array{Float64, 2}; addIntercept::Bool = true)
        
    # n - the sample sizes
    n = size(y, 1)

    # check dimensions
    if((size(g, 1) != n)|(size(K, 1)!= n))
        throw(error("Dimension mismatch."))
    end

    # make intercept
    if addIntercept
        intercept = ones(n, 1);
        X = [intercept g];
    else
        X = g; # Safe; will not make in-place changes to X.
    end

    ## Eigen-decomposition:
    EF = eigen(K);
    Ut = EF.vectors';

    # return an error if there are any negative eigenvalues
    if any(EF.values .< -1e-7)
        throw(error("Negative eigenvalues exist. The kinship matrix supplied may not be SPD."));
    end

    # rotate data so errors are uncorrelated

    return Ut*y, Ut*X, EF.values

end

# Takes the rotated data, evaluates the VC estimators (only based on the intercept model) for weights calculation, and finally re-weights the input data.
function transform_reweight(y0::Array{Float64, 2}, X0::Array{Float64, 2}, lambda0::Array{Float64, 1};
                    prior_a::Float64 = 0.0, prior_b::Float64 = 0.0, method::String = "qr",
                    reml::Bool = false)

        ## Note: estimate once the variance components from the null model and use for all marker scans
        # fit lmm
        vc = fitlmm(y0, reshape(X0[:, 1], :, 1), lambda0, [prior_a, prior_b]; reml = reml, method = method);
        # println(vc) # vc.b is estimated through weighted least square
        r0 = y0 - X0[:, 1]*vc.b

        # weights inversely-proportional to the variances
        sqrtw = sqrt.(makeweights(vc.h2, lambda0))
    
        # make copies so that the original data will not be overwritten
        copy_X0 = copy(X0);
        copy_r0 = copy(r0);
        # rescale by weights; now these have the same mean/variance and are independent
        ## NOTE: although rowDivide! makes in-place changes to the inputs, it only modifies the rotated data which are returned outputs
        rowMultiply!(copy_r0, sqrtw) # dont want to change the inputs: r0, X0
        rowMultiply!(copy_X0, sqrtw) # dont want to change the inputs: r0, X0
        X00 = resid(copy_X0[:, 2:end], reshape(copy_X0[:, 1], :, 1)) # after re-weighting X, calling resid on re-weighted X is the same as doing wls too.

        return (copy_r0, X00, vc.sigma2)

end

function transform_permute(r0::Array{Float64, 2}; 
                    nperms::Int64 = 1024, rndseed::Int64 = 0, original::Bool = true)

        ## random permutations; the first column is the original data
        rng = MersenneTwister(rndseed);
        r0perm = shuffleVector(rng, r0[:, 1], nperms; original = original) # permutation on r0 which have iid standard normal distribution under null
    
        return r0perm
end