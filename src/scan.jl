###########################################################
# Genome scan functions for a single trait plus permutation testing:
# allow modeling of additional non-genetic covariates, two genotype groups
###########################################################
"""
    scan(y, G, K; optional inputs) 
    scan(y, G, Z, K; optional inputs) - if modeling additional covariates Z

Perform genome scan for univariate trait and a set of genome markers

# Required Inputs

- `y::Array{Float64, 2} or Array{Float64, 1}`: Single univariate quantitative trait of N measurements (dimension: N*1)
- `G::Array{Float64, 2}`: Matrix of genotype probabilities at p tested markers (dimension: N*p)
- `K::Array{Float64, 2}`: Genetic relatedness matrix of the N subjects (dimension:N*N)

# Optional Inputs

## Essential Inputs:
- `addIntercept::Bool`: Option to add an intercept column to the design matrix (default: true)
- `reml::Bool`: Option of the scheme for estimating variance components (by REML or ML; default: false)
- `assumption::String`: Keyword argument indicating whether to estimate the variance components independently 
    for each marker ("alt") or to estimate once for the null model and use for testing all markers ("null)
    (default: "null")
- `output_pvals::Bool`: Option to additionally report the LRT p-values (default: false)

## Modeling Additional Covariates:   
- `Z::AbstractArray{Float64, 2}`: Matrix of additional non-genetic covariates (should be independent to tested 
    markers)

## Permutation Testing: 
- `permutation_test::Bool`: Option to perform permutation testing on the studied single trait (default: false)
- `nperms::Int64`: The number of permutations required, an integer (default: 1024)
- `rndseed::Int64`: An integer random seed set for performing random shuffling of the original trait 
    (default: 0)

## Structure of Weighted Residual Variances:
- `weights::Array{Float64, 1}`: Optional weights for modeling unequal, weighted structure of the residual variances 
    of the trait (default: Missing, i.e. equal residual variances)

## Numerical Techniques - for stabilizing the heritability optimization step
- `optim_interval::Int64`: The number of sub-divided regions of the interval [0, 1) of heritability to perform each 
    numerical optimization scheme (the Brent's method) on (default: 1, i.e. the entire interval)
- `prior_variance::Float64`: Scale parameter of the prior Scaled Inv-Chisq distributed residual variances (default: 0)
- `prior_sample_size::Float64`: Degree of freedom parameter of the prior Scaled Inv-Chisq distributed residual 
    variances (default: 0)

## Examining Profile Likelihood - as a function of the heritability estimates
- `ProfileLL::Bool`: Option to return values of the profile likelihood function under different h2 values 
    (default: false)
- `markerID::Int64`: The ID of the marker of interest
- `h2_grid::Array{Float64, 1}`: Different values of h2 for calculating the corresponding profile likelihood values
    (default: an empty array)

## Other Inputs:
- `method::String`: Keyword indicating the matrix factorization scheme for model evaluation; either by "qr" or 
    "cholesky" decomposition (default: "qr")
- `decomp_scheme::String`: Keyword indicating the decomposition scheme for the kinship matrix; either by "eigen" 
    or "svd" decomposition (default: "eigen")

# Returned Values:

The output of the single-trait scan function is an object. Depending on the user inputs and options, the fields of
    the output object will differ. For example, for the returned output named as `out`:

## Null-LMM: by the "Null" approximation of the h2 value and applied to testing all markers:
- `out.sigma2_e::Float64`: Estimated residual unexplained variances from the null model
- `out.h2_null::Float64`: Estimated heritability (h2) from the null model
- `out.lod::Array{Float64, 1}`: 1-dimensional array consisting of the LOD scores

## Exact-LMM: by re-estimating the h2 and sigma2_e independently while testing each marker:
- `out.sigma2_e::Float64`: Estimated residual unexplained variances from the null model
- `out.h2_null::Float64`: Estimated heritability (h2) from the null model
- `out.h2_each_marker::Array{Float64, 1}`: 1-dimensional array of the estimated heritability for each marker model
- `out.lod::Array{Float64, 1}`: 1-dimensional array consisting of the LOD scores

## Null-LMM and when permutation testing is required:
- `out.sigma2_e::Float64`: Estimated residual unexplained variances from the null model
- `out.h2_null::Float64`: Estimated heritability (h2) from the null model
- `out.lod::Array{Float64, 1}`: 1-dimensional array consisting of the LOD scores
- `out.L_perms::Array{Float64, 2}`: 2-dimensional array of the LOD scores from permutation testing; each column
    is a vector of length p of p LOD scores for each permuted copy.

## If the option for reporting p-values is on, the p-values results will be returned as:
- `out.log10pvals::Array{Float64, 1}`: 1-dimensional array consisting of the -log10(p-values)
- `out.log10Pvals_perms::Array{Float64, 2}`: 2-dimensional array consisting of the -log10(p-values) for each test
(for testing the association between each marker and each permuted trait).

## Additionally, if the user wants to examine the profile likelihood values under a given set of h2-values:
- `out.ll_list_null::Array{Float64, 1}`: gives the values under the null model under each h2-value
- `out.ll_list_alt::Array{Float64, 1}`: gives the values under the user-specified marker model under each h2-value

"""
function scan(y::Array{Float64, 1}, g::Array{Float64, 2}, K::Array{Float64, 2};
              # weighted environmental variances:
              weights::Union{Missing, Array{Float64, 1}} = missing,
              # regularization options:
              prior_variance::Float64 = 0.0, prior_sample_size::Float64 = 0.0, addIntercept::Bool = true,
              # vc estimation options:
              reml::Bool = false, assumption::String = "null", method::String = "qr", optim_interval::Int64 = 1,
              # permutation testing options:
              permutation_test::Bool = false, nperms::Int64 = 1024, rndseed::Int64 = 0,
              # option for inspecting h2 estimation process:
              profileLL::Bool = false, markerID::Int = 0, h2_grid::Array{Float64, 1} = Array{Float64, 1}(undef, 1),
              # option for kinship decomposition scheme:
              decomp_scheme::String = "eigen",
              # option for returning p-values results:
              output_pvals::Bool = false, chisq_df::Int64 = 1
              )

    return scan(reshape(y, :, 1), g, K; 
                weights = weights,
                addIntercept = addIntercept,
                prior_variance = prior_variance, prior_sample_size = prior_sample_size,
                reml = reml, assumption = assumption, method = method, optim_interval = optim_interval,
                permutation_test = permutation_test, nperms = nperms, rndseed = rndseed,
                profileLL = profileLL, markerID = markerID, h2_grid = h2_grid,
                decomp_scheme = decomp_scheme, output_pvals = output_pvals, chisq_df = chisq_df)

end

function scan(y::Array{Float64, 1}, g::Array{Float64, 2}, covar::Array{Float64, 2}, K::Array{Float64, 2};
              # weighted environmental variances:
              weights::Union{Missing, Array{Float64, 1}} = missing,
              # regularization options:
              prior_variance::Float64 = 0.0, prior_sample_size::Float64 = 0.0, addIntercept::Bool = true,
              # vc estimation options:
              reml::Bool = false, assumption::String = "null", method::String = "qr", optim_interval::Int64 = 1,
              # permutation testing options:
              permutation_test::Bool = false, nperms::Int64 = 1024, rndseed::Int64 = 0,
              # option for inspecting h2 estimation process:
              profileLL::Bool = false, markerID::Int = 0, h2_grid::Array{Float64, 1} = Array{Float64, 1}(undef, 1),
              # option for kinship decomposition scheme:
              decomp_scheme::String = "eigen",
              # option for returning p-values results:
              output_pvals::Bool = false, chisq_df::Int64 = 1
    )

    return scan(reshape(y, :, 1), g, covar, K; 
                weights = weights,
                addIntercept = addIntercept,
                prior_variance = prior_variance, prior_sample_size = prior_sample_size,
                reml = reml, assumption = assumption, method = method, optim_interval = optim_interval,
                permutation_test = permutation_test, nperms = nperms, rndseed = rndseed,
                profileLL = profileLL, markerID = markerID, h2_grid = h2_grid,
                decomp_scheme = decomp_scheme, output_pvals = output_pvals, chisq_df = chisq_df)

end 

function scan(y::Array{Float64, 2}, g::Array{Float64, 2}, K::Array{Float64, 2};
              # weighted environmental variances:
              weights::Union{Missing, Array{Float64, 1}} = missing,
              # regularization options:
              prior_variance::Float64 = 0.0, prior_sample_size::Float64 = 0.0, addIntercept::Bool = true,
              # vc estimation options:
              reml::Bool = false, assumption::String = "null", method::String = "qr", optim_interval::Int64 = 1,
              # permutation testing options:
              permutation_test::Bool = false, nperms::Int64 = 1024, rndseed::Int64 = 0,
              # option for inspecting h2 estimation process:
              profileLL::Bool = false, markerID::Int = 0, h2_grid::Array{Float64, 1} = Array{Float64, 1}(undef, 1),
              # option for kinship decomposition scheme:
              decomp_scheme::String = "eigen",
              # option for returning p-values results:
              output_pvals::Bool = false, chisq_df::Int64 = 1
              )

    if addIntercept == false
        error("Intercept has to be added when no other covariate is given.")
    end

    n = size(y, 1);
    return scan(y, g, ones(n, 1), K; 
                weights = weights,
                addIntercept = false,
                prior_variance = prior_variance, prior_sample_size = prior_sample_size,
                reml = reml, assumption = assumption, method = method, optim_interval = optim_interval,
                permutation_test = permutation_test, nperms = nperms, rndseed = rndseed,
                profileLL = profileLL, markerID = markerID, h2_grid = h2_grid,
                decomp_scheme = decomp_scheme, output_pvals = output_pvals, chisq_df = chisq_df)
end

function scan(y::Array{Float64,2}, g::Array{Float64,2}, covar::Array{Float64, 2}, K::Array{Float64,2};
              # weighted environmental variances:
              weights::Union{Missing, Array{Float64, 1}} = missing,
              # regularization options:
              prior_variance::Float64 = 0.0, prior_sample_size::Float64 = 0.0, addIntercept::Bool = true,
              # vc estimation options:
              reml::Bool = false, assumption::String = "null", method::String = "qr", optim_interval::Int64 = 1,
              # permutation testing options:
              permutation_test::Bool = false, nperms::Int64 = 1024, rndseed::Int64 = 0,
              # option for inspecting h2 estimation process:
              profileLL::Bool = false, markerID::Int = 0, h2_grid::Array{Float64, 1} = Array{Float64, 1}(undef, 1),
              # option for kinship decomposition scheme:
              decomp_scheme::String = "eigen",
              # option for returning p-values results:
              output_pvals::Bool = false, chisq_df::Int64 = 1
              )

    n = size(y, 1);

    if !ismissing(weights)
        # inv_weights = map(x -> 1/sqrt(x), weights);
        W = diagm(weights);
        y_st = W*y;
        g_st = W*g;
        
        if addIntercept == true
            covar_st = W*[ones(n) covar];
            addIntercept = false;
        else
            covar_st = W*covar;
        end
        K_st = W*K*W;

        return scan(y_st, g_st, covar_st, K_st;
                    addIntercept = false,
                    prior_variance = prior_variance, prior_sample_size = prior_sample_size,
                    reml = reml, assumption = assumption, method = method, optim_interval = optim_interval,
                    permutation_test = permutation_test, nperms = nperms, rndseed = rndseed,
                    profileLL = profileLL, markerID = markerID, h2_grid = h2_grid,
                    decomp_scheme = decomp_scheme, output_pvals = output_pvals, chisq_df = chisq_df)
    else
        y_st = y;
        g_st = g;
        covar_st = covar;
        K_st = K;
    end

    if assumption == "null"
        if permutation_test == true # if the user requires to perform permutation testing:
            if nperms <= 0
                throw(error("For permutation testing, the required number of permuted copies has to be a positive integer!"));
            end
        else # if permutation testing is not required by the user, set the number of permuted copies to 0:
            nperms = 0;
        end
        
        results = scan_perms_lite(y_st, g_st, covar_st, K_st; prior_variance = prior_variance, prior_sample_size = prior_sample_size,
                                  addIntercept = addIntercept, reml = reml, method = method, optim_interval = optim_interval,
                                  nperms = nperms, rndseed = rndseed, 
                                  decomp_scheme = decomp_scheme, output_pvals = output_pvals);
    elseif assumption == "alt"
        if permutation_test == true
            error("Permutation test option currently is not supported for the alternative assumption.");
        else
            results = scan_alt(y_st, g_st, covar_st, K_st, [prior_variance, prior_sample_size], addIntercept; 
                               reml = reml, method = method, optim_interval = optim_interval,
                               decomp_scheme = decomp_scheme, output_pvals = output_pvals, chisq_df = chisq_df)
        end

    else
        error("Assumption keyword is not supported. Please enter null or alt.")
    end

    if profileLL == true
        #= 
        println("Loglik plot: ")
        p = plotLL(y_st, g_st, covar_st, K_st, h2_grid, markerID; 
                   x_lims = x_lims, y_lims = y_lims,
                   prior = [prior_variance, prior_sample_size])

        display(p)
        =# 

        results_profileLL = profile_LL(y_st, g_st, covar_st, K_st, h2_grid, markerID; 
                                      prior = [prior_variance, prior_sample_size], reml = reml);

        return (results, results_profileLL);
    else
        return results
    end

end

###
# scan markers under the null
###

"""
    scan_null(y, g, K, reml, method)

Warning: scan_null is 

Performs genome scan for univariate trait and each of the gene markers, one marker at a time, 
assuming the variance components are the same for all markers.

# Arguments

- y = 1d array of floats consisting of the N observations for a certain quantitative trait (dimension: N*1)
- g = 2d array of floats consisting of all p gene markers (dimension: N*p)
- covar = 2d array of floats consisting of all covariates to adjust for (optional)
- K = 2d array of floats consisting of the genetic relatedness of the N observations (dimension:N*N)

# Keyword arguments

- addIntercept = Boolean flag indicating if an intercept column needs to be added to the design matrix; Default is to add an intercept
- reml = Bool flag indicating if VCs are estimated by REML likelihood; Default is ML
- method = String indicating the matrix factorization method to use; Default is QR.

# Value

A list of output values are returned:
- out00.sigma2 = Float; estimated marginal variance due to random errors (by null lmm)
- out00.h2 = Float; estimated heritability (by null lmm)
- lod = 1d array of floats consisting of the lod scores of this trait and all markers (dimension: p*1)

# Some notes

    This is a subsequent function that does univariate scan. The variance components are estimated once 
    and used for all markers. To be called by the `scan` function when the `method = ` field is passed 
    as `null` (default).

"""
# Note: use of the `scan_null()` is now deprecated because of its computational inefficiency compared to using `scan_perms_lite()`,
#       which performs faster single-trait scan by vectorized operations.
#       The following source code of `scan_null()` may be removed in future releases.

# function scan_null(y::Array{Float64, 2}, g::Array{Float64, 2}, covar::Array{Float64, 2}, K::Array{Float64, 2}, 
#                    prior::Array{Float64, 1}, addIntercept::Bool;
#                    reml::Bool = false, method::String = "qr", optim_interval::Int64 = 1,
#                    decomp_scheme::String = "eigen", 
#                    # option for returning p-values results:
#                    output_pvals::Bool = false, chisq_df::Int64 = 1)

#     # number of markers
#     (n, p) = size(g)

#     num_of_covar = addIntercept ? (size(covar, 2)+1) : size(covar, 2);

#     # rotate data
#     (y0, X0, lambda0) = transform_rotation(y, [covar g], K; 
#                                            addIntercept = addIntercept, decomp_scheme = decomp_scheme)
#     X0_covar = X0[:, 1:num_of_covar];

#     if size(X0_covar, 2) == 1
#         X0_covar = reshape(X0_covar, :, 1);
#     end

#     # fit null lmm
#     out00 = fitlmm(y0, X0_covar, lambda0, prior; reml = reml, method = method, optim_interval = optim_interval)
#     # weights proportional to the variances
#     sqrtw = sqrt.(makeweights(out00.h2, lambda0))
#     # rescale by weights
#     rowMultiply!(y0, sqrtw)
#     rowMultiply!(X0, sqrtw)
#     rowMultiply!(X0_covar, sqrtw);

#     # perform genome scan
#     rss0 = rss(y0, X0_covar; method = method)[1]
#     lod = zeros(p)

#     X = X0[:, 1:(num_of_covar+1)]
#     for i = 1:p
#         X[:, (num_of_covar+1)] = X0[:, num_of_covar+i]
#         rss1 = rss(y0, X; method = method)[1]
#         lod[i] = (-n/2)*(log10(rss1) .- log10(rss0))
#         # lrt = (rss0 - rss1)/out00.sigma2
#         # lod[i] = lrt/(2*log(10))
#     end

#     if output_pvals
#         log10pvals = lod2log10p.(lod, chisq_df);
#         return (sigma2_e = out00.sigma2, h2_null = out00.h2, lod = lod, log10pvals = log10pvals)
#     else
#         return (sigma2_e = out00.sigma2, h2_null = out00.h2, lod = lod)
#     end

# end

## re-estimate variance components under alternative


"""
    scan_alt(y, g, K, reml)

Performs genome scan for univariate trait and each of the gene markers, one marker at a time (one-df test),
assuming the variance components may not be the same for all markers. 

# Arguments

- y = 1d array of floats consisting of the N observations for a certain quantitative trait (dimension: N*1)
- g = 2d array of floats consisting of all p gene markers (dimension: N*p)
- K = 2d array of floats consisting of the genetic relatedness of the N observations (dimension:N*N)

# Keyword arguments

- addIntercept = Boolean flag indicating if an intercept column needs to be added to the design matrix; Default is to add an intercept
- reml = Bool flag indicating if VCs are estimated by REML likelihood; Default is ML
- method = String indicating the matrix factorization method to use; Default is QR.

# Value

A list of output values are returned:
- out00.sigma2 = Float; estimated marginal variance due to random errors (by null lmm)
- out00.h2 = Float; estimated heritability (by null lmm)
- lod = 1d array of floats consisting of the lod scores of this trait and all markers (dimension: p*1)

# Some notes

    This is a subsequent function that does univariate scan. For every scan for each genetic marker, the 
    variance components will need to be re-estimated. To be called by the `scan` function when the `method = ` 
    field is passed as `alt`.

"""
function scan_alt(y::Array{Float64, 2}, g::Array{Float64, 2}, covar::Array{Float64, 2}, K::Array{Float64, 2}, 
                  prior::Array{Float64, 1}, addIntercept::Bool;
                  fixed_effects::Bool = true,
                  reml::Bool = false, method::String = "qr", optim_interval::Int64 = 1,
                  decomp_scheme::String = "eigen",
                  # option for returning p-values results:
                  output_pvals::Bool = false, chisq_df::Int64 = 1)

    # number of markers
    (n, p) = size(g)

    num_of_covar = addIntercept ? (size(covar, 2)+1) : size(covar, 2);

    # rotate data
    (y0, X0, lambda0) = transform_rotation(y, [covar g], K; 
                                           addIntercept = addIntercept,
                                           decomp_scheme = decomp_scheme)
    X0_covar = X0[:, 1:num_of_covar];

    if size(X0_covar, 2) == 1
        X0_covar = reshape(X0_covar, :, 1);
    end

    pve_list = Array{Float64, 1}(undef, p);

    # fit null lmm
    out00 = fitlmm(y0, X0_covar, lambda0, prior; reml = reml, method = method, optim_interval = optim_interval);

    lod = zeros(p); # LOD scores
    b = zeros(p); # Fixed marker effect estimates
    se = zeros(p); # Standard errors of the fixed marker effect estimates

    X = X0[:, 1:(num_of_covar+1)]

    for i = 1:p
        X[:, (num_of_covar+1)] = X0[:, num_of_covar+i]
        
        out11 = fitlmm(y0, X, lambda0, prior; reml = reml, method = method, optim_interval = optim_interval);
       
        # estimate variance components (vc) from the alt. model
        # sqrtw_null = sqrt.(makeweights(out00.h2, lambda0));
        sqrtw_alt = sqrt.(makeweights(out11.h2, lambda0));

        # re-scale both models (null, alt.) and evaluate the ells base on vc from alt. model
        wls_alt = wls(y0, X, sqrtw_alt, prior); # h2 is estimated from alt
        wls_null = wls(y0, X0_covar, sqrtw_alt, prior); # h2 is also estimated from alt

        # Compute LOD score:
        lod[i] = (wls_alt.ell - wls_null.ell)/log(10);
        # Extract and save the estimated marker effect size:
        b[i] = wls_alt.b[num_of_covar+1];
        # Compute and save the standard error of the estimated marker effect size:
        se[i] = calc_std_err(lod[i], b[i]);
        # Extract and save the estimated h2 under alt. model:
        pve_list[i] = out11.h2;
    end

    if output_pvals
        log10pvals = lod2log10p.(lod, chisq_df);
        return (b = b, sigma2_e = out00.sigma2, h2_null = out00.h2, h2_each_marker = pve_list, lod = lod, log10pvals = log10pvals);
    else
        return (b = b, sigma2_e = out00.sigma2, h2_null = out00.h2, h2_each_marker = pve_list, lod = lod);
    end


end

## genome scan with permutations
## no covariates
## one-df tests

"""
    scan(y, g, K, nperm, nprocs, rndseed, reml)

Performs genome scan on a number of resamples from permuting an univariate trait with each genome marker (one-df test). Variance components 
are estimated from the null model and assumed to be the same across markers.

# Arguments

- y = 1d array of floats consisting of the N observations for a certain quantitative trait (dimension: N*1)
- g = 2d array of floats consisting of all p gene markers (dimension: N*p)
- K = 2d array of floats consisting of the genetic relatedness of the N observations (dimension:N*N)

# Keyword arguments

- nperms = Number of permutations requested
- nprocs = Number of concurrent processes
- rndseed = Random seed for perform permutations
- reml = Boolean flag indicating if variance components will be estimated by REML (true) or ML (false)

# Value

- lod = 2d array of floats consisting of the LOD scores for each permuted trait (row) and the markers (columns).

# Some notes

"""
function scan_perms_lite(y::Array{Float64,2}, g::Array{Float64,2}, covar::Array{Float64, 2}, K::Array{Float64,2};
                         prior_variance::Float64 = 1.0, prior_sample_size::Float64 = 0.0, 
                         addIntercept::Bool = true, method::String = "qr", optim_interval::Int64 = 1,
                         nperms::Int64 = 1024, rndseed::Int64 = 0, 
                         reml::Bool = false,
                         decomp_scheme::String = "eigen",
                         # option for returning p-values results:
                         output_pvals::Bool = false, chisq_df::Int64 = 1)


    # check the number of traits as this function only works for permutation testing of univariate trait
    if size(y, 2) != 1
        error("Can only handle one trait.")
    end

    ## Issue: when covar is passed with a vector that is proportional to the column of ones, standard deviation will be 0
    ## it is recommended that the user to standardize the input matrices and then use the prior_variance of 1.
    # sy = colStandardize(y);
    # sg = colStandardize(g);
    # scovar = colStandardize(covar);


    # n - the sample size
    # p - the number of markers
    n = size(g, 1)

    ## Note: estimate once the variance components from the null model and use for all marker scans
    # fit lmm
    (y0, X0, lambda0) = transform_rotation(y, [covar g], K; 
                                           addIntercept = addIntercept,
                                           decomp_scheme = decomp_scheme); # rotation of data
    

    # if the intercept is added, the number of covariates to be regressed out will be one more (the intercept)
    n_covars = addIntercept ? (size(covar, 2)+1) : (size(covar, 2)); 

    (r0, X00, sigma2_e, h2_null) = transform_reweight(y0, X0, lambda0;
                                   n_covars = n_covars,  
                                   prior_a = prior_variance, 
                                   prior_b = prior_sample_size, 
                                   reml = reml, method = method, optim_interval = optim_interval); # reweighting and taking residuals

    b = (X00 \ r0) |> vec; # estimate fixed effects

    # Compute the matrix of pair-wise correlations between permuted copies and markers:
    r0perm = transform_permute(r0; nperms = nperms, rndseed = rndseed, original = true);
    norm_y = mapslices(x -> norm(x), r0perm, dims = 1) |> vec;
    norm_X = mapslices(x -> norm(x), X00, dims = 1) |> vec;
    colDivide!(r0perm, norm_y);
    colDivide!(X00, norm_X);
    L = X00' * r0perm # the matrix of correlations

    threaded_map!(r2lod, L, n); # map elementwise-ly to compute LOD scores

    # LOD scores for the original trait with the markers;
    lod = L[:, 1]; 
    # Standard errors of the estimated marker effects
    se = map((a, b) -> calc_std_err(a, b), lod, b)

    L_perms = L[:, 2:end]; # lod scores for the permuted copies of the original, excluding the lod scores for the original trait

    if output_pvals
        log10pvals = lod2log10p.(lod, chisq_df);
        if nperms == 0 # if no permutation is required, return results only for the input trait
            return (b = b, sigma2_e = sigma2_e, h2_null = h2_null, lod = lod, log10pvals = log10pvals)
        end
        log10Pvals_perms = lod2log10p.(L_perms, chisq_df);
        return (b = b, sigma2_e = sigma2_e, h2_null = h2_null, lod = lod, log10pvals = log10pvals,
                           L_perms = L_perms, log10Pvals_perms = log10Pvals_perms)
    else
        if nperms == 0 # if no permutation is required, return results only for the input trait
            return (b = b, sigma2_e = sigma2_e, h2_null = h2_null, lod = lod)
        end
        return (b = b, sigma2_e = sigma2_e, h2_null = h2_null, lod = lod, L_perms = L_perms)
    end

end

