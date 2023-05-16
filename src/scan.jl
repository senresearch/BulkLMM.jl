###########################################################
# Genome scan functions for a single trait plus permutation testing:
# allow modeling additional covariates, two genotype groups
###########################################################
"""
    scan(y, g, K, reml, method)

Performs genome scan for univariate trait and each of the gene markers, one marker at a time (one-df test) 

# Arguments

- y = 1d array of floats consisting of the N observations for a certain quantitative trait (dimension: N*1)
- g = 2d array of floats consisting of all p gene markers (dimension: N*p)
- K = 2d array of floats consisting of the genetic relatedness of the N observations (dimension:N*N)
- prior_a = a float of prior distribution parameter
- prior_b = a float of prior distribution parameter

# Keyword arguments

- addIntercept = Boolean flag indicating if an intercept column needs to be added to the design matrix; Default is to add an intercept
- reml = Bool flag indicating if VCs are estimated by REML likelihood; Default is ML
- assumption = String indicating whether the variance component parameters are the same across all markers (null) or not (alt); Default is `null`
- method = String indicating the matrix factorization method to use; Default is QR.

# Value

A list of output values are returned:
- out00.sigma2 = Float; estimated marginal variance due to random errors (by null lmm)
- out00.h2 = Float; estimated heritability (by null lmm)
- lod = 1d array of floats consisting of the lod scores of this trait and all markers (dimension: p*1)

# Some notes

    This function calls either `scan_null` or `scan_alt` depending on the input passed as `method`.
    Output data structure might need some revisions.

"""
function scan(y::Array{Float64,2}, g::Array{Float64,2}, K::Array{Float64,2};
              # weighted environmental variances:
              weights::Union{Missing, Array{Float64, 1}} = missing,
              # regularization options:
              prior_variance::Float64 = 0.0, prior_sample_size::Float64 = 0.0, addIntercept::Bool = true,
              # vc estimation options:
              reml::Bool = false, assumption::String = "null", method::String = "qr", optim_interval::Int64 = 1,
              # permutation testing options:
              permutation_test::Bool = false, nperms::Int64 = 1024, rndseed::Int64 = 0, original::Bool = true,
              # option for inspecting h2 estimation process:
              plot_loglik::Bool = false, markerID::Int = 0, h2_grid::Array{Float64, 1} = Array{Float64, 1}(undef, 1),
              x_lims::Array{Float64, 1} = [0.0, 1.0], y_lims::Array{Float64, 1} = [-100.0, 100.0]
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
                permutation_test = permutation_test, nperms = nperms, rndseed = rndseed, original = original,
                plot_loglik = plot_loglik, markerID = markerID, h2_grid = h2_grid,
                x_lims = x_lims, y_lims = y_lims)
end

function scan(y::Array{Float64,2}, g::Array{Float64,2}, covar::Array{Float64, 2}, K::Array{Float64,2};
              # weighted environmental variances:
              weights::Union{Missing, Array{Float64, 1}} = missing,
              # regularization options:
              prior_variance::Float64 = 0.0, prior_sample_size::Float64 = 0.0, addIntercept::Bool = true,
              # vc estimation options:
              reml::Bool = false, assumption::String = "null", method::String = "qr", optim_interval::Int64 = 1,
              # permutation testing options:
              permutation_test::Bool = false, nperms::Int64 = 1024, rndseed::Int64 = 0, original::Bool = true,
              # option for inspecting h2 estimation process:
              plot_loglik::Bool = false, markerID::Int = 0, h2_grid::Array{Float64, 1} = Array{Float64, 1}(undef, 1),
              x_lims::Array{Float64, 1} = [0.0, 1.0], y_lims::Array{Float64, 1} = [-100.0, 100.0]
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
                    permutation_test = permutation_test, nperms = nperms, rndseed = rndseed, original = original,
                    plot_loglik = plot_loglik, markerID = markerID, h2_grid = h2_grid,
                    x_lims = x_lims, y_lims = y_lims)
    else
        y_st = y;
        g_st = g;
        covar_st = covar;
        K_st = K;
    end

    if assumption == "null"
        if permutation_test == true
            results = scan_perms_lite(y_st, g_st, covar_st, K_st; prior_variance = prior_variance, prior_sample_size = prior_sample_size,
                                      addIntercept = addIntercept, reml = reml, method = method, optim_interval = optim_interval,
                                      nperms = nperms, rndseed = rndseed, original = original);
        else
            results = scan_null(y_st, g_st, covar_st, K_st, [prior_variance, prior_sample_size], addIntercept; 
                                reml = reml, method = method, optim_interval = optim_interval);
        end 
    elseif assumption == "alt"
        if permutation_test == true
            error("Permutation test option currently is not supported for the alternative assumption.");
        else
            results = scan_alt(y_st, g_st, covar_st, K_st, [prior_variance, prior_sample_size], addIntercept; 
                               reml = reml, method = method, optim_interval = optim_interval)
        end

    else
        error("Assumption keyword is not supported. Please enter null or alt.")
    end

    # return results

    if plot_loglik == true
        println("Loglik plot: ")
        p = plotLL(y_st, g_st, covar_st, K_st, h2_grid, markerID; 
                   x_lims = x_lims, y_lims = y_lims,
                   prior = [prior_variance, prior_sample_size])
        # plot(p)
        display(p)
        return results
    else
        return results
    end

end

###
# scan markers under the null
###

"""
    scan_null(y, g, K, reml, method)

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
function scan_null(y::Array{Float64, 2}, g::Array{Float64, 2}, covar::Array{Float64, 2}, K::Array{Float64, 2}, 
                   prior::Array{Float64, 1}, addIntercept::Bool;
                   reml::Bool = false, method::String = "qr", optim_interval::Int64 = 1)

    # number of markers
    (n, p) = size(g)

    num_of_covar = addIntercept ? (size(covar, 2)+1) : size(covar, 2);

    # rotate data
    (y0, X0, lambda0) = transform_rotation(y, [covar g], K; addIntercept = addIntercept)
    X0_covar = X0[:, 1:num_of_covar];

    if size(X0_covar, 2) == 1
        X0_covar = reshape(X0_covar, :, 1);
    end

    # fit null lmm
    out00 = fitlmm(y0, X0_covar, lambda0, prior; reml = reml, method = method, optim_interval = optim_interval)
    # weights proportional to the variances
    sqrtw = sqrt.(makeweights(out00.h2, lambda0))
    # rescale by weights
    rowMultiply!(y0, sqrtw)
    rowMultiply!(X0, sqrtw)
    rowMultiply!(X0_covar, sqrtw);

    # perform genome scan
    rss0 = rss(y0, X0_covar; method = method)[1]
    lod = zeros(p)

    X = X0[:, 1:(num_of_covar+1)]
    for i = 1:p
        X[:, (num_of_covar+1)] = X0[:, num_of_covar+i]
        rss1 = rss(y0, X; method = method)[1]
        lod[i] = (-n/2)*(log10(rss1) .- log10(rss0))
        # lrt = (rss0 - rss1)/out00.sigma2
        # lod[i] = lrt/(2*log(10))
    end

    return (sigma2_e = out00.sigma2, h2_null = out00.h2, lod = lod)

end

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
                  reml::Bool = false, method::String = "qr", optim_interval::Int64 = 1)

    # number of markers
    (n, p) = size(g)

    num_of_covar = addIntercept ? (size(covar, 2)+1) : size(covar, 2);

    # rotate data
    (y0, X0, lambda0) = transform_rotation(y, [covar g], K; addIntercept = addIntercept)
    X0_covar = X0[:, 1:num_of_covar];

    if size(X0_covar, 2) == 1
        X0_covar = reshape(X0_covar, :, 1);
    end

    pve_list = Array{Float64, 1}(undef, p);

    # fit null lmm
    out00 = fitlmm(y0, X0_covar, lambda0, prior; reml = reml, method = method, optim_interval = optim_interval);

    lod = zeros(p);

    X = X0[:, 1:(num_of_covar+1)]

    for i = 1:p
        X[:, (num_of_covar+1)] = X0[:, num_of_covar+i]
        
        out11 = fitlmm(y0, X, lambda0, prior; reml = reml, method = method, optim_interval = optim_interval);
       
        # estimate variance components (vc) from the alt. model
        sqrtw_alt = sqrt.(makeweights(out11.h2, lambda0));

        # re-scale both models (null, alt.) and evaluate the ells base on vc from alt. model
        wls_alt = wls(y0, X, sqrtw_alt, prior);
        wls_null = wls(y0, X0_covar, sqrtw_alt, prior);
        lod[i] = (wls_alt.ell - wls_null.ell)/log(10);

        pve_list[i] = out11.h2;
    end

    return (sigma2_e = out00.sigma2, h2_null = out00.h2, h2_each_marker = pve_list, lod = lod);


end

## genome scan with permutations
## no covariates
## one-df tests
## with parallelization

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
function scan_perms(y::Array{Float64,2}, g::Array{Float64,2}, K::Array{Float64,2};
              prior_variance::Float64 = 1.0, prior_sample_size::Float64 = 0.0, addIntercept::Bool = true, method::String = "qr",
              nperms::Int64 = 1024, rndseed::Int64 = 0, 
              reml::Bool = false, original::Bool = true, optim_interval::Int64 = 1)

    # check the number of traits as this function only works for permutation testing of univariate trait
    if(size(y, 2) != 1)
        error("Can only handle one trait.")
    end

    # sy = colStandardize(y);
    # sg = colStandardize(g);

    # n - the sample size
    # p - the number of markers
    (n, p) = size(g)

    ## Note: estimate once the variance components from the null model and use for all marker scans
    # fit lmm
    (y0, X0, lambda0) = transform_rotation(y, g, K; addIntercept = addIntercept); # rotation of data
    (r0, X00) = transform_reweight(y0, X0, lambda0; prior_a = prior_variance, prior_b = prior_sample_size, 
                                                    reml = reml, method = method, optim_interval = optim_interval); # reweighting and taking residuals

    # If no permutation testing is required, move forward to process the single original vector
    if nperms == 0

        if original == false
            throw(error("If no permutation testing is required, input value of `original` has to be `true`."));
        end
    
        r0perm = r0;
    else
        r0perm = transform_permute(r0; nperms = nperms, rndseed = rndseed, original = original);
    end


    ## Null RSS:
    # By null hypothesis, mean is 0. RSS just becomes the sum of squares of the residuals (r0perm's)
    # (For theoretical derivation of the results, see supplement results)
    rss0 = sum(r0perm[:, 1].^2) # a scalar; bc rss0 for every permuted trait is the same under the null (zero mean);
    
    ## make array to hold Alternative RSS's for each permutated trait
    if original
        rss1 = Array{Float64, 2}(undef, nperms+1, p)
    else
        rss1 = Array{Float64, 2}(undef, nperms, p)
    end
    
    ## loop over markers
    for i = 1:p
        ## alternative rss
        @inbounds rss1[:, i] = rss(r0perm, @view X00[:, i]);
        
    end

    
    lod = (-n/2)*(log10.(rss1) .- log10(rss0))

    return lod

end


function scan_perms_lite(y::Array{Float64,2}, g::Array{Float64,2}, covar::Array{Float64, 2}, K::Array{Float64,2};
                         prior_variance::Float64 = 1.0, prior_sample_size::Float64 = 0.0, 
                         addIntercept::Bool = true, method::String = "qr", optim_interval::Int64 = 1,
                         nperms::Int64 = 1024, rndseed::Int64 = 0, 
                         reml::Bool = false, original::Bool = true)


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
    (y0, X0, lambda0) = transform_rotation(y, [covar g], K; addIntercept = addIntercept); # rotation of data
    
    (r0, X00) = transform_reweight(y0, X0, lambda0;
                                   n_covars = size(covar, 2),  
                                   prior_a = prior_variance, 
                                   prior_b = prior_sample_size, 
                                   reml = reml, method = method, optim_interval = optim_interval); # reweighting and taking residuals

    # If no permutation testing is required, move forward to process the single original vector
    if nperms == 0

        if original == false
            throw(error("If no permutation testing is required, input value of `original` has to be `true`."));
        end

        r0perm = r0;
    else
        r0perm = transform_permute(r0; nperms = nperms, rndseed = rndseed, original = original);
    end

    norm_y = mapslices(x -> norm(x), r0perm, dims = 1) |> vec;

    norm_X = mapslices(x -> norm(x), X00, dims = 1) |> vec;


    colDivide!(r0perm, norm_y);
    colDivide!(X00, norm_X);

    lods = X00' * r0perm
    threaded_map!(r2lod, lods, n);

    return lods

end

