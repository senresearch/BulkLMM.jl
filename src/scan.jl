###########################################################
# genome scan function; no covariates, two genotype groups
###########################################################

"""
    scan(y, g, K, reml, method)

Performs genome scan for univariate trait and each of the gene markers, one marker at a time (one-df test) 

# Arguments

- y = 1d array of floats consisting of the N observations for a certain quantitative trait (dimension: N*1)
- g = 2d array of floats consisting of all p gene markers (dimension: N*p)
- K = 2d array of floats consisting of the genetic relatedness of the N observations (dimension:N*N)

# Keyword arguments

- reml = Boolean flag indicating how variance component parameters are to be estimated
- method = String indicating whether the variance component parameters are the same across all markers (null) or not (alt); Default is `null`.

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
              prior_a::Float64 = 0.0, prior_b::Float64 = 0.0, addIntercept::Bool = true,
              reml::Bool = false, assumption::String = "null", method::String = "qr")

    if(assumption == "null")
        return scan_null(y, g, K, [prior_a, prior_b], addIntercept; reml = reml, method = method)
    elseif(assumption == "alt")
        return scan_alt(y, g, K, [prior_a, prior_b], addIntercept; reml = reml, method = method)
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
- K = 2d array of floats consisting of the genetic relatedness of the N observations (dimension:N*N)

# Keyword arguments

- reml = Boolean flag indicating how variance component parameters are to be estimated

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


function scan_null(y::Array{Float64, 2}, g::Array{Float64, 2}, K::Array{Float64, 2}, prior::Array{Float64, 1}, addIntercept::Bool;
                   reml::Bool = false, method::String = "qr")

    # number of markers
    (n, p) = size(g)

    # rotate data
    (y0, X0, lambda0) = transform_rotation(y, g, K; addIntercept = addIntercept)

    # fit null lmm
    out00 = fitlmm(y0, reshape(X0[:, 1], :, 1), lambda0, prior; reml = reml, method = method)
    # weights proportional to the variances
    sqrtw = sqrt.(makeweights(out00.h2, lambda0))
    # rescale by weights
    rowMultiply!(y0, sqrtw)
    rowMultiply!(X0, sqrtw)

    # perform genome scan
    rss0 = rss(y0, reshape(X0[:, 1], n, 1); method = method)[1]
    lod = zeros(p)
    X = zeros(n, 2)
    X[:, 1] = X0[:, 1]
    for i = 1:p
        X[:, 2] = X0[:, i+1]
        rss1 = rss(y0, X; method = method)[1]
        lod[i] = (-n/2)*(log10(rss1) .- log10(rss0))
        # lrt = (rss0 - rss1)/out00.sigma2
        # lod[i] = lrt/(2*log(10))
    end

    return (out00.sigma2, out00.h2, lod)

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

- reml = Boolean flag indicating how variance component parameters are to be estimated

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

function scan_alt(y::Array{Float64, 2}, g::Array{Float64, 2}, K::Array{Float64, 2}, prior::Array{Float64, 1}, addIntercept::Bool;
                 reml::Bool = false, method::String = "qr")

    # number of markers
    (n, p) = size(g)

    # rotate data
    (y0, X0, lambda0) = transform_rotation(y, g, K; addIntercept = addIntercept)

    pve_list = Array{Float64, 1}(undef, p);

    X00 = reshape(X0[:, 1], :, 1)
    # fit null lmm
    out00 = fitlmm(y0, X00, lambda0, prior; reml = reml, method = method);

    lod = zeros(p)
    X = zeros(n, 2)
    X[:, 1] = X0[:, 1]
    for i = 1:p
        X[:, 2] = X0[:, i+1]
        
        out11 = fitlmm(y0, X, lambda0, prior; reml = reml, method = method);

        pve_list[i] = out11.h2;

        lod[i] = (out11.ell - out00.ell)/log(10)
    end

    return (out00.sigma2, out00.h2, pve_list, lod)

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
              prior_a::Float64 = 0.0, prior_b::Float64 = 0.0, addIntercept::Bool = true, method::String = "qr",
              nperms::Int64 = 1024, rndseed::Int64 = 0, 
              reml::Bool = false, original::Bool = true)

    # check the number of traits as this function only works for permutation testing of univariate trait
    if(size(y, 2) != 1)
        error("Can only handle one trait.")
    end

    sy = colStandardize(y);
    sg = colStandardize(g);

    # n - the sample size
    # p - the number of markers
    (n, p) = size(g)

    ## Note: estimate once the variance components from the null model and use for all marker scans
    # fit lmm
    (y0, X0, lambda0) = transform_rotation(sy, sg, K; addIntercept = addIntercept); # rotation of data
    (r0, X00, sigma2) = transform_reweight(y0, X0, lambda0; prior_a = prior_a, prior_b = prior_b, reml = reml, method = method); # reweighting and taking residuals

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
    # (For theoretical derivation of the results, see notebook)
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


####################################################################################################
######################################### Distributed Processes ####################################
####################################################################################################

#=

Functions hierarchy:

scan_perms_distributed(original data, 
                       total number of permutations required, 
                       option of which algorithm to be used, ...):

    > distribute_by_blocks(number of blocks, ...): 
        (parallel_helpers.jl)   
        - createBlocks(number of blocks (or the size of each block?), ...): create blocks with the same sizes
        - calcLODs_block(...): calculate LOD scores for all markers in the given block 
    
    > distribute_by_nperms():
        (parallel_helpers.jl)   
        - calcLODs_perms(rndseed, ...): calculate LOD scores for a (subset of the total) number of permutations


=#

function scan_perms_distributed(y::Array{Float64,2}, g::Array{Float64,2}, K::Array{Float64,2};
                                reml::Bool = false,
                                nperms::Int64 = 1024, rndseed::Int64 = 0, original::Bool = true,
                                # (options for blocks, nperms distribution methods...)
                                option::String = "by blocks", nblocks::Int64 = 1, ncopies::Int64 = 1, 
                                nprocs::Int64 = 0)

    ## Note: estimate once the variance components from the null model and use for all marker scans
    # fit lmm
    (y0, X0, lambda0) = transform_rotation(y, g, K; addIntercept = addIntercept); # rotation of data
    (r0, X00) = transform_reweight(y0, X0, lambda0; prior_a = prior_a, prior_b = prior_b, reml = reml, method = method); # reweighting and taking residuals

    # If no permutation testing is required, move forward to process the single original vector
    if nperms == 0
        if original == false
            throw(error("If no permutation testing is required, input value of `original` has to be `true`."));
        end

        r0perm = r0;
    else
        r0perm = transform_permute(r0; nperms = nperms, rndseed = rndseed, original = original);
    end

    if option == "by blocks"
        results = distribute_by_blocks(r0perm, X00, nblocks);
    elseif option == "by nperms"
        results = distribute_by_nperms(r0, X00, nperms, ncopies, original);
    else
        throw(error("Option unsupported."))
    end

    return results

end

#= scan(y, g, K; Control type arg)

    Control type args:

        ScanType: control Type input 
            > Serial(nperms allowed to be 0 indicating no permutations)
            > Distributed(nperms, nblocks, type of algorithm)


=#

#= 
Next: 
    - multiple threads;
    - tests, plus comparing with GEMMA performances;
    - adjust for additional covariates (X), other than G;
        > first scan(y_i, X, G, K)
    - multiple traits...
    - profile

Future: 
    - single precision (Union type to control precision type: Float64, Float32, ...)

=#
