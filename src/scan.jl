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
              reml::Bool = false, method::String = "null")
    if(method == "null")
        return scan_null(y, g, K; reml = reml)
    elseif(method == "alt")
        return scan_alt(y, g, K; reml = reml)
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


function scan_null(y::Array{Float64,2}, g::Array{Float64,2}, K::Array{Float64,2};
                   reml::Bool = false)

    # number of markers
    (n, m) = size(g)
    # make intercept
    intercept = ones(n, 1)
    # rotate data
    (y0, X0, lambda0) = rotateData(y,[intercept g],K)
    # fit null lmm
    out00 = fitlmm(y0, reshape(X0[:, 1], :, 1), lambda0; reml = reml)
    # weights proportional to the variances
    sqrtw = sqrt.(makeweights(out00.h2, lambda0))
    # rescale by weights
    rowMultiply!(y0, sqrtw)
    rowMultiply!(X0, sqrtw)

    # perform genome scan
    out0 = rss(y0, reshape(X0[:,1], n, 1))
    lod = zeros(m)
    X = zeros(n,2)
    X[:, 1] = X0[:, 1]
    for i = 1:m
        X[:, 2] = X0[:, i+1]
        out1 = rss(y0, X)
        lod[i] = (n/2)*(log10(out0[1]) - log10(out1[1]))
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

function scan_alt(y::Array{Float64,2},g::Array{Float64,2}, K::Array{Float64,2}; 
                  reml::Bool = false)

    # number of markers
    (n, m) = size(g)
    # make intercept
    intercept = ones(n, 1)
    # rotate data
    (y0, X0, lambda0) = rotateData(y, [intercept g], K)

    X00 = reshape(X0[:, 1], :, 1)
    # fit null lmm
    out00 = fitlmm(y0, X00, lambda0; reml = reml);


    lod = zeros(m)
    X = zeros(n, 2)
    X[:,1] = X0[:, 1]
    for i = 1:m
        X[:, 2] = X0[:, i+1]
        out11 = fitlmm(y0, X, lambda0; reml = reml)
        lod[i] = (out11.ell - out00.ell)/log(10)
    end

    return (out00.sigma2, out00.h2, lod)

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
              nperms::Int64 = 1024, rndseed::Int64 = 0, 
              reml::Bool = false, original::Bool = true)

    # check the number of traits as this function only works for permutation testing of univariate trait
    if(size(y, 2) != 1)
        error("Can only handle one trait.")
    end

    # n - the sample size
    # p - the number of markers
    (n, p) = size(g)

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
    rowMultiply!(r0, sqrtw);
    rowMultiply!(X0, sqrtw);

    
    # after re-weighting X, calling resid on re-weighted X is the same as doing wls on the X after rotation.
    X00 = resid(X0[:, 2:end], reshape(X0[:, 1], :, 1)) # consider not using sub-array, consider @view; in-place changes

    ## random permutations; the first column is the original trait (after transformation)
    rng = MersenneTwister(rndseed);
    ## permute r0 (which is an iid, standard normal distributed N-vector under the null)
    r0perm = shuffleVector(rng, r0[:, 1], nperms; original = original)

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


function scan_perms_distributed(y::Array{Float64,2}, g::Array{Float64,2}, K::Array{Float64,2};
                                reml::Bool = false,
                                nperms::Int64 = 1024, rndseed::Int64 = 0, original::Bool = true,
                                # (options for blocks, nperms distribution methods...)
                                option::String = "by blocks", nblocks::Int64 = 1, ncopies::Int64 = 1, 
                                nprocs::Int64 = 0)

    # addprocs(nprocs);
    # @everywhere using ParallelDataTransfer
    # sendto(workers(), y = y);
    #sendto(workers(), g = g);
    #sendto(workers(), K = K);
    #sendto(workers(), reml = reml);

    #@everywhere begin
        #include("../src/parallel_helpers.jl"); # for now; needs to be revised later
        #(y0, X0, lambda0) = transform1(y, g, K); # rotation of data
        # (r0, X00) = transform2(y0, X0, lambda0; reml = reml); # reweighting and taking residuals
    
    #end

    (y0, X0, lambda0) = transform1(y, g, K); # rotation of data
    (r0, X00) = transform2(y0, X0, lambda0; reml = reml); # reweighting and taking residuals
    r0perm = transform3(r0; nperms = nperms, rndseed = rndseed, original = original);

    if option == "by blocks"
        # @everywhere r0perm = transform3(r0; nperms = nperms, rndseed = rndseed, original = original); # permutations
        results = distribute_by_blocks(r0perm, X00, nblocks);
    elseif option == "by nperms"
        results = distribute_by_nperms(r0, X00, nperms, ncopies, original);
    else
        throw(error("Option unsupported."))
    end

    return results

end

# Inputs: r0perm, X00, number of blocks required
# Outputs: a matrix which is the hcat of LOD scores of all the blocks
# calculate the results of every block distributedly
function distribute_by_blocks(r0perm::Array{Float64, 2}, X00::Array{Float64, 2}, nblocks::Int64)
    # Does distributed processes of calculations of LOD scores for markers in each block

    p = size(X00, 2);

    ## (Create blocks...)
    blocks = createBlocks2(p, nblocks);

    LODs_blocks = pmap(x -> calcLODs_block(r0perm, X00, x), blocks);
    results = reduce(hcat, LODs_blocks);

    return results

end

function distribute_by_nperms(r0::Array{Float64, 2}, X00::Array{Float64, 2}, 
    nperms::Int64, ncopies::Int64, original::Bool)

    (n, m) = size(X00);
    seeds_list = sample((1:(10*ncopies)), ncopies; replace = false);

    # Does distributed process of the number of permutations
    each_nperms = Int(ceil(nperms/ncopies));
    LODs_nperms = pmap(x -> calcLODs_perms(r0, X00, each_nperms, x), seeds_list);
    results = reduce(vcat, LODs_nperms);

    # If we want the results for the original trait vector, calculate LODs separately from 
    # the distributed processes.
    if original == true
        rss0 = sum(r0[:, 1].^2);
        rss1 = Array{Float64, 2}(undef, 1, m);
        
        for i = 1:m

            ## alternative rss for the original trait vector
            rss1[:, i] = rss(r0, @view X00[:, i]);
            
        end

        original_LODs = (-n/2)*(log10.(rss1) .- log10(rss0));
        results = vcat(original_LODs, results);
    end

    return results

end

