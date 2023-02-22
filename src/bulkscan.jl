###########################################################
# Genome scan functions for multiple traits:
# allow modeling additional covariates, two genotype groups
###########################################################

###########################################################
## (1) Chunk methods:
## idea is to use multithreaded processes to run genome scans sequentially
## on traits that are inside a block that is a subset of total number of traits;
## results should be exact as given by running scan_null() on each trait.
###########################################################
"""
bulkscan_null(Y, G, K, nb; reml = true)

Calculates the LOD scores for all pairs of traits and markers, by a (multi-threaded) loop over blocks of traits and the LiteQTL-type of approach

# Arguments
- Y = 2d Array of Float; matrix of one trait or multiple traits
- G = 2d Array of Float; matrix of genotype probabilities
- K = 2d Array of Float; kinship matrix
- nb = Int; number of blocks of traits required; ideally to be the same number of threads used for parallelization 

# Value

- LOD = 2d Array of Float; LOD scores for all pairs of traits and markers

# Notes:

"""
function bulkscan_null(Y::Array{Float64, 2}, G::Array{Float64, 2}, K::Array{Float64, 2};
                   nb::Int64 = Threads.nthreads(), 
                   nt_blas::Int64 = 1, prior_variance = 1.0, prior_sample_size = 0.0,
                   reml::Bool = false)


    m = size(Y, 2);
    p = size(G, 2);

    BLAS.set_num_threads(nt_blas);

    # rotate data
    (Y0, X0, lambda0) = transform_rotation(Y, G, K);
    X0_intercept = reshape(X0[:, 1], :, 1);
    X0_covar = X0[:, 2:end];

    # distribute the `m` traits equally to every block
    (len, rem) = divrem(m, nb);

    results = Array{Array{Float64, 2}, 1}(undef, nb);
    h2_null_list = zeros(m);

    Threads.@threads for t = 1:nb # so the N blocks will share the (nthreads - N) BLAS threads

        lods_currBlock = Array{Float64, 2}(undef, p, len);

        # process every trait in the block by a @simd loop 
        @simd for i = 1:len
            j = i+(t-1)*len;

            outputs = univar_liteqtl(Y0[:, j], X0_intercept, X0_covar, lambda0; 
                                     prior_variance = prior_variance, prior_sample_size = prior_sample_size,
                                     reml = reml);
            @inbounds lods_currBlock[:, i] = outputs.R;
            @inbounds h2_null_list[j] = outputs.h2 
        end

        results[t] = lods_currBlock;

    end

    LODs_all = reduce(hcat, results);

    # if no remainder as the result of blocking, no remaining traits need to be scanned
    if rem == 0
        return LODs_all
    end
        
    # else, process up the remaining traits
    lods_remBlock = Array{Float64, 2}(undef, p, rem);

    for i in 1:rem

        j = m-rem+i;

        outputs = univar_liteqtl(Y0[:, j], X0_intercept, X0_covar, lambda0;
                                 prior_variance = prior_variance, prior_sample_size = prior_sample_size,
                                 reml = reml);
        
        lods_remBlock[:, i] = outputs.R;
        h2_null_list[j] = outputs.h2;

    end

    LODs_all = hcat(LODs_all, lods_remBlock);

    return (L = LODs_all, h2_null_list = h2_null_list)

end
### Modeling covariates version
function bulkscan_null(Y::Array{Float64, 2}, G::Array{Float64, 2}, 
                       Covar::Array{Float64, 2}, K::Array{Float64, 2};
                       nb::Int64 = Threads.nthreads,
                       nt_blas::Int64 = 1, prior_variance = 1.0, prior_sample_size = 0.0,
                       reml::Bool = false)


    m = size(Y, 2);
    p = size(G, 2);
    num_of_covar = size(Covar, 2)+1;

    BLAS.set_num_threads(nt_blas);

    # rotate data
    (Y0, X0, lambda0) = transform_rotation(Y, [Covar G], K);


    X0_intercept = X0[:, 1:num_of_covar];
    X0_covar = X0[:, (num_of_covar+1):end];

    # distribute the `m` traits equally to every block
    (len, rem) = divrem(m, nb);

    results = Array{Array{Float64, 2}, 1}(undef, nb);
    h2_null_list = zeros(m);

    Threads.@threads for t = 1:nb # so the N blocks will share the (nthreads - N) BLAS threads

    lods_currBlock = Array{Float64, 2}(undef, p, len);

    # process every trait in the block by a @simd loop 
    @simd for i = 1:len
        j = i+(t-1)*len;

        outputs = univar_liteqtl(Y0[:, j], X0_intercept, X0_covar, lambda0; 
                                 prior_variance = prior_variance, prior_sample_size = prior_sample_size,
                                 reml = reml);

        @inbounds lods_currBlock[:, i] = outputs.R;
        @inbounds h2_null_list[j] = outputs.h2
    end

        results[t] = lods_currBlock;

    end

    LODs_all = reduce(hcat, results);

    # if no remainder as the result of blocking, no remaining traits need to be scanned
    if rem == 0
        return LODs_all
    end

    # else, process up the remaining traits
    lods_remBlock = Array{Float64, 2}(undef, p, rem);

    for i in 1:rem

        j = m-rem+i;

        outputs = univar_liteqtl(Y0[:, j], X0_intercept, X0_covar, lambda0;
                                 prior_variance = prior_variance, prior_sample_size = prior_sample_size,
                                 reml = reml);
        
        lods_remBlock[:, i] = outputs.R;
        h2_null_list[j] = outputs.h2;

    end

    LODs_all = hcat(LODs_all, lods_remBlock);

    return (L = LODs_all, h2_null_list = h2_null_list)
end

###########################################################
## (2) Grid approximation methods:
## idea is to approximate the exact MLE/REML estimate of h2 using 
## a discrete grid of h2; results should be viewed as an approximation 
## of scan_null() results for each trait.
###########################################################
function bulkscan_null_grid(Y::Array{Float64, 2}, G::Array{Float64, 2}, K::Array{Float64, 2}, grid_list::Array{Float64, 1})

    m = size(Y, 2);
    p = size(G, 2);

    results_by_bin = gridscan_by_bin(Y, G, K, grid_list);
    LOD_grid = reorder_results(results_by_bin.idxs_by_bin, results_by_bin.LODs_by_bin, m, p);

    est_h2_per_y = get_h2_distribution(results_by_bin.h2_taken, results_by_bin.idxs_by_bin);


    return (L = LOD_grid, h2_null_list = est_h2_per_y)

end

function bulkscan_null_grid(Y::Array{Float64, 2}, G::Array{Float64, 2}, Covar::Array{Float64, 2}, 
                       K::Array{Float64, 2}, grid_list::Array{Float64, 1})

    m = size(Y, 2);
    p = size(G, 2);

    results_by_bin = gridscan_by_bin(Y, G, Covar, K, grid_list);
    LOD_grid = reorder_results(results_by_bin.idxs_by_bin, results_by_bin.LODs_by_bin, m, p);

    est_h2_per_y = get_h2_distribution(results_by_bin.h2_taken, results_by_bin.idxs_by_bin);


    return (L = LOD_grid, h2_null_list = est_h2_per_y)

end

function get_h2_distribution(h2_list::Array{Float64, 1}, idxs_by_bin::Vector{Vector{Bool}})

    h2_distr = zeros(size(idxs_by_bin[1], 1));
    
    for i in 1:length(h2_list)
        h2_distr[idxs_by_bin[i]] .= h2_list[i]
    end
    
    return h2_distr;
    
end

###########################################################
## (3) Grid + element-wise maximization approximation methods:
## idea is to approximate the exact MLE/REML estimate of h2 (independently for each marker)
## using a discrete grid of h2; results should be viewed as an approximation 
## of scan_alt() results for each trait.
###########################################################
"""
bulkscan_alt_grid(Y, G, K, hsq_list)

Calculates LOD scores for all pairs of traits and markers for each heritability in the supplied list, and returns the 
    maximal LOD scores for each pair among all calculated ones

# Arguments
- Y = 2d Array of Float; traits 
- G = 2d Array of Float; genotype probabilities
- K = 2d Array of Float; kinship matrix
- hsq_list = 1d array of Float; the list of heritabilities requested to choose from

# Value

- maxL = 2d Array of Float; matrix of LOD scores for all traits and markers estimated

# Notes:

Maximal LOD scores are taken independently for each pair of trait and marker; while the number of candidated hsq's are finite,
    doing such maximization is like performing maximum-likelihood approach on discrete values for the heritability parameter;
    this is a shortcut of doing the exact scan_alt() independently for each trait and each marker.

"""
function bulkscan_alt_grid(Y::Array{Float64, 2}, G::Array{Float64, 2}, K::Array{Float64, 2}, hsq_list::Array{Float64, 1})

    (Y0, X0, lambda0) = transform_rotation(Y, G, K);

    maxL = weighted_liteqtl(Y0, X0, lambda0, hsq_list[1]);

    for hsq in hsq_list[2:end]

        currL = weighted_liteqtl(Y0, X0, lambda0, hsq);
        tmax!(maxL, currL);

    end

    return maxL

end

function bulkscan_alt_grid(Y::Array{Float64, 2}, G::Array{Float64, 2}, 
                      Covar::Array{Float64, 2}, K::Array{Float64, 2}, hsq_list::Array{Float64, 1})

    (Y0, X0, lambda0) = transform_rotation(Y, [Covar G], K);

    num_of_covar = size(Covar, 2)+1;

    maxL = weighted_liteqtl(Y0, X0, lambda0, hsq_list[1]; num_of_covar = num_of_covar);

    for hsq in hsq_list[2:end]

        currL = weighted_liteqtl(Y0, X0, lambda0, hsq; num_of_covar = num_of_covar);
        tmax!(maxL, currL);

    end

    return maxL

end

