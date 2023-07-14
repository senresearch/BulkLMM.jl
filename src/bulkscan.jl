###########################################################
# Genome scan functions for multiple traits:
# allow modeling additional covariates, two genotype groups
###########################################################
"""
bulkscan(Y, G, K; optional inputs)
bulkscan(Y, G, Z, K; optional inputs) - if modeling additional covariates Z

Perform genome scan for multiple univariate traits and a set of genome markers

# Required Inputs
- `Y::Array{Float64, 2}`: Matrix of multiple (m) traits; each column is a trait (dimension: N*m)
- `G::Array{Float64, 2}`: Matrix of genotype probabilities at p tested markers (dimension: N*p)
- `K::Array{Float64, 2}`: Genetic relatedness matrix of the N subjects (dimension:N*N) 

# Optional Inputs

## Essential Inputs:
- `addIntercept::Bool`: Option to add an intercept column to the design matrix (default: true)
- `reml::Bool`: Option of the scheme for estimating variance components (by REML or ML; default: false)
- `method::String`: Keyword argument indicating which multi-trait scan method will be used; currently supported 
    options: "null-grid" (fastest, grid-search approximated Null-LMM), "null-exact" (Null-LMM), and "alt-grid" 
    (grid-search approximated Exact-LMM)

## Modeling Additional Covariates:   
- `Z::AbstractArray{Float64, 2}`: Matrix of additional non-genetic covariates (should be independent to tested 
    markers)

## Different Optional Inputs by the Method Chosen:
    For the multiple-trait scans, the user may choose to apply one of the three methods, depending on
    the need for gaining more precision or waiting for shorter time. The allowed inputs differ by the choice:
    
### Grid-search approximation method: "null-grid" (default method) and "alt-grid"
- `h2_grid::Array{Float64, 1}`: a grid of h2-values in [0, 1) where the optimization of profile likelihood 
    is performed; the finer the grid the better precision but longer wait-time. (default: 0.0:0.10:0.90, 10 values)

### Null-LMM through exact optimization (the Brent's method), multi-threaded: "null-exact"
- `nb::Int64`: The number of sub-groups of the total number of traits; each group is processed independently and 
    the processes are parallelized (default: the number of threads of the current Julia session)
- `nt_blas::Int64`: The number of threads BLAS library will be using (default: 1)

## Permutation Testing:
    Currently permutation testing is only supported for single-trait scans.

## Structure of Weighted Residual Variances:
- `weights::Array{Float64, 1}`: Optional weights for modeling unequal, weighted structure of the residual variances 
    of the trait (default: Missing, i.e. equal residual variances)

## Numerical Techniques - for stabilizing the heritability optimization step
- `optim_interval::Int64`: The number of sub-divided regions of the interval [0, 1) of heritability to perform each 
    numerical optimization scheme (the Brent's method) on (default: 1, i.e. the entire interval)
- `prior_variance::Float64`: Scale parameter of the prior Scaled Inv-Chisq distributed residual variances (default: 0)
- `prior_sample_size::Float64`: Degree of freedom parameter of the prior Scaled Inv-Chisq distributed residual 
    variances (default: 0)

# Returned Values:

The output of the single-trait scan function is an object. Depending on the user inputs and options, the fields of
    the output object will differ. For example, for the returned output named as `MT_out` as "multiple traits 
    outputs":

## Null-LMM ("null-grid", "null-exact"):
- `MT_out.h2_null_list::Array{Float64, 1}`: a list of h2_null estimated for each trait
- `MT_out.L::Array{Float64, 2}`: 2-dimensional array (dimension: p*m) consisting of the LOD scores for all input traits; each column 
    contains the LOD scores for one trait

## Exact-LMM ("alt-grid"):
- `MT_out.h2_panel::Array{Float64, 2}`: 2-dimensional array (dimension: p*m) h2 estimated for each marker and 
    each trait; each column contains the h2 estimated for each marker for one trait
- `MT_out.L::Array{Float64, 2}`: 2-dimensional array (dimension: p*m) consisting of the LOD scores for all input traits; each column 
    contains the LOD scores for one trait


"""
function bulkscan(Y::Array{Float64, 2}, G::Array{Float64, 2}, K::Array{Float64, 2};
                  method::String = "null-grid", h2_grid::Array{Float64, 1} = collect(0.0:0.1:0.9),
                  nb::Int64 = Threads.nthreads(), 
                  nt_blas::Int64 = 1, 
                  weights::Union{Missing, Array{Float64, 1}} = missing,
                  prior_variance::Float64 = 1.0, prior_sample_size::Float64 = 0.0,
                  reml::Bool = false, optim_interval::Int64 = 1)

    n = size(Y, 1);

    # when no covariates are added, make the intercept as the only covariate
    intercept = ones(n, 1);

    return bulkscan(Y, G, intercept, K;
                    method = method, 
                    h2_grid = h2_grid,
                    nb = nb, nt_blas = nt_blas, 
                    # key step: avoid adding the intercept twice
                    addIntercept = false, 
                    weights = weights,
                    prior_variance = prior_variance, prior_sample_size = prior_sample_size,
                    reml = reml, optim_interval = optim_interval)


end

function bulkscan(Y::Array{Float64, 2}, G::Array{Float64, 2}, Covar::Array{Float64, 2}, K::Array{Float64, 2};
                  method::String = "null-grid", h2_grid::Array{Float64, 1} = collect(0.0:0.1:0.9),
                  nb::Int64 = Threads.nthreads(), 
                  nt_blas::Int64 = 1,
                  addIntercept::Bool = true, 
                  weights::Union{Missing, Array{Float64, 1}} = missing,
                  prior_variance::Float64 = 1.0, prior_sample_size::Float64 = 0.0,
                  reml::Bool = false, optim_interval::Int64 = 1)
    
    if method == "null-exact"
        return bulkscan_null(Y, G, Covar, K; 
                             nb = nb, nt_blas = nt_blas,
                             addIntercept = addIntercept, 
                             weights = weights,
                             prior_variance = prior_variance, prior_sample_size = prior_sample_size,
                             reml = reml, optim_interval = optim_interval);
    end

    if method == "null-grid"
        return bulkscan_null_grid(Y, G, Covar, K, h2_grid; 
                                  weights = weights,
                                  addIntercept = addIntercept,
                                  prior_variance = prior_variance, prior_sample_size = prior_sample_size,
                                  reml = reml);
    end

    if method == "alt-grid"
        return bulkscan_alt_grid(Y, G, Covar, K, h2_grid; 
                                  weights = weights,
                                  addIntercept = addIntercept,
                                  prior_variance = prior_variance, prior_sample_size = prior_sample_size,
                                  reml = reml);
    end

end

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
                       nt_blas::Int64 = 1, 
                       weights::Union{Missing, Array{Float64, 1}} = missing,
                       prior_variance::Float64 = 1.0, prior_sample_size::Float64 = 0.0,
                       reml::Bool = false, optim_interval::Int64 = 1)

    n = size(Y, 1);

    # when no covariates are added, make the intercept as the only covariate
    intercept = ones(n, 1);

    return bulkscan_null(Y, G, intercept, K; 
                         nb = nb, nt_blas = nt_blas, 
                         # key step: avoid adding the intercept twice
                         addIntercept = false, 
                         weights = weights,
                         prior_variance = prior_variance, prior_sample_size = prior_sample_size,
                         reml = reml, optim_interval = optim_interval);

end
### Modeling covariates version
function bulkscan_null(Y::Array{Float64, 2}, G::Array{Float64, 2}, 
                       Covar::Array{Float64, 2}, K::Array{Float64, 2};
                       nb::Int64 = Threads.nthreads(), nt_blas::Int64 = 1, 
                       addIntercept::Bool = true, 
                       weights::Union{Missing, Array{Float64, 1}} = missing,
                       prior_variance::Float64 = 1.0, prior_sample_size::Float64 = 0.0,
                       reml::Bool = false, optim_interval::Int64 = 1)


    m = size(Y, 2);
    p = size(G, 2);

    if addIntercept == true
        num_of_covar = size(Covar, 2)+1;
    else
        num_of_covar = size(Covar, 2)
    end

    if !ismissing(weights)
        W = diagm(weights);
        Y_st = W*Y;
        G_st = W*G;

        if addIntercept == true
            Covar_st = W*[ones(size(Y, 1)) Covar];
        else
            Covar_st = W*Covar
        end

        addIntercept = false;
        K_st = W*K*W;

    else
        Y_st = Y;
        G_st = G;
        Covar_st = Covar;
        K_st = K;
    end

    BLAS.set_num_threads(nt_blas);

    # rotate data
    (Y0, X0, lambda0) = transform_rotation(Y_st, [Covar_st G_st], K_st; addIntercept = addIntercept);


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
                                 reml = reml, optim_interval = optim_interval);

        @inbounds lods_currBlock[:, i] = outputs.R;
        @inbounds h2_null_list[j] = outputs.h2
    end

        results[t] = lods_currBlock;

    end

    LODs_all = reduce(hcat, results);

    # if no remainder as the result of blocking, no remaining traits need to be scanned
    if rem == 0
        return (L = LODs_all, h2_null_list = h2_null_list)
    end

    # else, process up the remaining traits
    lods_remBlock = Array{Float64, 2}(undef, p, rem);

    for i in 1:rem

        j = m-rem+i;

        outputs = univar_liteqtl(Y0[:, j], X0_intercept, X0_covar, lambda0;
                                 prior_variance = prior_variance, prior_sample_size = prior_sample_size,
                                 reml = reml, optim_interval = optim_interval);
        
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
function bulkscan_null_grid(Y::Array{Float64, 2}, G::Array{Float64, 2}, 
                            K::Array{Float64, 2}, grid_list::Array{Float64, 1};
                            weights::Union{Missing, Array{Float64, 1}} = missing, 
                            prior_variance::Float64 = 1.0, prior_sample_size::Float64 = 0.0, 
                            reml::Bool = false)


    n = size(Y, 1);
    intercept = ones(n, 1);

    return bulkscan_null_grid(Y, G, intercept, K, grid_list; 
                              weights = weights,
                              addIntercept = false,
                              prior_variance = prior_variance, prior_sample_size = prior_sample_size,
                              reml = reml);

end
function bulkscan_null_grid(Y::Array{Float64, 2}, G::Array{Float64, 2}, Covar::Array{Float64, 2}, 
                            K::Array{Float64, 2}, grid_list::Array{Float64, 1};
                            weights::Union{Missing, Array{Float64, 1}} = missing, 
                            addIntercept::Bool = true,
                            prior_variance::Float64 = 1.0, prior_sample_size::Float64 = 0.0, 
                            reml::Bool = false)

    m = size(Y, 2);
    p = size(G, 2);

    if !ismissing(weights)
        W = diagm(weights);
        Y_st = W*Y;
        G_st = W*G;

        if addIntercept == true
            Covar_st = W*[ones(size(Y, 1)) Covar];
        else
            Covar_st = W*Covar
        end

        addIntercept = false;
        K_st = W*K*W;

    else
        Y_st = Y;
        G_st = G;
        Covar_st = Covar;
        K_st = K;
    end

    results_by_bin = gridscan_by_bin(Y_st, G_st, Covar_st, K_st, grid_list; 
                                     addIntercept = addIntercept, 
                                     prior_variance = prior_variance, prior_sample_size = prior_sample_size,
                                     reml = reml);
    
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
function bulkscan_alt_grid(Y::Array{Float64, 2}, G::Array{Float64, 2}, K::Array{Float64, 2}, 
                           hsq_list::Array{Float64, 1};
                           reml::Bool = false,
                           prior_variance::Float64 = 1.0, prior_sample_size::Float64 = 0.0, 
                           weights::Union{Missing, Array{Float64, 1}} = missing)

    n = size(Y, 1);
    intercept = ones(n, 1);

    return bulkscan_alt_grid(Y, G, intercept, K, hsq_list; 
                             reml = reml, 
                             prior_variance = prior_variance, prior_sample_size = prior_sample_size, 
                             weights = weights, addIntercept = false);

end

function bulkscan_alt_grid(Y::Array{Float64, 2}, G::Array{Float64, 2}, 
                           Covar::Array{Float64, 2}, K::Array{Float64, 2}, hsq_list::Array{Float64, 1};
                           reml::Bool = false,
                           prior_variance::Float64 = 1.0, prior_sample_size::Float64 = 0.0, 
                           weights::Union{Missing, Array{Float64, 1}} = missing, 
                           addIntercept::Bool = true)
    

    p = size(G, 2);
    m = size(Y, 2);

    if !ismissing(weights)
        W = diagm(weights);
        Y_st = W*Y;
        G_st = W*G;
                            
        if addIntercept == true
            Covar_st = W*[ones(size(Y, 1)) Covar];
        else
            Covar_st = W*Covar
        end
                            
        addIntercept = false;
        K_st = W*K*W;
                            
    else
        Y_st = Y;
        G_st = G;
        Covar_st = Covar;
        K_st = K;
    end

    (Y0, X0, lambda0) = transform_rotation(Y_st, [Covar_st G_st], K_st; addIntercept = addIntercept);

    if addIntercept == true
        num_of_covar = size(Covar, 2)+1;
    else
        num_of_covar = size(Covar, 2);
    end

    if num_of_covar == 1
        X0_base = reshape(X0[:, 1], :, 1);
    else
        X0_base = X0[:, 1:num_of_covar];
    end

    prior = [prior_variance, prior_sample_size];
    ## initializing outputs:
    logLR = weighted_liteqtl(Y0, X0, lambda0, hsq_list[1]; num_of_covar = num_of_covar);
    logLR = logLR .* log(10);
    weights_1 = makeweights(hsq_list[1], lambda0);
    logL0 = wls_multivar(Y0, X0_base, weights_1, prior; reml = reml).Ell;
    logL1 = logLR .+ repeat(logL0, p);

    logL0_all_h2 = zeros(length(hsq_list), m);
    logL0_all_h2[1, :] = logL0;
    k = 1;

    h2_panel = ones(p, m) .* hsq_list[1]; 
    h2_panel_counter = Int.(ones(p, m));

    for h2 in hsq_list[2:end]

        logLR_k = weighted_liteqtl(Y0, X0, lambda0, h2) .* log(10);
        weights_k = makeweights(h2, lambda0);
        logL0_k = wls_multivar(Y0, X0_base, weights_k, prior; reml = reml).Ell; 
        logL1_k = logLR_k .+ repeat(logL0_k, p);

        k = k+1;
        logL0_all_h2[k, :] = logL0_k;

        tmax!(logL1, logL1_k, h2_panel, h2_panel_counter, hsq_list);
    end

    logL0_optimum = mapslices(x -> maximum(x), logL0_all_h2, dims = 1) |> x -> repeat(x, p);
    L = (logL1 .- logL0_optimum) ./ log(10);

    return (L = L, h2_panel = h2_panel);

end

