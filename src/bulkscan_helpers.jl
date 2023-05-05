struct Results_by_bin
    idxs_by_bin::Array{Array{Bool, 1}, 1}
    LODs_by_bin::Array{Array{Float64, 2}, 1}
    h2_taken::Array{Float64, 1};
end

"""
r2lod(r, n)

Given the pairwise correlation `r` and sample size `n`, convert `r` to the corresponding LOD score. 

# Arguments

- r = Float; correlation between a trait and a genetic marker
- n = Int; sample size

# Value

- lod = Float; LOD score of the corresponding trait and marker

"""
function r2lod(r::Float64, n::Int64)
    return -(n/2.0) * log10(1.0-r^2);
end



"""
computeR_LMM(wY, wX, wIntercept)

Calculates the LOD scores for one trait, using the LiteQTL approach.

# Arguments
- wY = 2d Array of Float; (weighted) a single trait (matrix of one column) or multiple traits (matrix of more than one column)
- wX = 2d Array of Float; (weighted)
- wIntercept

# Value

- R = 2d Array of Float; p-by-m matrix the correlation coefficients between each pair of traits (in wY) and markers (in wX)

# Notes:

Inputs are rotated, re-weighted.

"""
function computeR_LMM(wY::Array{Float64, 2}, wX::Array{Float64, 2}, wIntercept::Array{Float64, 2})

    # exclude the effect of (rotated) intercept (idea is similar as centering data in the linear model case)
    Y00 = resid(wY, wIntercept);
    X00 = resid(wX, wIntercept);
    
    # n = size(y00, 1);

    # standardize the response and covariates by dividing by their norms
    norm_Y = mapslices(x -> norm(x), Y00, dims = 1) |> vec;
    norm_X = mapslices(x -> norm(x), X00, dims = 1) |> vec;

    colDivide!(Y00, norm_Y);
    colDivide!(X00, norm_X);

    R = X00' * Y00; # p-by-m matrix

    return R

end

"""
threaded_map!(f, M, args...; dims = dims)

Performs threaded in-place mappings of entries in M based on the given function `f`.

# Arguments
- f = generic function of R -> R mapping.
- M = 2d Array of Float
- args... = splatting of other arguments for function `f(x, args...)`
- dims = Int; user specified dimension to map along the given matrix `M`

# Value

- No return value; makes in-place mapping to each entry of the input matrix `M`, i.e. same as `f.(M, args...)`.

# Notes:

"""
function threaded_map!(f::Function, M::Array{Float64, 2}, args...; dims::Int64 = 2)
    m = size(M, dims);
    
    if dims == 1
        Threads.@threads for i in 1:m
            @inbounds M[i, :] .= @views f.(M[i, :], args)
        end
    elseif dims == 2
        Threads.@threads for j in 1:m
            @inbounds M[:, j] .= @views f.(M[:, j], args)
        end
    end

end

"""
univar_liteqtl(y0_j, X0_intercept, X0_covar, lambda0; reml = true)

Calculates the LOD scores for one trait, using the LiteQTL approach.

# Arguments

- y0_j = the j-th trait rotated
- X0_intercept = the intercept rotated
- X0_covar = the markers rotated
- lambda0 = eigenvalues of the kinship matrix

# Keyword Arguments

- reml = Bool indicating whether ML or REML estimate is required; default is REML.

# Value

- R = Float; LOD score of the corresponding trait and marker

# Notes:

Assumes the heritabilities only differ by traits but remain the same across all markers for the same trait;
    hence, VC is estimated once based on the null model and used for all markers scans for that trait.
    (VC: variance components)


"""
function univar_liteqtl(y0_j::AbstractArray{Float64, 1}, X0_intercept::AbstractArray{Float64, 2}, 
                        X0_covar::AbstractArray{Float64, 2}, lambda0::AbstractArray{Float64, 1}; 
                        prior_variance = 0.0, prior_sample_size = 0.0,
                        reml::Bool = false, optim_interval::Int64 = 1)

    n = size(y0_j, 1);
    y0 = reshape(y0_j, :, 1);

    # estimate the heritability from the null model and apply it to the reweighting of all markers;
    vc = fitlmm(y0, X0_intercept, lambda0, [prior_variance, prior_sample_size]; 
                reml = reml, optim_interval = optim_interval);
    sqrtw = sqrt.(abs.(makeweights(vc.h2, lambda0)));

    # re-weight the data; then in theory, the observations are homoskedestic and independent.
    wy0 = rowMultiply(y0, sqrtw);
    wX0_intercept = rowMultiply(X0_intercept, sqrtw);
    wX0_covar = rowMultiply(X0_covar, sqrtw);

    R = computeR_LMM(wy0, wX0_covar, wX0_intercept);
    threaded_map!(r2lod, R, n; dims = 2);

    return (R = R, h2 = vc.h2); # results will be p-by-1, i.e. all LOD scores for the j-th trait and p markers

end


### Helper functions for applying LiteQTL approach on multiple traits
"""
weighted_liteqtl(Y0, X0, hsq, lambda0)

Calculates LOD scores for all pairs of traits and markers with a given heritability estimate.

# Arguments
- Y0 = 2d Array of Float; rotated traits 
- X0 = 2d Array of Float; rotated genotype probabilities
- hsq = Float; heritability
- lambda0 = 1d Array of Float; eigenvalues of the kinship matrix


# Value

- LOD = 2d Array of Float; matrix of LOD scores for all traits and markers calculated under the given heritability

# Notes:

Inputs data are assumed to be rotated; the shortcut works for only testing 1-df fixed effects.

"""
function weighted_liteqtl(Y0::Array{Float64, 2}, X0::Array{Float64, 2}, 
                          lambda0::Array{Float64, 1}, hsq::Float64;
                          num_of_covar::Int64 = 1, dims::Int64 = 2)

    n = size(Y0, 1)

    # Pre-process the data based on a common weight matrix (for all traits in Y0)
    sqrtw = sqrt.(abs.(makeweights(hsq, lambda0)));

    wY0 = rowMultiply(Y0, sqrtw);
    wX0 = rowMultiply(X0, sqrtw);

    if num_of_covar == 1
        wX0_intercept = reshape(wX0[:, 1], :, 1);
    else
        wX0_intercept = wX0[:, 1:num_of_covar];
    end

    wX0_covar = wX0[:, (num_of_covar+1):end];

    # Perform the LiteQTL approach for fast calculations of LOD scores
    LOD = computeR_LMM(wY0, wX0_covar, wX0_intercept);

    threaded_map!(r2lod, LOD, n; dims = dims); # results will be p-by-1, i.e. all LOD scores for the j-th trait and p markers

    return LOD
end

## GridBulk helper functions:
function find_optim_h2(h2_list::Array{Float64, 1}, results::Array{Float64, 2})
    
    max_h2_idxs = mapslices(x -> findmax(x)[2], results; dims = 1);
    optim_h2_each_trait = h2_list[max_h2_idxs];
    
    return optim_h2_each_trait;
    
end

function distribute_traits_by_h2(idxs_sets::Dict{Int64, Float64}, h2_taken::Array{Float64, 1}, m::Int64, nbins::Int64)
    
    ids_each_bin = Array{Array{Bool, 1}, 1}(undef, nbins);
    
    for t in 1:nbins
        ids_each_bin[t] = map(x -> get(idxs_sets, x, Inf) == h2_taken[t], 1:m)
    end
    
    return ids_each_bin
end

function gridscan_by_bin(pheno::Array{Float64, 2}, geno::Array{Float64, 2}, kinship::Array{Float64, 2}, 
                         grid::Array{Float64, 1};
                         prior_variance::Float64 = 1.0, prior_sample_size::Float64 = 0.0,
                         reml::Bool = false)

    n = size(pheno, 1);
    intercept = ones(n, 1);
    
    return gridscan_by_bin(pheno, geno, intercept, kinship, grid; 
                           addIntercept = false, 
                           prior_variance = prior_variance, prior_sample_size = prior_sample_size,
                           reml = reml);

end

function gridscan_by_bin(pheno::Array{Float64, 2}, geno::Array{Float64, 2}, 
                         covar::Array{Float64, 2}, kinship::Array{Float64, 2}, 
                         grid::Array{Float64, 1}; 
                         addIntercept::Bool = true, 
                         prior_variance = 1.0, prior_sample_size = 0.0,
                         reml::Bool = false
                         )

    m = size(pheno, 2);

    # Y_std = colStandardize(pheno);
    Y_std = pheno;

    (Y0, X0, lambda0) = transform_rotation(Y_std, [covar geno], kinship; addIntercept = addIntercept);

    prior = [prior_variance, prior_sample_size];


    if addIntercept == true
        num_of_covar = size(covar, 2)+1; # `+1` for the intercept column
    else
        num_of_covar = size(covar, 2);
    end

    X0_intercept = X0[:, 1:num_of_covar];

    weights_each_h2 = map(x -> makeweights(x, lambda0), grid); # make weights evaluated on each h2 in grid
    ell_results = map(x -> wls_multivar(Y0, X0_intercept, x, prior; reml = reml).Ell, weights_each_h2);
    ell_results = reduce(vcat, ell_results);

    optim_h2 = find_optim_h2(grid, ell_results)

    idxs_sets = Dict{Int64, Float64}();
    for i in 1:m
       idxs_sets[i] = optim_h2[i];
    end

    h2_taken = unique(values(idxs_sets));
    nbins = length(h2_taken);

    blocking_idxs = distribute_traits_by_h2(idxs_sets, h2_taken, m, nbins);
    results = Array{Array{Float64, 2}, 1}(undef, nbins);

    Threads.@threads for t in 1:nbins
    ## for t in 1:nbins
        results[t] = weighted_liteqtl(Y0[:, blocking_idxs[t]], X0, lambda0, h2_taken[t]; 
                                      num_of_covar = num_of_covar);
    end

    return Results_by_bin(blocking_idxs, results, h2_taken)
    
end

function reorder_results(blocking_idxs::Array{Array{Bool, 1}, 1}, 
                         lods_by_block::Array{Array{Float64, 2}, 1}, 
                         m::Int64, p::Int64)
    
    LOD = Array{Float64, 2}(undef, p, m);
    
    
    for block in 1:length(blocking_idxs)
        idxs_curr_block = blocking_idxs[block];
        LOD[:, idxs_curr_block] = lods_by_block[block];
    end
    
    return LOD
    
end

## MaxBulk helper functions:
"""
tmax!(max, toCompare)

Does element-wise comparisons of two 2d Arrays and keep the larger elements in-place. 

# Arguments
- max = 2d Array of Float; matrix of current maximum values
- toCompare = 2d Array of Flopat; matrix of values to compare with the current maximum values

# Value

Nothing; does in-place maximizations.

# Notes:

Will modify input matrix `max` by a parallelized loop; uses @tturbo in the package `LoopVectorization.jl`

"""
function tmax!(max::Array{Float64, 2}, toCompare::Array{Float64, 2})
    
    (p, m) = size(max);
    
    @tturbo for j in 1:m
        for i in 1:p
            
            max[i, j] = (max[i, j] >= toCompare[i, j]) ? max[i, j] : toCompare[i, j];
            
        end
    end
    
end
