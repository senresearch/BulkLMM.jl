struct Results_by_bin
    idxs_by_bin::Array{Array{Bool, 1}, 1}
    LODs_by_bin::Array{Array{Float64, 2}, 1}
    Effect_sizes_by_bin::Array{Array{Float64, 2}, 1}
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

    # Exclude the effect of (rotated) intercept (idea is similar as centering data in the linear model case)
    Y00 = resid(wY, wIntercept);
    X00 = resid(wX, wIntercept);

    # Standardize the response and covariates by dividing by their norms
    norm_Y = mapslices(x -> norm(x), Y00, dims = 1) |> vec;
    norm_X = mapslices(x -> norm(x), X00, dims = 1) |> vec;

    # replace!(norm_Y, 0 => 1.0)
    # replace!(norm_X, 0 => 1.0)

    colDivide!(Y00, norm_Y);
    colDivide!(X00, norm_X);

    # Matrix of correlation coefficients between the trait and all markers
    R = X00' * Y00;

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
                        reml::Bool = false, optim_interval::Int64 = 1
                        )

    n = size(y0_j, 1);
    y0 = reshape(y0_j, :, 1);

    # Estimate the heritability from the null model
    vc = fitlmm(y0, X0_intercept, lambda0, [prior_variance, prior_sample_size]; 
                reml = reml, optim_interval = optim_interval);
    # Construct weights to adjust for heteroskedasticity due to heritability
    sqrtw = sqrt.(abs.(makeweights(vc.h2, lambda0)));

    # Re-weight the data: final transformed data wy0 are homoskedestic and independent
    wy0 = rowMultiply(y0, sqrtw);
    wX0_intercept = rowMultiply(X0_intercept, sqrtw);
    wX0_covar = rowMultiply(X0_covar, sqrtw);

    # Matrix of correlation coefficients between the trait and all markers
    R = computeR_LMM(wy0, wX0_covar, wX0_intercept);

    # Estimate effect sizes for baseline covariates and markers
    wX0_covar_intercept = hcat(wX0_intercept, wX0_covar);
    B = wX0_covar_intercept\wy0; # effect sizes for all markers + intercept

    threaded_map!(r2lod, R, n; dims = 2);

    # Return LOD scores, effect sizes, and heritability estimate
    return (B = B, R = R, h2 = vc.h2);

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
    b = wX0 \ wY0;

    if num_of_covar == 1
        wX0_intercept = reshape(wX0[:, 1], :, 1);
    else
        wX0_intercept = wX0[:, 1:num_of_covar];
    end

    wX0_covar = wX0[:, (num_of_covar+1):end];

    # Perform the LiteQTL approach for fast calculations of LOD scores
    LOD = computeR_LMM(wY0, wX0_covar, wX0_intercept);

    threaded_map!(r2lod, LOD, n; dims = dims); # results will be p-by-1, i.e. all LOD scores for the j-th trait and p markers

    return (LOD = LOD, B = b)
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
                         reml::Bool = false,
                         decomp_scheme::String = "eigen"
                         )

    m = size(pheno, 2);

    # Y_std = colStandardize(pheno);
    Y_std = pheno;
    ##################################################################################
    ## Step 1: Rotate to decorrelate the individuals: This is the same for all traits
    (Y0, X0, lambda0) = transform_rotation(Y_std, [covar geno], kinship; 
                                           addIntercept = addIntercept, decomp_scheme = decomp_scheme);

    prior = [prior_variance, prior_sample_size];


    if addIntercept == true
        num_of_covar = size(covar, 2)+1; # `+1` for the intercept column
    else
        num_of_covar = size(covar, 2);
    end

    X0_intercept = X0[:, 1:num_of_covar];

    ##################################################################################
    ## Step 2: Perform a quick scan of h2's (under null and by grid-search) for all traits
    
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

    ##################################################################################
    ## Step 3: Distribute traits by h2 values into bins: Traits inside a bin share the same h2
    
    blocking_idxs = distribute_traits_by_h2(idxs_sets, h2_taken, m, nbins);
    # LOD scores for each bin
    results = Array{Array{Float64, 2}, 1}(undef, nbins);
    # Effect sizes for each bin
    effect_sizes_by_bin = Array{Array{Float64, 2}, 1}(undef, nbins);
    

    ##################################################################################
    ## Step 4: Compute LOD scores for each bin of traits using the weighted LiteQTL approach
    
    for t in 1:nbins
        out = weighted_liteqtl(Y0[:, blocking_idxs[t]], X0, lambda0, h2_taken[t]; 
                                      num_of_covar = num_of_covar);

        results[t] = out.LOD;
        # selected_effects = vcat(1, (num_of_covar+1):size(out.B, 1)) # baseline (1) + marker (num_of_covar+1:end) effects
        selected_effects = (num_of_covar+1):size(out.B, 1) # marker effects only
        effect_sizes_by_bin[t] = out.B[selected_effects, :]; # exclude the covariate effects
        # print(size(effect_sizes_by_bin[t]))

    end

    ##################################################################################
    ## Final output: LOD scores for traits ordered by the order of bins
    ## (Downstream to reorder the results by the original trait order)
    return Results_by_bin(blocking_idxs, results, effect_sizes_by_bin, h2_taken)
    
end

function reorder_results(blocking_idxs::Array{Array{Bool, 1}, 1}, 
                         lods_by_block::Array{Array{Float64, 2}, 1},
                         effect_sizes_by_block::Array{Array{Float64, 2}, 1}, 
                         m::Int64, p::Int64)
    
    LOD = Array{Float64, 2}(undef, p, m);
    B = Array{Float64, 2}(undef, p, m);
    
    
    for block in 1:length(blocking_idxs)
        idxs_curr_block = blocking_idxs[block];
        LOD[:, idxs_curr_block] = lods_by_block[block];
        B[:, idxs_curr_block] = effect_sizes_by_block[block];
    end
    
    return (LOD = LOD, B = B)
    
end

## MaxBulk helper functions:
"""
tmax!(currL1, nextL1, currB, nextB, h2_panel, h2_panel_counter, h2_list)

Does element-wise comparisons of two 2d Arrays and keep the larger elements in-place. 

# Arguments
- currL1 = 2d Array of Float; matrix of current maximum l1 (alt-loglik) values over h2 grid
- nextL1 = 2d Array of Float; matrix of values to compare with the current maximum values
- currB = 2d Array of Float; matrix to store the effect sizes corresponding to the current maximum l1 values
- nextB = 2d Array of Float; matrix of effect sizes corresponding to the next L1 values
- h2_panel = 2d Array of Float; matrix to store the optimal h2 values for each entry
- h2_panel_counter = 2d Array of Int; matrix to store the indices of the optimal h2 values for each entry
- h2_list = 1d Array of Float; list of candidate h2 values

# Value

Nothing; does in-place maximizations.

# Notes:

Updates optimal values in `currL1` in-place, in a threaded loop.

"""
function tmax!(currL1::Array{Float64, 2}, nextL1::Array{Float64, 2},
               currB::Array{Float64, 2}, nextB::Array{Float64, 2},  
               h2_panel::Array{Float64, 2}, 
               h2_panel_counter::Array{Int64, 2},
               h2_list::Array{Float64, 1})
    
    (p, m) = size(currL1);
    
    Threads.@threads for j in 1:m # Multiprocessing over traits over threads
        for i in 1:p # Loop over markers
            
            # Update each l1(i, j) value in-place, if current l1(i, j) is smaller:
            if (currL1[i, j] < nextL1[i, j])

                # Update new maximum LOD score:
                currL1[i, j] = nextL1[i, j];

                # Record the new optimal h2 value:
                h2_panel_counter[i, j] = h2_panel_counter[i, j]+1; # Index of the h2 value in h2_list
                h2_panel[i, j] = h2_list[h2_panel_counter[i, j]]; # Update the optimal h2 value from indices

                # Also, update the effect sizes based on the new optimal l1:
                currB[i, j] = nextB[i, j];
            end

            # do nothing if not.
        end
    end
    
end
