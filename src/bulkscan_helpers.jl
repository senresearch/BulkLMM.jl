"""
    BulkLMM.Results_by_bin

Struct that stores BulkLMM grid-search-based method results. Essentially, the struct wraps the model outputs of LOD scores, 
estimated marker effects and heritabilities by groups, with the grouping defined by the h2 value taken from the given grid. 
Besides model outputs, the struct also records information such as the correponding indices as in the original data, needed 
to reorder the results to match with the original indexing for final reporting.

Grouped results are in forms of `Array`'s of model outputs for each group. For example, for scanning a total of five traits in 
the input data, y₁, y₂, y₃, y₄, y₅, suppose that by a BulkLMM grid-search-based method with a simple grid of three values,
(0.0, 0.5, 0.75). The estimated variance component h² for each trait forms the collection (0.0, 0.5, 0.0, 0.75, 0.5), which
indicates the grouping as group 1 with (y₁, y₃) corresponding to h² = 0.0, and similarly the other two groups, 
(y₂, y₅) for  ĥ² = 0.5, (y₄) for ĥ² = 0.75.

Then, `Results_by_bin` wraps:
    - `idxs_by_bin`: Array of boolean indicators indicating whether the trait appears in each group. With the given example, 
the ĥ² = 0.0 group picks traits (y₁, y₃), then this information is encoded as `[true, false, true, false, false]`, 
and similarly for the other two groups.
    - `LODs_by_bin`: Array of matrices, with each matrix stores marker associations (LOD scores) of the group of traits 
(e.g., the first matrix will have two columns, correponding to the LOD scores for y₁, y₃).
    - `Effect_sizes_by_bin`: Array of matrices, the same data structure as `LODs_by_bin` except that it stores grouped marker effects.
    - `h2_taken`: Array of floats indicating the h² value taken by each group.

"""
struct Results_by_bin
    idxs_by_bin::Array{Array{Bool, 1}, 1}
    LODs_by_bin::Array{Array{Float64, 2}, 1}
    Effect_sizes_by_bin::Array{Array{Float64, 2}, 1}
    h2_taken::Array{Float64, 1};
end

"""
    r2lod(r::Float64, n::Int64)

Given pairwise correlation `r` and sample size `n`, `r2lod` converts `r` to the corresponding LOD score `lod`. 

# Arguments
    - `r::Float64`: Pearson correlation coefficient between a trait and a genetic marker
    - `n::Int64`: Sample size

# Value
    `lod::Float64`: LOD score association from the given trait-marker correlation

"""
function r2lod(r::Float64, n::Int64)
    return -(n/2.0) * log10(1.0-r^2);
end



"""
    computeR_LMM(wY::Array{Float64, 2}, wX::Array{Float64, 2}, wIntercept::Array{Float64, 2})

Compute the matrix of trait-marker correlations after removing the effect of the (rotated) intercept in the linear mixed model 
transformed space.

`computeR_LMM` first residualizes both the transformed trait matrix `wY` and transformed marker matrix `wX` with respect to 
the transformed intercept `wIntercept`, then standardizes each column to unit norm, and finally returns 
the column-wise correlation matrix `R = X00' * Y00`.

# Arguments
    - `wY::Array{Float64, 2}`: Transformed response matrix (e.g., rotated and then reweighted traits), with columns 
        corresponding to traits
    - `wX::Array{Float64, 2}`: Transformed marker matrix (e.g., rotated markers), with columns corresponding to markers
    - `wIntercept::Array{Float64, 2}`: Transformed intercept (can include non-marker covariates) design matrix 
        used to residualize `wY` and `wX`
        

# Value
    `R::Array{Float64, 2}`: Matrix of Pearson correlations between markers (columns of `wX`) and traits (columns of `wY`) 
        after residualizing the intercept and column normalization

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
    threaded_map!(f::Function, M::Array{Float64, 2}, args...; dims::Int64 = 2)

Perform a threaded in-place elementwise mapping on a 2D matrix `M` using function `f`.

`threaded_map!` applies `f` to each entry of `M` (with optional extra arguments `args...`) and writes the results 
back into `M` in place. The computation is parallelized across rows or columns depending on `dims`.

# Arguments
    - `f::Function`: Elementwise mapping function, applied as `f(x, args...)`
    - `M::Array{Float64, 2}`: Input matrix to be modified in place
    - `args...`: Additional arguments passed to `f`
    - `dims::Int64 = 2`: Dimension along which threading is performed (`1` = thread over rows, `2` = thread over columns)

# Value
    No return value. The input matrix `M` is modified in place, equivalent in effect to applying `f.(M, args...)` elementwise.

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
    univar_liteqtl(y0_j::AbstractArray{Float64, 1}, X0_intercept::AbstractArray{Float64, 2},
                   X0_covar::AbstractArray{Float64, 2}, lambda0::AbstractArray{Float64, 1};
                   prior_variance = 0.0, prior_sample_size = 0.0,
                   reml::Bool = false, optim_interval::Int64 = 1)

Calculate trait-marker association statistics for one trait using the LiteQTL approach.

`univar_liteqtl` estimates variance components (heritability) for a single rotated trait under the null model, 
    uses the estimated heritability to construct weights, re-weights the trait/intercept/marker matrices, 
    computes trait-marker correlations in the weighted space, and converts them to LOD scores. 
    It also returns regression coefficients in the weighted space and the estimated heritability.

# Arguments
    - `y0_j::AbstractArray{Float64, 1}`: Rotated vector for the `j`-th trait
    - `X0_intercept::AbstractArray{Float64, 2}`: Rotated intercept design matrix
    - `X0_covar::AbstractArray{Float64, 2}`: Rotated marker (covariate) design matrix
    - `lambda0::AbstractArray{Float64, 1}`: Eigenvalues of the kinship matrix

# Keyword Arguments
    - `prior_variance = 0.0`: Prior variance hyperparameter used in variance-component estimation
    - `prior_sample_size = 0.0`: Prior sample size (prior strength) hyperparameter used in variance-component estimation
    - `reml::Bool = false`: Whether to use REML (`true`) or ML (`false`) for variance-component estimation
    - `optim_interval::Int64 = 1`: Optimization interval/control parameter passed to `fitlmm`

# Value
    A named tuple `(B, R, h2)` where:
    - `B::Array{Float64, 2}`: Weighted-space regression coefficients for intercept and markers
    - `R::Array{Float64, 2}`: LOD scores for the given trait across all markers (stored in the correlation 
    matrix after `r2lod` conversion)
    - `h2::Float64`: Estimated heritability from the null-model LMM fit

# Notes
Assumes heritability (and thus variance components) is trait-specific but shared across all marker scans for the same trait. 
Variance components are estimated once from the null model and reused for all markers for that trait.

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

"""
    weighted_liteqtl(Y0::Array{Float64, 2}, X0::Array{Float64, 2},
                     lambda0::Array{Float64, 1}, hsq::Float64;
                     num_of_covar::Int64 = 1, dims::Int64 = 2)

Calculate trait-marker LOD scores for all pairs of traits and markers under a fixed heritability value.

`weighted_liteqtl` assumes the inputs are already rotated (e.g., by the eigendecomposition of the kinship matrix), 
constructs a common weight vector from the provided heritability `hsq`, re-weights the trait and design matrices, 
and computes LOD scores using the LiteQTL correlation-based shortcut. It also returns weighted-space regression coefficients.

# Arguments
    - `Y0::Array{Float64, 2}`: Rotated trait matrix, with columns corresponding to traits
    - `X0::Array{Float64, 2}`: Rotated design matrix containing intercept/covariates and marker columns
    - `lambda0::Array{Float64, 1}`: Eigenvalues of the kinship matrix
    - `hsq::Float64`: Heritability value used to construct the weight matrix

# Keyword Arguments
    - `num_of_covar::Int64 = 1`: Number of leading columns in `X0` treated as baseline covariates (typically intercept only if `1`)
    - `dims::Int64 = 2`: Dimension passed to `threaded_map!` when converting correlations to LOD scores 
    (`1` = thread over rows, `2` = thread over columns)

# Value
    A named tuple `(LOD, B)` where:
    - `LOD::Array{Float64, 2}`: Matrix of LOD scores for all marker-trait pairs under the given heritability
    - `B::Array{Float64, 2}`: Weighted-space regression coefficients from `wX0 \\ wY0`

# Notes
Inputs are assumed to be pre-rotated. This shortcut is intended for testing 1-df fixed effects (e.g., one marker effect at a time).

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

"""
    find_optim_h2(h2_list::Array{Float64, 1}, results::Array{Float64, 2})

Find the optimal heritability value for each trait from a grid of candidate heritability values.

`find_optim_h2` identifies, for each trait (column of `results`), the row index corresponding to the maximum 
value across the tested heritability grid, and returns the matching heritability values from `h2_list`.

# Arguments
    - `h2_list::Array{Float64, 1}`: Vector of candidate heritability values used in the grid search
    - `results::Array{Float64, 2}`: Matrix of objective values (or scores), where rows correspond to entries 
    in `h2_list` and columns correspond to traits

# Value
    `optim_h2_each_trait::Array{Float64}`: Vector of selected heritability values, one for each trait, 
    corresponding to the maximizing row in `results`.

"""
function find_optim_h2(h2_list::Array{Float64, 1}, results::Array{Float64, 2})
    
    max_h2_idxs = mapslices(x -> findmax(x)[2], results; dims = 1);
    optim_h2_each_trait = h2_list[max_h2_idxs];
    
    return optim_h2_each_trait;
    
end

"""
    distribute_traits_by_h2(idxs_sets::Dict{Int64, Float64}, h2_taken::Array{Float64, 1}, m::Int64, nbins::Int64)

Group traits into heritability bins and return Boolean membership indicators for each bin.

`distribute_traits_by_h2` constructs, for each heritability value in `h2_taken`, 
a Boolean vector of length `m` indicating whether each trait index is assigned to that heritability value in `idxs_sets`.

# Arguments
    - `idxs_sets::Dict{Int64, Float64}`: Dictionary mapping trait index to an assigned heritability value
    - `h2_taken::Array{Float64, 1}`: Vector of heritability values defining the bins
    - `m::Int64`: Total number of traits (or indices) to evaluate, typically indexed from `1:m`
    - `nbins::Int64`: Number of heritability bins (typically equal to `length(h2_taken)`)

# Value
    `ids_each_bin::Array{Array{Bool, 1}, 1}`: Vector of Boolean vectors, where `ids_each_bin[t][i]` is `true` 
    if trait/index `i` is assigned to `h2_taken[t]`, and `false` otherwise

# Notes
`distribute_traits_by_h2` is a helper function for BulkLMM `Null-Grid` method. It assigns traits sharing the same optimal value 
    of h² fitted into the same group. The returned value `ids_each_bin` contains a collection of boolean vectors, each of length 
    `m` and indicating whether each trait is in (`true`) or not-in (`false`) the group. 

"""
function distribute_traits_by_h2(idxs_sets::Dict{Int64, Float64}, h2_taken::Array{Float64, 1}, m::Int64, nbins::Int64)
    
    ids_each_bin = Array{Array{Bool, 1}, 1}(undef, nbins);
    
    for t in 1:nbins
        ids_each_bin[t] = map(x -> get(idxs_sets, x, Inf) == h2_taken[t], 1:m)
    end
    
    return ids_each_bin
end

"""
    gridscan_by_bin(pheno::Array{Float64, 2}, geno::Array{Float64, 2}, kinship::Array{Float64, 2},
                    grid::Array{Float64, 1};
                    prior_variance::Float64 = 1.0, prior_sample_size::Float64 = 0.0,
                    reml::Bool = false)

Perform a grid-based genome scan by heritability bins using phenotype, genotype, and kinship matrices, with an 
intercept added automatically.

This method is a convenience wrapper for `gridscan_by_bin` that constructs an intercept matrix (`ones(n,1)`) internally 
    and calls the corresponding method that accepts an explicit covariate/intercept design matrix.

# Arguments
    - `pheno::Array{Float64, 2}`: Phenotype matrix, with rows corresponding to samples and columns corresponding to traits
    - `geno::Array{Float64, 2}`: Genotype (or marker design) matrix, with rows corresponding to samples
    - `kinship::Array{Float64, 2}`: Kinship matrix used for linear mixed model adjustment
    - `grid::Array{Float64, 1}`: Grid of candidate heritability values used in the scan/binning procedure

# Keyword Arguments
    - `prior_variance::Float64 = 1.0`: Prior variance hyperparameter used in variance-component estimation
    - `prior_sample_size::Float64 = 0.0`: Prior sample size (prior strength) hyperparameter used in variance-component estimation
    - `reml::Bool = false`: Whether to use REML (`true`) or ML (`false`) in variance-component estimation

# Value
    Returns the output of the corresponding `gridscan_by_bin(pheno, geno, intercept, kinship, grid; ...)` method 
    (with `addIntercept = false`), using an internally constructed intercept matrix.

# Notes
This wrapper always adds an intercept column internally and passes `addIntercept = false` to the downstream method 
to avoid duplicating the intercept.

"""
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

"""
    gridscan_by_bin(pheno::Array{Float64, 2}, geno::Array{Float64, 2},
                    covar::Array{Float64, 2}, kinship::Array{Float64, 2},
                    grid::Array{Float64, 1};
                    addIntercept::Bool = true,
                    prior_variance = 1.0, prior_sample_size = 0.0,
                    reml::Bool = false,
                    decomp_scheme::String = "eigen")

Perform a grid-search-based BulkLMM genome scan by grouping traits into bins that share the same estimated heritability.

`gridscan_by_bin` is the main workflow function for the grid-search-based BulkLMM scan. It first rotates the phenotype and design matrices using the kinship matrix, then performs a grid search over candidate heritability values (`grid`) under the null model to select an optimal `h²` for each trait. Traits sharing the same selected `h²` are grouped into bins, and each bin is scanned efficiently using `weighted_liteqtl`, which reuses a common weight structure and computes marker-trait LOD scores via matrix multiplication (LiteQTL shortcut).

The function returns a `Results_by_bin` object that stores grouped LOD scores, grouped marker effect sizes, bin memberships (for reordering to original trait order), and the heritability value assigned to each bin.

# Arguments
    - `pheno::Array{Float64, 2}`: Phenotype matrix, with rows corresponding to samples and columns corresponding to traits
    - `geno::Array{Float64, 2}`: Genotype/marker design matrix, with rows corresponding to samples and columns corresponding to markers
    - `covar::Array{Float64, 2}`: Covariate design matrix (excluding genotype markers; may or may not include intercept depending on `addIntercept`)
    - `kinship::Array{Float64, 2}`: Kinship matrix used for mixed-model correlation adjustment
    - `grid::Array{Float64, 1}`: Grid of candidate heritability values used to select trait-specific `h²` by grid search

# Keyword Arguments
    - `addIntercept::Bool = true`: Whether to add an intercept column during the rotation/design construction step
    - `prior_variance = 1.0`: Prior variance hyperparameter used in null-model variance-component estimation over the `h²` grid
    - `prior_sample_size = 0.0`: Prior sample size (prior strength) hyperparameter used in null-model variance-component estimation
    - `reml::Bool = false`: Whether to use REML (`true`) or ML (`false`) in the grid-based null-model evaluation
    - `decomp_scheme::String = "eigen"`: Matrix decomposition scheme passed to `transform_rotation`

# Value
    `Results_by_bin`: A struct that stores grouped scan results from the BulkLMM grid-search-based method, including:
    - `idxs_by_bin`: Boolean membership indicators for traits in each heritability bin
    - `LODs_by_bin`: LOD-score matrices for each bin (markers × traits-in-bin)
    - `Effect_sizes_by_bin`: Marker-effect matrices for each bin (markers × traits-in-bin)
    - `h2_taken`: Heritability value assigned to each bin

# Notes
`gridscan_by_bin` is the core function of BulkLMM "Null-Grid" method. It calls `find_optim_h2` (to "fit" the h2 parameter through grid-search), 
    `distribute_traits_by_h2` (to group traits that share the same optimal value from grid), 
    and finally `weighted_liteqtl`` (to compute LOD scores efficiently through matrix multiplication technique (LiteQTL method) 
        for each block/group of traits) in sequence.

The returned results are grouped by heritability bin (not original trait order). The "Null-Grid" then uses `idxs_by_bin` to map/reorder 
outputs back to the original trait indexing for downstream reporting.

"""
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

"""
    reorder_results(blocking_idxs::Array{Array{Bool, 1}, 1},
                    lods_by_block::Array{Array{Float64, 2}, 1},
                    effect_sizes_by_block::Array{Array{Float64, 2}, 1},
                    m::Int64, p::Int64)

Reorder grouped BulkLMM scan results back to the original trait order.

`reorder_results` takes LOD-score and marker-effect matrices that are stored by heritability bins (blocks) and reconstructs full 
`p × m` matrices whose columns match the original trait indexing. The trait membership of each block is specified 
by the Boolean indicators in `blocking_idxs`.

# Arguments
    - `blocking_idxs::Array{Array{Bool, 1}, 1}`: Array of Boolean membership vectors, where each vector indicates which 
    traits belong to a given block/bin
    - `lods_by_block::Array{Array{Float64, 2}, 1}`: Array of LOD-score matrices, one per block, with dimensions 
    `(p, number_of_traits_in_block)`
    - `effect_sizes_by_block::Array{Array{Float64, 2}, 1}`: Array of marker-effect matrices, one per block, with dimensions 
    `(p, number_of_traits_in_block)`
    - `m::Int64`: Total number of traits (number of columns in the reordered output)
    - `p::Int64`: Total number of markers/effects (number of rows in the reordered output)

# Value
    A named tuple `(LOD, B)` where:
    - `LOD::Array{Float64, 2}`: Reconstructed LOD-score matrix of size `(p, m)` in the original trait order
    - `B::Array{Float64, 2}`: Reconstructed marker-effect matrix of size `(p, m)` in the original trait order

# Notes
This function is typically used downstream of `gridscan_by_bin`, which returns grouped results by heritability bin. It assumes that:
- each trait belongs to exactly one block, and
- the columns of each block matrix are ordered consistently with the `true` positions in the corresponding `blocking_idxs[block]`.

"""
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

"""
    tmax!(currL1::Array{Float64, 2}, nextL1::Array{Float64, 2},
          currB::Array{Float64, 2}, nextB::Array{Float64, 2},
          h2_panel::Array{Float64, 2},
          h2_panel_counter::Array{Int64, 2},
          h2_list::Array{Float64, 1})

Perform threaded in-place elementwise maximization across two matrices and update the corresponding effect sizes and `h²` records.

`tmax!` compares `currL1` and `nextL1` entry by entry. Whenever `nextL1[i, j]` is larger than `currL1[i, j]`, the function 
updates the current maximum value in `currL1`, replaces the corresponding effect size in `currB` with `nextB[i, j]`, 
and records the updated optimal heritability value in `h2_panel` using `h2_panel_counter` and `h2_list`.

# Arguments
    - `currL1::Array{Float64, 2}`: Matrix of current maximum objective values (e.g., alternative log-likelihood values) 
    over the `h²` grid
    - `nextL1::Array{Float64, 2}`: Matrix of candidate objective values to compare against `currL1`
    - `currB::Array{Float64, 2}`: Matrix storing effect sizes corresponding to the current maxima in `currL1`
    - `nextB::Array{Float64, 2}`: Matrix of effect sizes corresponding to `nextL1`
    - `h2_panel::Array{Float64, 2}`: Matrix storing the currently selected optimal `h²` value for each entry
    - `h2_panel_counter::Array{Int64, 2}`: Matrix of integer indices used to track positions in `h2_list` for each entry
    - `h2_list::Array{Float64, 1}`: Vector of candidate `h²` values

# Value
    No return value. The function updates `currL1`, `currB`, `h2_panel`, and `h2_panel_counter` in place.

# Notes
`tmax!` is the core function for BulkLMM "Alt-Grid" method. It essentially computes intermediate outputs, null model profile likelihoods 
    and pseudo-LOD scores evaluated on each h² value in the grid, as panel-like data structures (i.e., coordinates corresponding
    to each trait-marker pair), and then optimizes alternative profile likelihood values in-place by comparing 
    the two panels `currL1` and `nextL1` corresponding to the current and next h² value from the grid.

    Comparisons between corresponding values across positions are parallelized through threaded operations.

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
