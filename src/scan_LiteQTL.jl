
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
    rsq = (r/n)^2
    return -(n/2.0) * log10(1.0-rsq);
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
    y00 = resid(wY, wIntercept);
    X00 = resid(wX, wIntercept);

    # standardize the response and 
    sy = std(y00, dims = 1) |> vec;
    sx = std(X00, dims = 1) |> vec;
    colDivide!(y00, sy);
    colDivide!(X00, sx);

    R = X00' * y00; # p-by-1 matrix

    return R

end



"""
tR2LOD!(R, n)

Converts the input matrix R of pairwise correlations to the corresponding LOD score matrix

# Arguments
- R = 2d Array of Float; matrix of pairwise correlations
- n = Float; sample sizes

# Value

Nothing; does in-place conversions from correlation coefficients to LOD scores.

# Notes:

Will modify input matrix R; uses a multi-threaded loop.

"""
function tR2LOD!(R::Array{Float64, 2}, n::Int64)
    
    (p, m) = size(R)
    
    Threads.@threads for j in 1:m
        for i in 1:p
            @inbounds R[i, j] = r2lod(R[i, j], n)
        end
    end
    
end


"""
scan_lite_univar(y0_j, X0_intercept, X0_covar, lambda0; reml = true)

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


"""


function scan_lite_univar(y0_j::Array{Float64, 1}, X0_intercept::Array{Float64, 2}, 
    X0_covar::Array{Float64, 2}, lambda0::Array{Float64, 1};
    reml::Bool = true)

    n = size(y0_j, 1);
    y0 = reshape(y0_j, :, 1);

    # estimate the heritability from the null model and apply it to the reweighting of all markers;
    vc = fitlmm(y0, X0_intercept, lambda0; reml = reml);
    sqrtw = sqrt.(makeweights(vc.h2, lambda0));

    # re-weight the data; then in theory, the observations are homoskedestic and independent.
    wy0 = rowMultiply(y0, sqrtw);
    wX0_intercept = rowMultiply(X0_intercept, sqrtw);
    wX0_covar = rowMultiply(X0_covar, sqrtw);

    R = computeR_LMM(wy0, wX0_covar, wX0_intercept);
    tR2LOD!(R, n);

    return R; # results will be p-by-1, i.e. all LOD scores for the j-th trait and p markers

end




"""
scan_lite(Y, G, K, nb; reml = true)

Calculates the LOD scores for all pairs of traits and markers, by a (multi-threaded) loop over blocks of traits and the LiteQTL-type of approach

# Arguments
- Y = 2d Array of Float; matrix of one trait or multiple traits
- G = 2d Array of Float; matrix of genotype probabilities
- K = 2d Array of Float; kinship matrix
- nb = Int; number of blocks of traits required; ideally to be the same number of threads used for parallelization 

# Value

- LOD = 2d Array of Float; LOD scores for all pairs of traits and markers

# Notes:

Inputs are rotated, re-weighted.

"""

function scan_lite(Y::Array{Float64, 2}, G::Array{Float64, 2}, K::Array{Float64, 2}, nb::Int64; nt_blas::Int64 = 1,
    reml::Bool = true)


    (n, m) = size(Y);
    p = size(G, 2);

    BLAS.set_num_threads(nt_blas);

    # rotate data
    (Y0, X0, lambda0) = transform_rotation(Y, G, K);
    X0_intercept = reshape(X0[:, 1], :, 1);
    X0_covar = X0[:, 2:end];

    # distribute the `m` traits equally to every block
    (len, rem) = divrem(m, nb);

    results = Array{Array{Float64, 2}, 1}(undef, nb);

    Threads.@threads for t = 1:nb # so the N blocks will share the (nthreads - N) BLAS threads

        lods_currBlock = Array{Float64, 2}(undef, p, len);

        # process every trait in the block by a @simd loop 
        @simd for i = 1:len
            j = i+(t-1)*len;

            @inbounds lods_currBlock[:, i] = scan_lite_univar(Y0[:, j], X0_intercept, X0_covar, lambda0; reml = reml);
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

        lods_remBlock[:, i] = scan_lite_univar(Y0[:, j], X0_intercept, X0_covar, lambda0;
                   reml = reml);

    end

    LODs_all = hcat(LODs_all, lods_remBlock);

    return LODs_all

end 


###### Given the heritability (hsq), compute all LOD scores with performing LiteQTL once.


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

"""
scan_lite(Y0, X0, hsq, lambda0)

Calculates LOD scores for all pairs of traits and markers with a given hsq estimate.

# Arguments
- Y0 = 2d Array of Float; rotated traits 
- X0 = 2d Array of Float; rotated genotype probabilities
- hsq = Float; heritability
- lambda0 = 1d Array of Float; eigenvalues of the kinship matrix


# Value

- R = 2d Array of Float; matrix of LOD scores for all traits and markers calculated under the given heritability

# Notes:

Inputs data are assumed to be rotated.

"""
function scan_lite(Y0::Array{Float64, 2}, X0::Array{Float64, 2}, 
    hsq::Float64, lambda0::Array{Float64, 1})

    n = size(Y0, 1)
    sqrtw = sqrt.(makeweights(hsq, lambda0));

    wY0 = rowMultiply(Y0, sqrtw);
    wX0 = rowMultiply(X0, sqrtw);

    wX0_intercept = reshape(wX0[:, 1], :, 1);
    wX0_covar = wX0[:, 2:end];

    R = computeR_LMM(wY0, wX0_covar, wX0_intercept);

    tR2LOD!(R, n); # results will be p-by-1, i.e. all LOD scores for the j-th trait and p markers

    return R
end



"""
bulkscan(Y, G, K, hsq_list)

Calculates LOD scores for all pairs of traits and markers for each heritability in the supplied list, and returns the 
    maximal LOD scores for each pair among all calculated ones

# Arguments
- Y = 2d Array of Float; traits 
- G = 2d Array of Float; genotype probabilities
- K = 2d Array of Floatl kinship matrix
- hsq_list = 1d array of Float; the list of heritabilities requested to choose from

# Value

- R = 2d Array of Float; matrix of LOD scores for all traits and markers estimated
# Notes:

Maximal LOD scores are taken independently for each pair of trait and marker; while the number of candidated hsq's are finite,
    doing such maximization is like performing maximum-likelihood approach on discrete values for the heritability parameter;
    this is a shortcut of doing the exact scan_alt() independently for each trait and each marker.

"""

function bulkscan(Y::Array{Float64, 2}, G::Array{Float64, 2}, K::Array{Float64, 2}, hsq_list::Array{Float64, 1})

    (Y0, X0, lambda0) = transform_rotation(Y, G, K);

    maxL = scan_lite(Y0, X0, hsq_list[1], lambda0);

    for hsq in hsq_list[2:end]

        currL = scan_lite(Y0, X0, hsq, lambda0);
        tmax!(maxL, currL);

    end

    return maxL

end

