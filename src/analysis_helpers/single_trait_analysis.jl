## Helper functions for conducting genome-wide assoication analysis of a single trait

struct LODthresholds
    probs::Array{Float64, 1};
    thrs::Array{Float64, 1}
end

### Function to compute thresholds from permutation testing results:
### Inputs: 
###     - nperms_results - A matrix of LOD scores, each column contains the LOD scores 
###       fitted for each permuted copy;
###     - probs - A list contains the requested estimated (right-tail) probabilities 
###       the thresholds correspond to.
### Outputs:
###     - An object containing the quantiles of maximal LOD scores among all maximal 
###       LOD scores for all permutations and the corresponding probabilities.

function get_thresholds(nperms_results::Array{Float64, 2}, probs::Array{Float64, 1})

    max_lods_each_perm = vec(mapslices(x -> maximum(x), nperms_results; dims = 1));
    thrs = map(x -> quantile(max_lods_each_perm, x), probs);

    return LODthresholds(probs, thrs);

end