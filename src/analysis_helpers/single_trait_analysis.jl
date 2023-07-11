## Helper functions for conducting genome-wide assoication analysis of a single trait

### Function to compute thresholds from permutation testing results:
### Inputs: 
###     - nperms_results - A matrix of LOD scores, each column contains the LOD scores 
###       fitted for each permuted copy;
###     - probs - A list contains the requested estimated (right-tail) probabilities 
###       the thresholds correspond to.
### Outputs:
###     - An object containing the quantiles of maximal LOD scores among all maximal 
###       LOD scores for all permutations and the corresponding probabilities . 

function get_thresholds(L::Array{Float64, 2}, signif_level::Array{Float64, 1})

    # Get the LOD score peak for each trait
    peak_each_trait = vec(mapslices(x -> maximum(x), L; dims = 1));
    # Thresholds will be determined by the quantiles of the peaks
    thr_probs = 1 .- signif_level;
    thrs = map(x -> quantile(peak_each_trait, x), thr_probs);

    return (probs = thr_probs, thrs = thrs);

end

## Function to compute the loglikelihood value of the given data under LMM model:
## Inputs: data after rotation, a given h2 to evaluate loglik on
##         (optional) prior for regularization loglik near the upper boundary pt.
## Outputs: the logliks (null, alt mean model) under the given h2
function getLL(y0::Array{Float64, 2}, X0::Array{Float64, 2}, lambda0::Array{Float64, 1},
               num_of_covar::Int64, 
               markerID::Int64, h2::Float64; prior::Array{Float64, 1} = [0.0, 0.0])
    
    n = size(y0, 1);
    w = makeweights(h2, lambda0);

    if num_of_covar == 1
        X0_covar = reshape(X0[:, 1], :, 1);
    else
        X0_covar = X0[:, 1:num_of_covar];
    end

    X_design = zeros(n, num_of_covar+1);
    X_design[:, 1:num_of_covar] = X0_covar;
    X_design[:, num_of_covar+1] = X0[:, markerID+num_of_covar];
    
    return (ll_null = wls(y0, X0_covar, w, prior).ell, ll_markerID = wls(y0, X_design, w, prior).ell)
end

function profileLL(y::Array{Float64, 2}, G::Array{Float64, 2}, covar::Array{Float64, 2}, 
                   K::Array{Float64, 2}, 
                   h2_grid::Array{Float64, 1}, markerID::Int64;
                   prior::Array{Float64, 1} = [0.0, 0.0])

    ## Initiate the vector to store the profile likelihood values evaluated under each given parameter value
    ell_null = zeros(length(h2_grid)); # loglikelihood under null
    ell_alt = zeros(length(h2_grid)); # loglikelihood under alternative


    num_of_covar = size(covar, 2);
    (y0, X0, lambda0) = transform_rotation(y, [covar G], K; addIntercept = false);

    ## Loop through the supplied h2 values, evaluate the profile loglik under each h2
    for k in 1:length(h2_grid)
        curr_h2 = h2_grid[k];
        output = getLL(y0, X0, lambda0, num_of_covar, markerID, curr_h2; prior = prior);
        ell_null[k] = output.ll_null;
        ell_alt[k] = output.ll_markerID;
    end

    ## Return values will be two (null, alternative models) lists of all loglikelihood values evaluated
    return (ll_list_null = ell_null, ll_list_alt = ell_alt);

end