## Helper functions for conducting genome-wide assoication analysis of a single trait

struct LODthresholds
    thr_probs::Array{Float64, 1};
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
###       LOD scores for all permutations and the corresponding probabilities . 

function get_thresholds(nperms_results::Array{Float64, 2}, thr_probs::Array{Float64, 1})

    max_lods_each_perm = vec(mapslices(x -> maximum(x), nperms_results; dims = 1));
    thrs = map(x -> quantile(max_lods_each_perm, x), thr_probs);

    return LODthresholds(thr_probs, thrs);

end

## Function to compute the loglikelihood value of the given data under LMM model:
## Inputs: data after rotation, a given h2 to evaluate loglik on
##         (optional) prior for regularization loglik near the upper boundary pt.
## Outputs: the logliks (null, alt mean model) under the given h2
function getLL(y0::Array{Float64, 2}, X0::Array{Float64, 2}, lambda0::Array{Float64, 1}, 
               markerID::Int64, h2::Float64; prior::Array{Float64, 1} = [0.0, 0.0])
    
    n = size(y0, 1);
    w = makeweights(h2, lambda0);

    X0_inter = reshape(X0[:, 1], :, 1);
    X_design = zeros(n, 2);
    X_design[:, 1] = X0[:, 1];
    X_design[:, 2] = X0[:, markerID+1];
    
    return (ll_null = wls(y0, X0_inter, w, prior).ell, ll_markerID = wls(y0, X_design, w, prior).ell)
end

function plotLL(y::Array{Float64, 2}, G::Array{Float64, 2}, K::Array{Float64, 2}, 
                h2_grid::Array{Float64, 1}, markerID::Int64;
                x_lims::Array{Float64, 1}, y_lims::Array{Float64, 1})


    ell_null = zeros(length(h2_grid));
    ell_alt = zeros(length(h2_grid));


    (y0, X0, lambda0) = transform_rotation(y, G, K; addIntercept = true);

    for k in 1:length(h2_grid)
        curr_h2 = h2_grid[k];
        output = getLL(y0, X0, lambda0, markerID, curr_h2; prior = prior);
        ell_null[k] = output.ll_null;
        ell_alt[k] = output.ll_markerID;
    end

    opt_ll_null = findmax(ell_null)[1];
    opt_h2_null = h2_grid[findmax(ell_null)[2]];
    opt_ll_alt = findmax(ell_alt)[1];
    opt_h2_alt = h2_grid[findmax(ell_alt)[2]];

    p = plot(h2_grid, ell_null, xlabel = "h2", ylabel = "loglik", label = "Null", color = "blue", legend=:bottomleft)
    scatter!(p, [opt_h2_null], [opt_ll_null], label = "maxLL_null", color = "blue")
    plot!(p, h2_grid, ell_alt, xlabel = "h2", ylabel = "loglik", label = ("Alt_$markerID"), color = "red")
    scatter!(p, [opt_h2_alt], [opt_ll_alt], label = "maxLL_alt", color = "red");

    plot!(p, ones(2).*opt_h2_null, [y_lims[1]-0.05, opt_ll_null], color = "blue", style = :dash, label = "")
    plot!(p, [x_lims[1]-0.05, opt_h2_null], ones(2).*opt_ll_null, color = "blue", style = :dash, label = "")
    annotate!(p, opt_h2_null, y_lims[1]-0.05, 
              text("$opt_h2_null", :blue, :below, 8))

    plot!(p, ones(2).*opt_h2_alt, [y_lims[1]-0.05, opt_ll_alt], color = "red", style = :dash, label = "")
    plot!(p, [x_lims[1]-0.05, opt_h2_alt], ones(2).*opt_ll_alt, color = "red", style = :dash, label = "")
    annotate!(p, opt_h2_alt, y_lims[1]-0.05, 
              text("$opt_h2_alt", :red, :below, 8))



    xlims!(p, (x_lims[1]-0.05, x_lims[2]+0.05))
    ylims!(p, (y_lims[1]-0.05), y_lims[2]+0.05)
    #= 
    ylims!(p, (minimum([y_lims[1], minimum(ell_null), minimum(ell_alt)])-0.05, 
               maximum([y_lims[2], maximum(ell_null), maximum(ell_alt)])+0.05))

    =#
    return p

end