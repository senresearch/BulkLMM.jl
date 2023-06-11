###############################################################
# function to calculate kinship from genotype probability array
###############################################################
function calcKinship(geno::Array{Float64, 2})

    (n, p) = size(geno); # n - the sample size; p - the number of tested markers

    K = ones(n, n);

    for i in 1:(n-1)
        for j in (i+1):n

            K[i, j] = K[j, i] = calcCorr_IBD(vec(geno[i, :]), vec(geno[j, :]));

        end
    end

    return K;

end

function calcCorr_IBD(vg_i::Array{Float64, 1}, vg_j::Array{Float64, 1})

    x_i = 0.5 .- vg_i;
    x_j = 0.5 .- vg_j;

    return mean(0.5 .+ 2 .* (x_i .* x_j))

end