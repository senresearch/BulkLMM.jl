###############################################################
# function to calculate kinship from genotype probability array
###############################################################
function calcKinship(geno::Array{Float64, 2})

    (n, p) = size(geno); # n - the sample size; p - the number of tested markers

    X = geno .- 0.5;
    K = 2 .* (X*X')./size(X, 2) .+ 0.5;
    K[diagind(K)] .= 1.0

    return K;

end