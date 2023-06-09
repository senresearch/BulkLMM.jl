###############################################################
# function to calculate kinship from genotype probability array
###############################################################

function calcKinship(geno::Matrix{Float64})

    # get dimensions
    sz = size(geno)

    # assign to variables for convenience
    nr = sz[1]
    nc = sz[2]

    # if empty then there is nothing to do
    if(nr==0)
        error("Nothing to do here.")
    else
        # make matrix to hold distances
        d = zeros(nr,nr)
    end

    # assign diagonals to ones
    for i=1:nr
        d[i,i] = 1.0
    end

    ncomplete = nc
    # off-diagonal elements need to be calculated
    if(nr>=2)
        for i=1:(nr-1)
            for j=(i+1):nr
                p1 = geno[i,:]
                p2 = geno[j,:]
                d[i,j] = d[j,i] = sum( p1 .* p2 + (1 .- p1) .* (1 .- p2) ) / ncomplete
            end
        end
    end
    return d
end

function calcKinship2(geno::Array{Float64, 2})

    (n, p) = size(geno); # n - the sample size; p - the number of tested markers

    K = ones(n, n);

    for i in 1:(n-1)
        for j in (i+1):n

            K[i, j] = calcCorr_IBD(vec(geno[i, :]), vec(geno[j, :]));

        end
    end

    return K;

end

function calcCorr_IBD(vg_i::Array{Float64, 1}, vg_j::Array{Float64, 1})

    x_i = 0.5 .- vg_i;
    x_j = 0.5 .- vg_j;

    return sum(0.5 .+ 2 .* (x_i .* x_j))

end