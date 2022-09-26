
# Inputs: total number of markers to be divided, number of blocks required
# Outputs: a list of ranges for every block
# 
function createBlocks(nmarkers::Int64, nblocks::Int64)

    block_size = floor(Int, nmarkers/nblocks);

    blocks = UnitRange{Int64}[];

    # Interate to create the bounds of each block
    for k = 1:nblocks
        id_start::Int = 1 + block_size * (k - 1);
        id_end::Int = id_start - 1 + block_size;

        # for the last block, (remainder)
        if k == nblocks
            if id_end < nmarkers
                id_end = nmarkers
            end
        end

        push!(blocks, id_start:id_end)
    end
        
    return blocks

end

function createBlocks2(nmarkers::Int64, block_size::Int64)

    nblocks = ceil(Int, nmarkers/block_size);

    blocks = UnitRange{Int64}[];

    # Interate to create the bounds of each block
    for k = 1:nblocks
        id_start::Int = 1 + block_size * (k - 1);
        id_end::Int = id_start - 1 + block_size;

        if id_end > nmarkers
            id_end = nmarkers
        end

        push!(blocks, id_start:id_end)
    end
        
    return blocks

end

# Inputs: r0perm, X00, number of blocks required
# Outputs: a matrix which is the hcat of LOD scores of all the blocks
# calculate the results of every block distributedly
function calcLODs_block(r0perm::Array{Float64, 2}, X00::Array{Float64, 2}, blockRange::UnitRange{Int64})
    # Given a block of markers, return the LOD scores of all markers in the block

    (n, p) = size(X00); # n - number of observations; p - number of markers
    np = size(r0perm, 2); # may include the original

    rss0 = sum(r0perm[:, 1].^2) # a scalar; bc rss0 for every permuted trait is the same under the null (zero mean);

    ## make array to hold Alternative RSS's for each permutated trait
    rss1_i = Array{Float64, 2}(undef, np, length(collect(blockRange)))

    if blockRange.start < 1 || blockRange.stop > p
        throw(error("Block is out of range of the input markers data."))
    end

    iter = 1
    for j in blockRange

        ## alternative rss
        @inbounds rss1_i[:, iter] = rss(r0perm, @view X00[:, j]);
        iter = iter + 1;

    end

 
    lod_i = (-n/2)*(log10.(rss1_i) .- log10(rss0))
 
    return lod_i
end

function calcLODs_perms(r0::Array{Float64, 2}, X00::Array{Float64, 2}, nperms::Int64, rndseed::Int64)

    r0perm = transform3(r0; nperms = nperms, rndseed = rndseed, original = false);

    (n, m) = size(X00);

    rss0 = sum(r0perm[:, 1].^2) # a scalar; bc rss0 for every permuted trait is the same under the null (zero mean);

    rss1 = Array{Float64, 2}(undef, nperms, m) 

    ## loop over markers
    for i = 1:m

        ## alternative rss
        rss1[:, i] = rss(r0perm, @view X00[:, i]);

    end

    lod = (-n/2)*(log10.(rss1) .- log10(rss0))

    return lod

end

function transform1(y::Array{Float64, 2}, g::Array{Float64, 2}, K::Array{Float64, 2})
        
    # n - the sample sizes
    n = size(g, 1);

    # make intercept
    intercept = ones(n, 1);
    # rotate data so errors are uncorrelated
    (y0, X0, lambda0) = rotateData(y, [intercept g], K);

    return (y0, X0, lambda0);

end

# Take the rotated data
function transform2(y0::Array{Float64, 2}, X0::Array{Float64, 2}, lambda0::Array{Float64, 1};
                    reml::Bool = false)

        ## Note: estimate once the variance components from the null model and use for all marker scans
        # fit lmm
        vc = fitlmm(y0, reshape(X0[:, 1], :, 1), lambda0; reml = reml)
        # println(vc) # vc.b is estimated through weighted least square
        r0 = y0 - X0[:, 1]*vc.b

        # weights inversely-proportional to the variances
        sqrtw = sqrt.(makeweights(vc.h2, lambda0))

        copy_X0 = copy(X0);
        copy_r0 = copy(r0);
        # rescale by weights; now these have the same mean/variance and are independent
        ## NOTE: although rowDivide! makes in-place changes to the inputs, it only modifies the rotated data which are returned outputs
        rowMultiply!(copy_r0, sqrtw) # dont want to change the inputs: r0, X0
        rowMultiply!(copy_X0, sqrtw) # dont want to change the inputs: r0, X0
        X00 = resid(copy_X0[:, 2:end], reshape(copy_X0[:, 1], :, 1)) # after re-weighting X, calling resid on re-weighted X is the same as doing wls too.

        return (copy_r0, X00)

end

function transform3(r0::Array{Float64, 2}; 
                    nperms::Int64 = 1024, rndseed::Int64 = 0, original::Bool = true)

        ## random permutations; the first column is the original data
        rng = MersenneTwister(rndseed);
        r0perm = shuffleVector(rng, r0[:, 1], nperms; original = original) # permutation on r0 which have iid standard normal distribution under null
    
        return r0perm
end