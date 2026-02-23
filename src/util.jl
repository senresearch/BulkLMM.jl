###################################
# utility functions
###################################


# using DataStructures

# centers each column
function colCenter!(A::Matrix{Float64})

    (n,m) = size(A)

    if(n == 1)
        throw(error("Each column should contain at least two elements to average from!"))
    end

    # get mean of each column; convert to vector
    colMeans = mean(A,dims=1) |> vec

    for i=1:n
        for j=1:m
            A[i,j] = A[i,j] - colMeans[j]
        end
    end
end

# centers each row
function rowCenter!(A::Matrix{Float64})

    (n,m) = size(A)

    if(m == 1)
        throw(error("Each row should contain at least two elements to average from!"))
    end

    # get mean of each column; convert to vector
    rowMeans = mean(A,dims=2) |> vec

    for i=1:m
        for j=1:n
            A[j,i] = A[j,i] - rowMeans[j]
        end
    end
end

# a helper function that checks if any element in an 1-dimensional array is 0 (for use in colDivide! and rowDivide!)
function checkZeros(x::Vector{Float64})

    for i in x
        if isapprox(i, 0.0; atol=eps(Float64), rtol=0)
            return true
        end
    end

    return false
end

function colDivide!(A::Matrix{Float64}, x::Vector{Float64})

    (n,m) = size(A)

    # Checking validity of inputs:
    ## Checking dimensions of inputs
    if(length(x)!=m)
        throw(error("Matrix and vector size do not match."))
    end

    # Checking if dividing by zeros
    if(checkZeros(x))
        throw(error("Dividing by zeros: the input vector can not contain any zeros!"))
    end
    
    for i=1:n
        for j=1:m
            A[i,j] = A[i,j]/x[j]
        end
    end
end

function colStandardize!(A::Matrix{Float64})

    colCenter!(A)
    s = std(A,dims=1) |> vec
    colDivide!(A,s)

end

function colStandardize(A::Array{Float64, 2})

    sA = A .- mean(A; dims = 1);
    s = std(sA,dims=1) |> vec
    colDivide!(sA, s);

    return sA
    
end

function rowDivide!(A::Matrix{Float64}, x::Vector{Float64})

    (n, m) = size(A)

    # Checking validity of inputs:
    ## Checking dimensions of inputs
    if(length(x)!=n)
        throw(error("Matrix and vector size do not match."))
    end

    # Checking if dividing by zeros
    if(checkZeros(x))
        throw(error("Dividing by zeros: the input vector can not contain any zeros!"))
    end

    for i = 1:m
        for j = 1:n
            @inbounds A[j, i] = A[j, i]/x[j]
        end
    end

end

function rowMultiply!(A::Matrix{Float64}, x::Vector{Float64})

    (n, m) = size(A)

    # Checking validity of inputs:
    ## Checking dimensions of inputs
    if(length(x)!=n)
        throw(error("Matrix and vector size do not match."))
    end
    
    for i = 1:m
        for j = 1:n
            @inbounds A[j, i] = A[j, i] * x[j]
        end
    end

end

function rowMultiply(A::Matrix{Float64}, x::Vector{Float64})

    (n,m) = size(A)
    if(length(x)!=n)
        error("Matrix and vector size do not match.")
    end

    B = similar(A)
    
    for i=1:m
        for j=1:n
            @inbounds B[j,i] = A[j,i] * x[j]
        end
    end

    return B
    
end

"""
perform random shuffles of vector
the first column is the original vector if original=true
"""
function shuffleVector(rng::AbstractRNG, x::Vector{Float64}, nshuffle::Int64;
                       original::Bool = true)
    if(original)
        xx = zeros(length(x), nshuffle+1)
        xx[:,1] = x
        istart = 1
    else
        xx = zeros(length(x), nshuffle)

        istart = 0
    end

    for i = 1:nshuffle
        xx[:, i+istart] = shuffle(rng,x)
    end

    return xx
end

function p2lod(pval::Float64, df::Int64)
    
    lrs = invlogcdf(Chisq(df), log(1-pval))
    lod = lrs/(2*log(10))
    
    return lod

end

function lod2p(lod::Float64, df::Int64)
    
    lrs = lod*2*log(10);
    pval = ccdf(Chisq(df), lrs)
    
    return pval
    
end

function lod2log10p(lod::Float64, df::Int64)
    
    lrs = lod*2*log(10);
    logpval = logccdf(Chisq(df), lrs)
    
    return -logpval/log(10);
    
end

function calc_std_err(lod::Float64, 
                      beta_1::Float64)

    # Convert LOD score to LRT statistic:
    lrt = 2*log(10)*lod

    # Compute standard error from LRT and effect size:
    # (Based on the fact that LRT = squared t-statistic under simple linear regression)
    std_err = abs(beta_1)/sqrt(lrt)

    return std_err
    
end

function calc_p_wald(beta_1::Float64, se_beta_1::Float64)
    t_stats = (beta_1/se_beta_1)^2
    p_wald = ccdf(Chisq(1), t_stats)
    return p_wald
end
