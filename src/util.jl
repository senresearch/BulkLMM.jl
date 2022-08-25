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

    for i in 1:length(x)
        if x[i] == 0.0
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

function rowDivide!(A::Matrix{Float64}, x::Vector{Float64})

    (n,m) = size(A)

    # Checking validity of inputs:
    ## Checking dimensions of inputs
    if(length(x)!=n)
        throw(error("Matrix and vector size do not match."))
    end

    # Checking if dividing by zeros
    if(checkZeros(x))
        throw(error("Dividing by zeros: the input vector can not contain any zeros!"))
    end

    for i=1:m
        for j=1:n
            @inbounds A[j,i] = A[j,i]/x[j]
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

## Compare two arrays and return the number of elements with the same indices of each array and are match
function compareValues(x_true::Array{Float64,1}, x::Array{Float64,1}, tolerance::Float64, threshold::Float64)
    if size(x_true) != size(x)
        throw(error("Dimention Mismatch! Must compare two arrays of same length!"))
    end

    passes = falses(size(x_true))
    t_passes = falses(0)
    for i in 1:size(x_true)[1]
        e = abs(x[i]-x_true[i])
        if e <= tolerance
            passes[i] = true
        end

        if x[i] >= threshold
            if e <= tolerance
                push!(t_passes, true)
            else
                push!(t_passes,false)
            end
        end

    end
    # pass_rate = sum(passes) / size(x_true)[1]
    # pass_rate = sum(t_passes) / size(t_passes)[1]

    return (sum(passes), size(t_passes)[1], sum(t_passes))

end
