
function runtime(funlist::Array{String, 1}, n::Int64)
    # number of functions to be compared
    nf = length(funlist)

    elapsedTimes = Array{Float64, 2}(undef, n, nf);
    medianTimes = Array{Float64, 1}(undef, nf);
    idx = 1

    for i = 1:n # do n times running each function
        for j = 1:nf # run each function
            t0 =time()
            eval(funlist[j])
            t1 = time()
            elapsedTimes[i, j] = t1-t0
        end
    end

    for j in 1:nf
        medianTimes[j] = median(elapsedTimes[:, j])
    end

    return elapsedTimes, medianTimes
end
