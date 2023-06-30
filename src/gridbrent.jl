"""
    gridbrent(f::Function,a::Float64,b::Float64,ninterval::Int64=1)

    f: univariate function to be minimized
    a: lower limit of interval to be searched
    b: upper limit of interval to be searched
    ninterval: number of subintervals to be searched
"""
function gridbrent(f::Function,a::Float64,b::Float64,ninterval::Int64=1)
    # calculate points spanning the interval
    points = collect(range(a,b,length=ninterval+1))
    # left and right endpoints of subintervals
    av = points[1:ninterval]
    bv = points[2:(ninterval+1)]
    # apply Brent's method to subintervals
    res = broadcast(optimize,f,av,bv,[Brent()])
    # find the minimum in each interval
    minimumv = Optim.minimum.(res)
    # find index of global minimum
    idx = argmin(minimumv)
    # return named array with minimum and minimizer
    res = (minimum=Optim.minimum(res[idx]),minimizer=Optim.minimizer(res[idx]))
    return res
end