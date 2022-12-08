function testHelper(test::Expr; ntests::Integer = 10)
    [eval(test) for i in 1:ntests]
end

## Helper functions for comparing results:
function maxSqDiff(a::Array{Float64, 2}, b::Array{Float64, 2})

    return maximum((a .- b) .^2)

end

function sumSqDiff(a::Array{Float64, 2}, b::Array{Float64, 2})

    return sum((a .- b) .^2)

end


## Conversion functions between LOD scores and LR test p-values, for comparing with GEMMA results.
function p2lod(pval::Float64, df::Int64)
    
    lrs = invlogccdf(Chisq(df), log(pval))
    lod = lrs/(2*log(10))
    
    # return lrs
    return lod

end

function lod2p(lod::Float64, df::Int64)

    lrs = 2*log(10)*lod
    p = ccdf(Chisq(df), lrs);

    # return 
    return p

end