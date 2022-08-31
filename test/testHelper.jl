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