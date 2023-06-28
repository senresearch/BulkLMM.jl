# function to minimize
f(x) = -(x^3+0.2*(x-2)^2+3)
# tolerance
tol = 1e-6

test_gridbrent = quote
    @test abs(gridbrent(f,-3.0,1.0,100).minimizer - 1.0) <= tol
end

println("Brent's method test: ", 
eval(test_gridbrent)
)
