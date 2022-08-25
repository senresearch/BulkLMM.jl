# Util Functions Tests

## Loading required packages:
using Test
using BenchmarkTools
using LinearAlgebra
using Statistics
using Random

## Loading functions to be tested:
include("../src/util.jl")
include("testHelper.jl")


##########################################################################################################
## TEST: colCenter!()
##########################################################################################################


### When data is a vector (1-dim array)
test1_colCenter = quote
    N = 100;
    A = rand([0.0, 1.0], N, 1);
    ## Manually center columns of A
    colMean_A = sum(A)/N;
    true_centered_A = transpose(round.(transpose(A) .- colMean_A; digits = 2));
    colCenter!(A);
    A .= round.(A; digits = 2);
    @test true_centered_A == A;
end;

### When data is a matrix (2-dim array)
test2_colCenter = quote
    N = 100;
    p = rand([1, 5]); # randomly generate the number of rows
    
    A = rand([0.0, 1.0], N, p);
    
    ## Manually center columns of A
    colMean_A = sum(A, dims = 1)./N;
    true_centered_A = round.(A .- colMean_A; digits = 2);
    colCenter!(A);
    A .= round.(A; digits = 2);
    @test true_centered_A == A;
end;

tests_colCenter = quote
    testHelper(test1_colCenter);
    testHelper(test2_colCenter);
end;


##########################################################################################################
## TEST: rowCenter!()
##########################################################################################################

### When data is a vector (has only one element in each row, should throw an error)
test1_rowCenter = quote
    N = 100;
    p = 1; # randomly generate the number of rows
    A = rand([0.0, 1.0], N, p);

    try 
        rowCenter!(A)
    catch e
        @test typeof(e) == ErrorException
    end
end;

### When data is a matrix (2-dim array)
test2_rowCenter = quote
    N = 100;
    p = rand([2, 5]); # randomly generate the number of rows
    
    A = rand([0.0, 1.0], N, p);
    
    ## Manually center columns of A
    rowMean_A = sum(A, dims = 2)./p;
    true_centered_A = round.(A .- rowMean_A; digits = 2);
    
    rowCenter!(A);
    A .= round.(A; digits = 2);
    
    @test true_centered_A == A;
end;

tests_rowCenter = quote
    testHelper(test1_rowCenter);
    testHelper(test2_rowCenter);
end;

##########################################################################################################
## TEST: colDivide!()
##########################################################################################################

### Check if dividing by zeros
test1_colDivide = quote
    N = 100;
    p = rand([1, 5]); # randomly generate the number of rows
    A = rand([0.0, 1.0], N, p);

    w = 1.0*ones(p);
    w[length(w)] = 0.0;

    try 
        colDivide!(A, w)
    catch e
        @test typeof(e) == ErrorException
    end
end;

### Normal case test:
test2_colDivide = quote
    N = 100;
    p = rand([1, 5]); # randomly generate the number of rows
    A = rand([0.0, 1.0], N, p);
    
    colVars = mapslices(var, A, dims = 1) |> vec;
    true_weighted_A = transpose(round.(transpose(A) ./ colVars; digits = 2));
    
    colDivide!(A, colVars);
    A = round.(A; digits = 2);
    
    @test true_weighted_A == A;
end;

tests_colDivide = quote
    testHelper(test1_colDivide);
    testHelper(test2_colDivide);
end;

##########################################################################################################
## TEST: rowDivide!()
##########################################################################################################

### Check if dividing by zeros
test1_rowDivide = quote
    N = 100;
    p = rand([1, 5]); # randomly generate the number of rows
    A = rand([0.0, 1.0], N, p);

    w = 1.0*ones(N);
    w[length(w)] = 0.0;

    try 
        rowDivide!(A, w)
    catch e
        @test typeof(e) == ErrorException
    end
end;

### Normal case test:
test2_rowDivide = quote
    N = 100;
    p = rand([2, 5]); # randomly generate the number of rows
    A = rand([0.0, 1.0], N, p);
    
    rowVars = mapslices(var, A, dims = 2) |> vec;
    rowVars = round.(rowVars; digits = 2);
    
    for i in 1:length(rowVars)
        if(rowVars[i] == 0.0)
            rowVars[i] = 1.0
        end
    end
    
    true_weighted_A = round.(A ./ rowVars; digits = 2);
    
    rowDivide!(A, rowVars);
    A = round.(A; digits = 2);
    
    @test true_weighted_A == A;
end;

tests_rowDivide = quote
    testHelper(test1_rowDivide);
    testHelper(test2_rowDivide);
end;

##########################################################################################################
## TEST: colStandardize()
##########################################################################################################

test_colStandardize = quote
    N = 100;
    p = rand([1, 5]); # randomly generate the number of rows
    A = rand([0.0, 1.0], N, p);
    
    colMeans = mean(A, dims = 1) |> vec; # mean of each column
    colVars = mapslices(std, A, dims = 1) |> vec; # std of each column
    true_standardized_A = round.(transpose(A) .- colMeans; digits = 2)
    true_standardized_A = transpose(round.(true_standardized_A ./ colVars; digits = 2));
    
    colStandardize!(A);
    A = round.(A; digits = 2);
    
    @test true_standardized_A == A;
end;

tests_colStandardize = quote
    testHelper(test_colStandardize);
end;

##########################################################################################################
## TEST: rowMultiply()
##########################################################################################################

### Test1: check if dimensions match
test1_rowMultiply = quote 

    N = 100;
    p = rand([2, 5]); # randomly generate the number of rows
    A = rand([0.0, 1.0], N, p);

    x = zeros(N+1)

    try 
        rowMultiply(A, x)
    catch e
        @test typeof(e) == ErrorException
    end

end;

### Test2: check if results match
test2_rowMultiply = quote
    N = 200;
    p = 100;
    A = rand(N, p);
    
    w = rand(N)
    
    true_weighted_A = round.(A .* w; digits = 2);
    
    test_weighted_A = round.(rowMultiply(A, w); digits = 2)

    
    @test true_weighted_A == test_weighted_A;
end;

### Test3: check if the original is not modified
test3_rowMultiply = quote
    N = 200;
    p = 100; # randomly generate the number of rows
    A = rand(N, p);
    
    w = rand(N)
    
    test_weighted_A = round.(rowMultiply(A, w); digits = 2)
    
    rowDivide!(A, 1.0 ./w)

    A = round.(A; digits = 2)
    
    @test A == test_weighted_A;
end;

tests_rowMultiply = quote
    testHelper(test1_rowMultiply);
    testHelper(test2_rowMultiply);
    testHelper(test3_rowMultiply);
end;

##########################################################################################################
## TEST: shuffleVector()
##########################################################################################################

### Test1: output dimension - when by default, the original is kept
test1_shuffleVector = quote
    N = 100;
    p = 1;
    A = rand([0.0, 1.0], N, p) |> vec; # fixed to be a vector
    rng = MersenneTwister();
    
    result = shuffleVector(rng, A, 5);
    
    
    @test size(result, 2) == 6;
end;

### Test2: output dimension - when passing original == false, the original is dropped
test2_shuffleVector = quote
    N = 100;
    p = 1;
    A = rand([0.0, 1.0], N, p) |> vec; # fixed to be a vector
    rng = MersenneTwister();
    
    result = shuffleVector(rng, A, 5; original = false);
    
    
    @test size(result, 2) == 5;
end;

### Test3: check if the permuted array preserve the same set of elements
test3_shuffleVector = quote
    N = 100;
    p = 1;
    A = rand([0.0, 1.0], N, p) |> vec; # fixed to be a vector
    rng = MersenneTwister();
    
    result = shuffleVector(rng, A, 5);
    
    colSums = sum(result, dims = 1) |> vec;
    
    @test (colSums./colSums[1]) == ones(6);
end;

tests_shuffleVector = quote
    testHelper(test1_shuffleVector);
    testHelper(test2_shuffleVector);
    testHelper(test3_shuffleVector);
end;

##########################################################################################################
## TEST: compareValues()
##########################################################################################################

### Test1: check if dimensions match

test1_compareValues = quote 
    b_true = convert(Array{Float64, 1}, zeros(rand(collect(1:5))));
    b_test = convert(Array{Float64, 1}, zeros(length(b_true)+1));

    try 
        compareValues(b_true, b_test, 1e-3, 1e5)
    catch e
        @test typeof(e) == ErrorException
    end
end;

### Test2: check if the results are as desired

test2_compareValues = quote
    b_true = convert(Array{Float64, 1}, zeros(rand(collect(2:5))));
    b_test = copy(b_true);

    b_test[length(b_true)] = 1.0
    b_test[length(b_true)-1] = 2.0

    @test compareValues(b_test, b_true, 0.0, 1e5)[1] == length(b_true) - 2
end;

tests_compareValues = quote
    testHelper(test1_compareValues);
    testHelper(test2_compareValues);
end;

@testset "Utility Functions Tests" begin

    eval(tests_colCenter);
    eval(tests_colDivide);
    eval(tests_colStandardize);
    eval(tests_rowCenter);
    eval(tests_rowDivide);
    eval(tests_rowMultiply);
    eval(tests_shuffleVector);
    eval(tests_compareValues);

end;

