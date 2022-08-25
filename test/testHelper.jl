function testHelper(test::Expr; ntests::Integer = 10)
    [eval(test) for i in 1:ntests]
end