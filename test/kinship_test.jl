## Tests for function to construct kinship:


# kinship_ref = Helium.readhe(joinpath(@__DIR__, "ref_data_for_tests", "kinship_ref.he"))
kinship_test = Helium.readhe(joinpath(@__DIR__, "ref_data_for_tests", "kinship_test.he"))
println("Kinship test: ", @test kinship == kinship_test);