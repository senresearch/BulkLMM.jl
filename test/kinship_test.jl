## Tests for function to construct kinship:


kinship_ref = Helium.readhe(joinpath(@__DIR__, "ref_data_for_tests", "K_ref.he");)
println("Kinship test: ", @test kinship == kinship_ref);