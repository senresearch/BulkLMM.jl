## Tests for function to construct kinship:


## kinship_ref created from calling calcKinship from Julia on Beale
kinship_ref = Helium.readhe(joinpath(@__DIR__, "ref_data_for_tests", "kinship_ref.he"))
kinship_test = Helium.readhe(joinpath(@__DIR__, "ref_data_for_tests", "kinship_test.he"))
# println("Max abs diff: ", maximum(abs.(kinship_test .- kinship_ref)))
println("Kinship test: ", @test kinship_test == kinship_ref)