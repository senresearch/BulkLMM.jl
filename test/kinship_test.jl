## Tests for function to construct kinship:


## kinship_ref created from calling calcKinship from Julia on Beale
kinship_ref = Helium.readhe(joinpath(@__DIR__, "ref_data_for_tests", "kinship_ref.he"))
println("Max abs diff: ", maximum(abs.(kinship .- kinship_ref)))
println("Kinship test: ", @test kinship == kinship_ref)