## Tests for function to construct kinship:


## kinship_ref created from calling calcKinship from Julia on Beale
kinship_ref = Helium.readhe(joinpath(@__DIR__, "ref_data_for_tests", "kinship_ref.he")) |> x -> round.(x, digits = 13);
# println("Max abs diff: ", maximum(abs.(kinship_test .- kinship_ref)))
println("Kinship test: ", @test kinship == kinship_ref)

test_perms = scan(pheno_y, geno, kinship; permutation_test = true).L_perms;
ref_perms = scan(pheno_y, geno, kinship_ref; permutation_test = true).L_perms;
println("Permutation LODs test: ", @test test_perms == ref_perms)

test_thrs = BulkLMM.get_thresholds(test_perms, [0.90, 0.95]).thrs;
ref_thrs = BulkLMM.get_thresholds(ref_perms, [0.90, 0.95]).thrs;
println("Permutation thresholds test: ", @test test_thrs == ref_thrs)