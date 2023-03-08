# Test analysis helper functions:

##########################################################################################################
## TEST:  getLL()
##########################################################################################################

pheno_id = 1997;
pheno_y = reshape(pheno[:, pheno_id], :, 1);

(y0, X0, lambda0) = BulkLMM.transform_rotation(pheno_y, geno, kinship; addIntercept = true);

test_h2 = 0.5;
markerID = 7321;
prior = [1.0, 0.1];
tol = 1e-6;

test_getLL = quote 
    ll_results = BulkLMM.getLL(y0, X0, lambda0, markerID, test_h2; prior = prior);

    w_test = BulkLMM.makeweights(h2, lambda0);

    ll_null_test = BulkLMM.wls(y0, reshape(X0[:, 1], :, 1), w_test, prior).ell;
    ll_alt_test = BulkLMM.wls(y0, X0[:, [1, markerID+1]], w_test, prior).ell;

    @test abs(ll_results.ll_null - ll_null_test) <= tol;
    @test abs(ll_results.ll_markerID - ll_alt_test) <= tol;
end

##########################################################################################################
## TEST:  get_threshold()
##########################################################################################################
test_getThreshold = quote 

    perms_results = scan(pheno_y, geno, kinship; permutation_test = true, nperms = 100, original = false);
    probs = [0.5];

    max_lods = zeros(size(perms_results, 2));
    for i in 1:size(perms_results, 2)
        max_lods[i] = maximum(perms_results[:, i]);
    end

    thr_obj = BulkLMM.get_thresholds(perms_results, probs);

    @test quantile(max_lods, thr_obj.probs[1]) == thr_obj.thrs[1];

end


##########################################################################################################
## TEST: run all tests
##########################################################################################################
@testset "Test Analysis Helpers" begin
    eval(test_getLL);
    eval(test_getThreshold);
end