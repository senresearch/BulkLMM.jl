


function r2lod(r::Float64, n::Int64)
    rsq = (r/n)^2
    return -(n/2.0) * log10(1.0-rsq);
end






function scan_lite_univar(y0_j::Array{Float64, 1}, X0_intercept::Array{Float64, 2}, 
    X0_covar::Array{Float64, 2}, lambda0::Array{Float64, 1};
    reml::Bool = true)

    n = size(y0_j, 1);

    y0 = reshape(y0_j, :, 1);
    vc = fitlmm(y0, X0_intercept, lambda0; reml = reml);
    sqrtw = sqrt.(makeweights(vc.h2, lambda0));

    wy0 = rowMultiply(y0, sqrtw);
    wX0_intercept = rowMultiply(X0_intercept, sqrtw);
    wX0_covar = rowMultiply(X0_covar, sqrtw);

    y00 = resid(wy0, wX0_intercept);
    X00 = resid(wX0_covar, wX0_intercept);


    sy = std(y00, dims = 1) |> vec;
    sx = std(X00, dims = 1) |> vec;
    colDivide!(y00, sy);
    colDivide!(X00, sx);

    R = X00' * y00; # p-by-1 matrix

    return r2lod.(R, n); # results will be p-by-1, i.e. all LOD scores for the j-th trait and p markers

end




function scan_lite_multithreads(Y::Array{Float64, 2}, G::Array{Float64, 2}, K::Array{Float64, 2}, nb::Int64;
    reml::Bool = true)


    (n, m) = size(Y);
    p = size(G, 2);

    # rotate data
    (Y0, X0, lambda0) = transform_rotation(Y, G, K);
    X0_intercept = reshape(X0[:, 1], :, 1);
    X0_covar = X0[:, 2:end];

    (len, rem) = divrem(m, nb);

    results = Array{Array{Float64, 2}, 1}(undef, nb);

    Threads.@threads for t = 1:nb # so the N blocks will share the (nthreads - N) BLAS threads

    lods_currBlock = Array{Float64, 2}(undef, p, len);

    @simd for i = 1:len
        j = i+(t-1)*len;
        #@inbounds
        lods_currBlock[:, i] = scan_lite_univar(Y0[:, j], X0_intercept, X0_covar, lambda0; reml = reml);
    end

    results[t] = lods_currBlock;

    end

    # process up the remaining data
    lods_remBlock = Array{Float64, 2}(undef, p, rem);

    for i in 1:rem

        j = m-rem+i;

        lods_remBlock[:, i] = scan_lite_univar(Y0[:, j], X0_intercept, X0_covar, lambda0;
                   reml = reml);

    end

    LODs_all = reduce(hcat, results);
    LODs_all = hcat(LODs_all, lods_remBlock);

    return LODs_all

end 