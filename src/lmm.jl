##################################################################
# Fast linear mixed models
##################################################################
#
# We implement linear mixed models for data which has covariance of
# the form tau2*K + sigma2*I, where sigma2 and tau2 are positive
# scalars, K is a symmetric positive definite "kinship" matrix and I
# is the identity matrix.
#
# Models of this type are used for GWAS and QTL mapping in structured
# populations.
#
# ################################################################

function makeweights(h2::Float64, lambda::Array{Float64,1})

    delta = h2/(1-h2);

    if isinf(delta)
        throw(error("Heritability of 1 is not allowed."));
    end

    vars = (delta.*lambda .+ 1);

    if any(vars .<= 0.0)
        throw(error("Non-positive environmental variance is not allowed; check input values."));
    end

    return 1.0./vars

end


##################################################################
# function to fit linear mixed model by optimizing heritability
##################################################################
mutable struct LMMEstimates
    b::Array{Float64,2}
    sigma2::Float64
    h2::Float64
    ell::Float64
end


"""
fitlmm: fit linear mixed model

y: 2-d array of (rotated) phenotypes
X: 2-d array of (rotated) covariates
lambda: 1-d array of eigenvalues
reml: boolean indicating ML or REML estimation

"""
function fitlmm(y::Array{Float64, 2}, X::Array{Float64, 2}, lambda::Array{Float64, 1}, prior::Array{Float64, 1};
                reml::Bool = false, loglik::Bool = true, method::String = "qr", 
                h20::Float64 = 0.5, d::Float64 = 1.0)

    function logLik0(h2::Float64)
        out = wls(y, X, makeweights(h2, lambda), prior; reml = reml, loglik = loglik, method = method)
        return -out.ell
    end
    ## avoid the use of global variable in inner function;

    opt = optimize(logLik0, max(h20-d, 0.0), min(h20+d, 1.0))
    h2 = opt.minimizer
    est = wls(y, X, makeweights(h2, lambda), prior; reml = reml, loglik = loglik, method = method)
    return LMMEstimates(est.b, est.sigma2, h2, est.ell)
end
