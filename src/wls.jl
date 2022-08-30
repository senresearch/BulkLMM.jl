
##################################################################
# wls: weighted least squares
##################################################################

mutable struct LS_estimates
    b::Array{Float64, 2}
    sigma2::Float64
    ell::Float64
end


"""
wls: Weighted least squares estimation

y = outcome, matrix
X = predictors, matrix
w = weights (positive, inversely proportional to variance), one-dim vector

"""
function wls(y::Array{Float64, 2}, X::Array{Float64, 2}, w::Array{Float64, 1};
             reml::Bool = false, loglik::Bool = true, method = "qr")

    (n, p) = size(X); # get number of observations and the number of markers from geno dimensions      

    n = size(y, 1); # get number of observations       

    # check if weights are positive
    if(any(w .<= .0))
        error("Some weights are not positive.")
    end

    # square root of the weights
    sqrtw = sqrt.(w)

    logdetXtX = logdet(X' * X);

    # scale by weights
    yy = rowMultiply(y, sqrtw)
    XX = rowMultiply(X, sqrtw)

    # least squares solution
    # faster but numerically less stable
    if(method == "cholesky")
        fct = cholesky(XX'XX)
        b = fct\(XX'yy)
        logdetXXtXX = logdet(fct)
    end

    # slower but numerically more stable
    if(method == "qr")
        fct = qr(XX)
        b = fct\yy
        # logdetXXtXX = 2*logdet(fct.R) # need 2 for logdet(X'X)
        logdetXXtXX = logdet(fct.R' * fct.R);
        # println(logdetXXtXX) for testing
    end

    yyhat = XX*b
    rss0 = sum((yy-yyhat).^2)

    if(reml)
        sigma2 = rss0/(n-p)
    else
        sigma2 = rss0/n
    end

    # see formulas (2) and (3) of Kang (2008)
    if(loglik)
        # ell = -0.5 * ( n*log(sigma2) + sum(log.(w)) + rss0/sigma2 )
        ell = -0.5 * (n*log(sigma2) - sum(log.(w)) + rss0/sigma2)

        if(reml)
            ell = ell + 0.5 * (p*log(sigma2) + logdetXtX - logdetXXtXX)
            # ell = ell + 0.5 * (p*log(sigma2) - logdetXXtXX)
        end
        
    else
        ell = missing;
    end

    return LS_estimates(b, sigma2, ell)

end

function ls(y::Array{Float64, 2}, X::Array{Float64, 2};
            reml::Bool = false, loglik = true)

    # number of individuals
    n = size(y,1)
    # number of covariates
    p = size(X,2)

    b = X\y # uses QR decomposition
    yhat = X*b
    rss0 = sum((y-yhat).^2)
    
    if( reml )
        sigma2 = rss0/(n-p)
    else
        sigma2 = rss0/n
    end

    if(loglik) 
        if ( reml )
            logdetSigma = (n-p)*log(sigma2)
        else
            logdetSigma = n*log(sigma2)
        end
        
        ell = -0.5 * ( logdetSigma + rss0/sigma2 )
    else
        ell = missing
    end

    return LS_estimates(b, sigma2, ell)

end


"""
rss: residual sum of squares

y = outcome, matrix
X = predictors, matrix

Calculates the residual sum of squares using a Cholesky or
QRdecomposition.  The outcome matrix can be multivariate in which case
the function returns the residual sum of squares of each column. The
return values is a (row) vector of length equal to the number of columns of y.

"""
function rss(y::Array{Float64, 2}, X::Array{Float64, 2}; method = "cholesky")

    r = resid(y, X; method)
    rss = reduce(+, r.^2, dims = 1)

    return rss

end

"""
resid: calculate residuals

y = outcome, matrix
X = predictors, matrix

Calculates the residual sum of squares using a QR decomposition.  The
outcome matrix can be multivariate in which case the function returns
the residual matrix of the same size as the outcome matrix.

"""
function resid(y::Array{Float64, 2}, X::Array{Float64, 2}; method = "cholesky")

    # least squares solution
    # faster but numerically less stable
    if(method=="cholesky")
        b = (X'X)\(X'y)
    end

    # slower but numerically more stable
    if(method=="qr")
    fct = qr(X)
    b = fct\y
    end

    # estimate yy and calculate rss
    yhat = X*b
    resid = y-yhat

    return resid

end
