##################################################################
# wls: weighted least squares
##################################################################

mutable struct LSEstimates
    b::Array{Float64, 2}
    sigma2::Float64
    ell::Float64
end

struct LSEstimates_multivar
    B::Array{Float64, 2}
    Sigma2::Array{Float64, 2}
    Ell::Array{Float64, 2}
end


"""
wls: Weighted least squares estimation

y = outcome, matrix
X = predictors, matrix
w = weights (positive, inversely proportional to variance), one-dim vector

"""
function wls(y::Array{Float64, 2}, X::Array{Float64, 2}, w::Array{Float64, 1}, prior::Array{Float64, 1};
             reml::Bool = false, loglik::Bool = true, method::String = "qr")

    (n, p) = size(X); # get number of observations and the number of markers from geno dimensions      

    n = size(y, 1); # get number of observations       

    # check if weights are positive
    if(any(w .<= .0))
        @warn "Some weights are not positive."
    end

    # square root of the weights
    sqrtw = sqrt.(w)

    # logdetXtX = logdet(X' * X); constant term that does not depend on the parameters (weights); is not needed

    # scale by weights
    yy = rowMultiply(y, sqrtw)
    XX = rowMultiply(X, sqrtw)

    # least squares solution
    # faster but numerically less stable
    if(method == "cholesky")
        fct = cholesky(XX'XX)
        coef = fct\(XX'yy)
        logdetXXtXX = logdet(fct)
    end

    # slower but numerically more stable
    if(method == "qr")
        fct = qr(XX)
        coef = fct\yy

        # logdetXXtXX = 2*logdet(fct.R) # need 2 for logdet(X'X)
        # logdetXXtXX = logdet(fct.R' * fct.R);
        logdetXXtXX = 2*logabsdet(fct.R)[1];
    end

    yyhat = XX*coef
    rss0 = norm(yy-yyhat)^2 #sum((yy-yyhat).^2)

    if prior[2] > 0.0
        prior_df = prior[2]+2;
    else
        prior_df = prior[2];
    end

    if(reml)
        sigma2_e = (rss0+prior[1]*prior[2])/((n-p)+prior_df)
    else
        sigma2_e = (rss0+prior[1]*prior[2])/(n+prior_df)
    end

    # see formulas (2) and (3) of Kang (2008)
    if(loglik)

        ll = -0.5 * ((n+prior_df)*log(sigma2_e) - sum(log,w) + (rss0+prior[1]*prior[2])/sigma2_e)
        
        if(reml)
            # ell = ell + 0.5 * (p*log(2pi*sigma2) + logdetXtX - logdetXXtXX) # full log-likelihood including the constant terms;
            ll = ll + 0.5 * (p*log(sigma2_e) - logdetXXtXX)
        end
        
    else
        ll = missing;
    end

    return LSEstimates(coef, sigma2_e, ll);

end

"""
wls_multivar: Multivariate weighted least-square estimation

"""
function wls_multivar(Y::Array{Float64, 2}, X::Array{Float64, 2}, w::Array{Float64, 1}, prior::Array{Float64, 1};
             reml::Bool = false, loglik::Bool = true, method::String = "qr")

    (n, p) = size(X); # get number of observations and the number of markers from geno dimensions      

    n = size(Y, 1); # get number of observations       

    # check if weights are positive
    if(any(w .<= .0))
        @warn "Some weights are not positive."
    end

    # square root of the weights
    sqrtw = sqrt.(w)

    # logdetXtX = logdet(X' * X); constant term that does not depend on the parameters (weights); is not needed

    # scale by weights
    YY = rowMultiply(Y, sqrtw)
    XX = rowMultiply(X, sqrtw)

    # least squares solution
    # faster but numerically less stable
    if(method == "cholesky")
        fct = cholesky(XX'XX)
        coef = fct\(XX'YY)
        logdetXXtXX = logdet(fct)
    end

    # slower but numerically more stable
    if(method == "qr")
        fct = qr(XX)
        coef = fct\YY # will be a matrix with each column be the fitted coefficients for one response variable

        # logdetXXtXX = 2*logdet(fct.R) # need 2 for logdet(X'X)
        # logdetXXtXX = logdet(fct.R' * fct.R);
        logdetXXtXX = 2*logabsdet(fct.R)[1];
    end

    YYhat = XX*coef
    # rss0 = mapslices(x -> sum(x.^2), YY-YYhat; dims = 1)
    rss0 = mapslices(x -> norm(x)^2, YY-YYhat; dims = 1)

    if prior[2] > 0.0
        prior_df = prior[2]+2;
    else
        prior_df = prior[2];
    end

    if(reml)
        sigma2_e = (rss0.+prior[1]*prior[2])./((n-p)+prior_df)
    else
        sigma2_e = (rss0.+prior[1]*prior[2])./(n+prior_df)
    end

    
    # see formulas (2) and (3) of Kang (2008)
    if(loglik)

        ll = -0.5 * ((n+prior_df)*log.(sigma2_e) .- sum(log,w) .+ (rss0.+prior[1]*prior[2])./sigma2_e)
        
        if(reml)
            # ll = ll + 0.5 * (p*log(2pi*sigma2) + logdetXtX - logdetXXtXX) # full log-likelihood including the constant terms;
            ll = ll .+ 0.5 * (p*log.(sigma2_e) .- logdetXXtXX)
        end
        
    else
        ll = missing;
    end
    

    return LSEstimates_multivar(coef, sigma2_e, ll);
    
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
function rss(y::Array{Float64, 2}, 
             X::Union{Array{Float64, 1}, Array{Float64, 2}}; 
             method = "qr")

    r = resid(y, X; method = method)
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
function resid(y::Array{Float64, 2}, 
               X::Union{Array{Float64, 1}, Array{Float64, 2}}; 
               method::String = "qr")

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