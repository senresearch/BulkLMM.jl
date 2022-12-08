# run-lmmlite.r: this is an R script generating data outputs used for testing. 
# It runs the R package lmmlite on BXD data to perform single trait genome scans on the 7919-th BXD trait.
# To be able to execute the script, one will need to set the working directory as 
# setwd("../BulkLMM.jl/test/run-lmmlite_R")

rm(list = ls())
# install.packages("devtools")
library(devtools)

# install.packages("remotes")
# library(remotes)
# install_github("kbroman/lmmlite")

library(lmmlite)

library(knitr)


## Check working directory
getwd()



pheno = read.csv("processed_bxdData/BXDpheno.csv")
geno = read.csv("processed_bxdData/BXDgeno.csv")
K = read.csv("processed_bxdData/BXDkinship.csv")



pheno_y = pheno[, 7919]
K_mat = data.matrix(K)
geno_mat = data.matrix(geno)



e_null = eigen_rotation(K_mat, pheno_y, NULL)



n = nrow(geno_mat)
p = ncol(geno_mat)


## params_null = fitLMM(e_null$Kva, e_null$y, e_null$X, reml = T)
params_null = fitLMM(e_null$Kva, e_null$y, e_null$X, reml = F)
est_hsq = params_null$hsq
est_sigmasq = params_null$sigmasq



## Helper functions:


## Function to construct the design matrix for each marker G_j, j = 1,..., p.
construct_Gj = function(geno, intercept = TRUE){
  
  list_G = list()
  n = nrow(geno) ## number of individuals
  p = ncol(geno) ## number of markers
  
  intercept = rep(1, n)
  
  for(j in 1:p){
    Gj = cbind(intercept, geno[, j])
    list_G[[j]] = Gj
  }
  
  return(list_G)
  
}

K_eVects = e_null$Kve_t # left eigen-vectors of kinship matrix
K_eVals = e_null$Kva # eigen-values of kinship matrix 

run_model = function(K_eVals, K_eVects, Gj, y, hsq){
  
  # Rotate data:
  y_star1 = K_eVects %*% y # may not need to do everytime; needs refinement
  Gj_star1 = K_eVects %*% Gj
  
  # Get RSS:
  # ml_soln = getMLsoln(hsq, K_eVals, y_star1, Gj_star1, reml = T)
  ml_soln = getMLsoln(hsq, K_eVals, y_star1, Gj_star1, reml = F)
  
  
  
  return(ml_soln)
  
}

rss2Lod = function(rss_null, rss_mod, n){
  
  lod = (n/2)*(log10(rss_null) - log10(rss_mod))
  
  return(lod)
}


# Run model for each marker:

list_Gj = construct_Gj(geno)


### You can manually give the hsq estimated from BulkLMM flmm to getMLsoln
## est_hsq = 
###

# Compute the residual sum of squares for the null model (including just the intercept)
# results_null = getMLsoln(est_hsq, e_null$Kva, e_null$y, e_null$X, reml = T) # to evaluate the null model results using REML
results_null = getMLsoln(est_hsq, e_null$Kva, e_null$y, e_null$X, reml = F)


list_RSS = rep(NA, p+1)
rss_null = attributes(results_null)$rss
list_RSS[1] = rss_null

list_ml_solns = list(results_null)

# Extract the residual sum of squares for each model including an intercept and one of the p markers
for(j in 1:p){
  
  ml_soln = run_model(K_eVals, K_eVects, list_Gj[[j]], pheno_y, est_hsq)
  rss = attributes(ml_soln)$rss
  
  list_ml_solns[[j+1]] = ml_soln
  list_RSS[j+1] = rss

}


# Compute the LOD scores using residual sum of squares of the models
list_LOD = sapply(list_RSS[2:length(list_RSS)], 
       function(x) rss2Lod(list_RSS[1], x, n))
list_LOD = c(NA, list_LOD)

list_Sigma_e = rep(NA, length(list_ml_solns))
list_Beta_0 = rep(NA, length(list_ml_solns))
list_Beta_1 = rep(NA, length(list_ml_solns))

for(j in 1:length(list_ml_solns)){
  
  list_Beta_0[j] = list_ml_solns[[j]][[1]][1]
  list_Beta_1[j] = list_ml_solns[[j]][[1]][2] 
  list_Sigma_e[j] = list_ml_solns[[j]][2]
  
}

list_Beta_0 = unlist(list_Beta_0)
list_Beta_1 = unlist(list_Beta_1)
list_Sigma_e = unlist(list_Sigma_e)

head(list_LOD)
tail(list_LOD)

head(list_Sigma_e)
tail(list_Sigma_e)



# Generate the output files. 
# They will include the information about the estimated fixed effects coefficients, 
# the estimated environmental variance, and the LOD scores.
results_lmmlite = data.frame(cbind(list_Beta_0, list_Beta_1, 
                                   list_Sigma_e, list_LOD))
colnames(results_lmmlite) = c("Est_Beta_0", "Est_Beta_1", 
                              "Est_Sigma_e", "LOD")
rownames = c("Null")

for(j in 1:p){
  rownames = c(rownames, paste("G_", j))
}

rownames(results_lmmlite) = rownames
kable(head(results_lmmlite))

# write.csv(results_lmmlite, "output/result.lmmlite_REML.csv")
write.csv(results_lmmlite, "output/result.lmmlite_ML.csv")

