# BulkLMM.jl

Welcome to the documentation of BulkLMM.jl! 

## What is BulkLMM.jl?
`BulkLMM.jl` is a software toolkit that performs fast genome scans even on large-scale datasets, employing a Linear Mixed Modeling (LMM) approach for the correction of counfounding by genetic relatedness.
`BulkLMM.jl` is implemented as a package in **Julia**, a high-level, high-performance programming language, in favor of both the computational speed and the ease of application for downstream analysis.

## Installation:

The package `BulkLMM.jl` can be installed by running

```julia
using Pkg
Pkg.add("BulkLMM")
```

Or, to install from the **Julia** REPL, first press `]` to enter the Pkg mode and then use:

```julia
add BulkLMM
```

The most recent release of the package can be obtained by running 

```julia
using Pkg
Pkg.add(url = "https://github.com/senresearch/BulkLMM.jl", rev="main")
```

## Linear Mixed Model (LMM)

`BulkLMM.jl` provides a fast application of the standard univariate linear mixed model on GWAS, which is formulated as the following:

For a single trait $y$ of $N$ observations, representation of the effect of a particular marker on trait $y$ can be modeled as such

$$y = \beta_0 + G_{j}\beta_{j}+\sum_{k \neq j}G_{k}\beta{k} + \epsilon = X\beta + g + \epsilon$$

and 
$$y|X, K \sim Norm(X\beta, V) \text{  ,  } V = \sigma^2_g K+\sigma^2_e I$$

where $X$ is the design matrix for fixed effects including the baseline mean $\beta_0$ and each testing SNP $G_j$ (for generalization, $X$ can include covariates other than SNPs), and $g$ is the random component consisting of non-testing SNPs $G_{k \in \{1, 2, ..., p\} \backslash j}$. 

The covariance matrix $V$ is a mixture  of the kinship matrix $K$ reflecting the genetic similarity between the measured $N$ subjects and the environmental noise; $\sigma^2_g$ and $\sigma^2_e$ denote the marginal genetic variance and environmental variance, respectively.

For scanning multiple traits, `BulkLMM.jl` simply perform calculation of a LMM independently for each trait, resorting to strategies of vectorized operations and multi-threaded processes to greatly accelerate the work.

For more information about our methods, check out the [documentation of the package](link not created yet) in our next release in the near future!

## Example: application on BXD mouse strains data

We demonstrate basic usage of `BulkLMM.jl` through an example applying the package on the BXD mouse strains data.

First, after successfully installed the package, load it to the current *Julia* session by

```julia
# using Pkg
# Pkg.add("BulkLMM")

using BulkLMM

```

The BXD data are accessible through our published [github repo](https://github.com/senresearch/BulkLMM.jl) of the `BulkLMM.jl` package as .csv files under the `data/bxdData` directory. 

The raw BXD traits `BXDtraits_with_missing.csv`contains missing values. After removing the missings, load the BXD traits data

```julia
pheno_file = "data/bxdData/BXDtraits.csv"
pheno = readBXDpheno(pheno_file);
```

Required data format for traits should be .csv or .txt files with values separated by `','`, with each column being the observations of $n$ BXD strains on a particular trait and each row being the observations on all $m$ traits of a particular mouse strain. 

Also load the BXD genotypes data. The raw BXD genotypes file `BXDgeno_prob.csv` contains even columns that each contains the complement genotype probabilities of the column immediately preceded (odd columns). Calling the function `readBXDgeno` will read the BXD genotype file excluding the even columns.

```julia
geno_file = "../data/bxdData/BXDgeno_prob.csv"
geno = readBXDgeno(geno_file);
```

Required data format for genotypes should be .csv or .txt files with values separated by `','`, with each column being the observations of genotype probabilities of $n$ BXD strains on a particular marker place and each row being the observations on all $p$ marker places of a particular mouse strain.

For the BXD data, 


```julia
size(pheno) # (number of strains, number of traits)
```




    (79, 35556)




```julia
size(geno) # (number of strains, number of markers)
```




    (79, 7321)



Compute the kinship matrix $K$ from the genotype probabilities using the function `calcKinship` 

```julia
kinship = calcKinship(geno); # calculate K
```

### Single trait scanning:

For example, to conduct genome-wide association mappings on the 1112-th trait, ran the function `scan()` with inputs of the trait (as a 2D-array of one column), geno matrix, and the kinship matrix.


```julia
traitID = 1112;
pheno_y = reshape(pheno[:, traitID], :, 1);
```


```julia
@time single_results = scan(pheno_y, geno, kinship);
```

      0.059480 seconds (80.86 k allocations: 47.266 MiB)


The output structure `single_results` stores the model estimates about the variance components (VC, environmental variance, heritability estimated under the null intercept model) and the lod scores. They are obtainable by


```julia
# VCs: environmental variance, heritability, genetic_variance/total_variance
(single_results.sigma2_e, single_results.h2_null)
```




    (0.0942525841453798, 0.850587848871709)




```julia
# LOD scores calculated for a single trait under VCs estimated under the null (intercept model)
single_results.lod; 
```

`BulkLMM.jl` supports permutation testing for a single trait GWAS. Simply run the function `scan_perms_lite()` with the number of permutations required as the input `nperms`: 


```julia
@time single_results_perms = scan_perms_lite(pheno_y, geno, kinship; nperms = 1000, original = false);
```

      0.079464 seconds (94.02 k allocations: 207.022 MiB)


(use the input `original = false` to suppress the default of computations of LOD scores on the original trait)

The output `single_results_perms` is a matrix of LOD scores of dimension `p * nperms`, with each column being the LOD scores of the $p$ markers on a permuted copy and each row being the marker-specific LOD scores on all permuted copies.


```julia
size(single_results_perms)
```




    (7321, 1000)




```julia
max_lods = vec(mapslices(x -> maximum(x), single_results_perms; dims = 1));
```


```julia
thrs = map(x -> quantile(max_lods, x), [0.05, 0.95]);
```

Plot the LOD scores in comparison with [GEMMA](https://github.com/genetics-statistics/GEMMA) (needs to run GEMMA to generate outputs elsewhere), as well as the LOD rejection thresholds from permutation testing:
    
![svg](img/output_48_0.svg)


### Multiple traits scanning:

To get LODs for multiple traits, for better runtime performance, first start *julia* with multiple threads following [Instructions for starting Julia REPL with multi-threads](https://docs.julialang.org/en/v1/manual/multi-threading/) or switch to a multi-threaded *julia* kernel if using Jupyter notebooks. 

Then, run the function `scan_lite_multivar()` with the matrices of traits, genome markers, kinship. The fourth required input is the number of parallelized tasks and we recommend it to be the number of *julia* threads. 

Here, we started a 16-threaded *julia* and run the following on a Linux server with the Intel(R) Xeon(R) Silver 4214 CPU @ 2.20GHz to get the LOD scores for all **~35k** BXD traits:


```julia
@time multiple_results_allTraits = scan_lite_multivar(pheno, geno, kinship, Threads.nthreads());
```

     82.421037 seconds (2.86 G allocations: 710.821 GiB, 41.76% gc time)


The output `multiple_results_allTraits` is a matrix of LOD scores of dimension $p \times n$, with each column being the LOD scores from performing GWAS on each given trait.


```julia
size(multiple_results_allTraits)
```




    (7321, 35556)

## For Questions:
Feel free to let us know any suspected bugs in the current release by posting them to GitHub Issues or contacting authors directly. We also appreciate contributions from users including improving performance and adding new features.

Check out RELEASE-NOTES.md to see what's new in each `BulkLMM.jl` release.
