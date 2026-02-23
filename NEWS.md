## Version 1.2.2 (Feb 20, 2026)
### Extended features:
- Effect size reporting: 
    - Added estimates of effect sizes and the associated standard errors in the outputs.
    - The outputs of BulkLMM now contain the LOD scores, estimated effect sizes and standard errors, estimated variance components (in terms of heritabilities $h^2$), and p-values (optional if `output_pvals = true`).
    - Example: Given the `out` as the structure of BulkLMM executions, effect sizes reflecting strengths of associations between trait-marker pairs are available as `out.B`, and the standard errors are as `out.SE`.
    - By default, effect sizes and standard errors will always be returned.
- More reliable estimation of variance component $h^2$ under grid-search based methods:
    - For grid-search based BulkLMM methods(i.e., variance component estimation is discretized and done through grid-search for algorithmic efficiency), `bulkscan(..., method = "null-grid")` and `bulkscan(..., method = "alt-grid")`, instead of relying on a grid of arbitrary values or the user's choice, the package now supports the functionality to propose a grid based on exact values estimated from a smaller subset of traits, by specifying the proportion of the subset size in `subset_size_for_h2_scale`.
    - This will add slight computational overhead but with more reliable estimates.
    - By default, `subset_size_for_h2_scale = 0.0`—no dataset-specific grid will be proposed, but package estimates using a default grid of (0.0, 0.1, 0.2, ..., 0.9) or a grid defined by the user.


### Bugs fixed:
- There was a mistake in the formula for posterior calculation when using the single-trait scan function (`scan`). In this release that has been fixed and matches the correct version in multiple-trait scan function `bulkscan()`.

## Version 1.2.0 (Aug 17, 2023)
- Help documentation added: to view the help documentation for functions `scan()`, and `bulkscan()`, type `?` in front of the functions.
- New features:
    - Added the wrapper function `bulkscan()` for the three algorithms of multiple-trait scans previously named as `bulkscan_null()`, `bulkscan_null_grid()`, `bulkscan_alt_grid()`. Now, the user can simply call the common interface `bulkscan(...; method = )` by supplying with the method by the user's specific favor of computational speed or precision. Allowable inputs are string types, named as "null-exact", "null-grid", "alt-grid". The default option is "null-grid" with a loose grid of h2-step of size 0.1 (a grid of 10 values from 0.0, 0.10, ..., 0.90).
    - Added the option for SVD decomposition of the kinship matrix. To use this feature, supply the option `decomp_scheme = svd`.
    - Added the option in both `scan()` and `bulkscan()` functions for returning the $-log_{10}(p)$ result, where $p$ is the likelihood ratio test p-value: To use this feature, supply the option `output_pvals = true`. For more details, check the help instruction by `?scan()`, `?bulkscan()`.
- Fixed bugs: 
    - Fixed a bug causing compilation error in `bulkscan()` "null-grid" algorithm with REML due to a typo.
    - Fixed a bug causing output dimension mismatch when using the function `scan()` for permutation testing with adjusted covariates.

## Version 1.1.1 (July 11, 2023)
- Fixed bugs:  
    - REML option in multiple trait scan functions ("bulkscan"'s) used to lead to compilation errors.
    - `checkZeros()` before will not consider and detect small floats that are significantly close to zero as "zeros".
- Updated the interface for the function to obtain permutation testing thresholds `get_thresholds()`: now the user will specify the significance levels to obtain the corresponding thresholds. For example, user input of significance level of 0.05 will give the 95th quantile of LOD scores from permutations.

## Version 1.1.0 (June 13, 2023)
- Added the feature to output the comprehensive results for `bulkscan_alt_grid()`, which does multiple-trait scans with estimation of h2 under each alternative model using grid-search approximation. It now returns the heritability estimated for each trait and each genomic marker, stored in a matrix of size $p \times m$, where $p$ is the number of tested markers and $m$ is the number of tested traits. The user is able to access this information by getting the field `h2_panel` from the output object.

- Revised the output structure for the single-trait scan function `scan(...; permutation_testing = true, ...)` when permutation testing is requested, to preserve the consistency of results for `scan()` function in general. Just as when permutation testing is not requested, calling `scan(...; permutation_testing = true, ...)` now reports the variance components (residual variance and heritability estimated from the null), the LOD scores for the original trait, and with the option to do permutation testing on, it additionally reports the raw output of LOD scores for each permuted copies of the original, as a matrix of size $p \times nperms$, where $nperms$ is the number of permutations requested. The user is able to access this information by getting the field `L_perms` from the output object.

- Removed dependency to the package `LoopVectorization.jl`.

- Modified `bulkscan_null_grid()` (scan function using grid-search algorithm) which previously did an implicit standardization to input matrices, which could cause accuracy issues when using weighted variances feature. 

- Added new tests for the scanning functions using the `weights` keyword argument.


## Version 1.0.1 (April 19, 2023)
- Enabled modeling for heteroskedestic sources of error variance:

    By our assumption of the linear mixed models, 
        $$y_i = X_0 \beta_0 + g_j \beta_{ij}+\epsilon_{ij}$$
    and
        $$\epsilon_{ij} \sim N(0, \sigma^2_g K + \sigma^2_e V)$$

    where in the usual case $W = I$. 

    However, there are cases where this may not be true. For example, when we take $y$ as the strain means by taking averages of the expression trait measurements of individual samples by strain for each strain, where the number of samples may not be all equal for all the strains, we may want $V$ to be the diagonal matrix with the diagonal inversely-proportional to the sample size of each strain.

    The new version of BulkLMM enables this feature, in all of our scan functions. To use, simply call the desired function with an additional input `weights` as a vector of the diagonal elements of the matrix $V^{-1/2}$, such as:

    Suppose the first strain has four samples, the second strain has two samples, and the third strain has only one sample. Then, one can call the function on the strain means with additional input of `weights` $= [\sqrt{4}, \sqrt{2}, \sqrt{1}]$:

    ```julia
    lod = scan(y, G, K; weights = weights, ...).lod

    L = bulkscan_null(Y, G, K; weights = weights, ...).L # LOD scores for every input trait

    L = bulkscan_null_grid(Y, G, K, grid; weights = weights, ...).L # LOD scores for every input trait
    
    L = bulkscan_alt_grid(Y, G, K, grid; weights = weights, ...).L # LOD scores for every input trait
    ```

    where $y$ contains the `number of strains` strain means of the expression trait of interest and $Y$ contains the strain means of multiple expression traits of interest.

## Version 1.0.0 (March 09, 2023)

- Renamed the scan functions: now use `scan()` for performing single-trait scans, and use `bulkscan_...()` for performing multiple-trait scans.

- Permutation testing for single-trait scans is now through the same interface `scan(...)` by specifying the optional argument `permutation_testing = true` and also specifying how the permutation testing should be done.

    For example, 
    ```julia
    scan(y, G, K; permutation_testing = true, nperms = 1000, rndseed = 0, original = true)
    ```

    will return the matrix of LOD scores done on permutated copies of the original trait, including itself, where each column is a vector of length $p-$the number of tested markers of the LOD scores after performing LMM scans on each permutated copy.

- `bulkscan_...()`: there are three choices of algorithms for doing multiple-trait scans (using multi-threading). 

    - `bulkscan_null()` will estimate the heritability parameter independently for each trait supplied under the null model. Based on the estimated heritability for each trait, the scan (overall markers) will be done independently for each trait, using multi-threaded loops.

    Example:
    ```julia
    results = bulkscan_null(Y, G, K) # results is a Julia object

    ## L: the array of LOD scores for all traits in Y and all markers in G
    results.L
    ``` 

    - `bulkscan_null_grid()` will estimate the heritability parameter under the null by approximating the possible values on a discrete grid for all traits and then group the traits with the "same" estimates together for downstream scans to get LOD scores. This method can be seen as an approximation of the `bulkscan_null()` method. (This algorithm is the fastest and empirically performs much faster than the other two multi-threaded methods for multiple-trait scans).

    Example: (need to supply a grid of candidates heritabilities as an array)
    ```julia
    grid = collect(0.0:0.05:0.95) # using a grid from 0.0 to 0.95 with step size of 0.05
    results = bulkscan_null_grid(Y, G, K, grid) # results is a Julia object

    ## L: the array of LOD scores for all traits in Y and all markers in G
    results.L
    ``` 

    - `bulkscan_alt_grid()` will estimate the heritability parameter independently for every combination of trait and marker but on a discrete grid. This method can be seen as an approximation of performing `scan(y, G, K; assumption = alt)` iteratively for every trait $y$.

    Example: (need to supply a grid of candidates heritabilities as an array)
    ```julia
    grid = collect(0.0:0.05:0.95) # using a grid from 0.0 to 0.95 with step size of 0.05
    results = bulkscan_alt_grid(Y, G, K, grid)

    ## the output `results` is the array of LOD scores for all traits in Y and all markers in G
    results
    ``` 

- All scan functions can now model genome-wide associations after adjusting for additional covariates that are independent of the tested marker. 
    - For example, to include the matrix $Z$ of additional covariates in a single-trait scan on the trait $y$, use

    ```julia
    scan(y, G, Z, K);
    ```

    - To use any of the `bulkscan_...()` functions for scanning multiple traits and to include the covariates $Z$, use

    ```julia

    ## bulkscan_null:
    bulkscan_null(y, G, Z, K);

    ## bulkscan_null_grid:
    grid = collect(0.0:0.05:0.95);
    bulkscan_null_grid(y, G, Z, K, grid);

    ## bulkscan_alt_grid:
    bulkscan_alt_grid(y, G, Z, K, grid);
    ```

## Version 0.1.0 - Initial Release
