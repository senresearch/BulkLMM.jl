## Version 0.2.1 (April 19, 2023)
- Enabled modeling for heteroskedestic sources of error variance:

    By our assumption of the linear mixed models, 
        $$y_i = X_0 \beta_0 + g_j \beta_{ij}+\epsilon_{ij}$$
    and
        $$\epsilon_{ij} \sim N(0, \sigma^2_g K + \sigma^2_e V)$$

    where in the usual case $W = I$. 

    However, there are cases where this may not be true. For example, when we take $y$ as the strain means by taking averages of the expression trait measurements of individual samples by strain for each strain, where the number of samples may not be all equal for all the strains, we may want $V$ to be the diagonal matrix with the diagonal inversely-proportional to the sample size of each strain.

    The new version of BulkLMM enables this feature, in all of our scan functions. To use, simply call the desired function with an additional input `weights` as a vector of the diagonal elements of the matrix $V^{-1/2}$, such as:

    Suppose the first strain has four samples, the second strain has two samples, and the third strain has only one sample. Then, one can call the function on the strain means with additional input of `weights` $= [\sqrt{4}, \sqrt{2}, \sqrt{1}]$:

    `lod = scan(y, G, K; weights = weights, ...).lod`

    `L = bulkscan_null(Y, G, K; weights = weights, ...).L` # LOD scores for every input trait

    `L = bulkscan_null_grid(Y, G, K, grid; weights = weights, ...).L` # LOD scores for every input trait

    where $y$ contains the `number of strains` strain means of the expression trait of interest and $Y$ contains the strain means of multiple expression traits of interest.

## Version 0.2.0 (March 09, 2023)

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
