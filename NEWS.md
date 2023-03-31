Version 0.0.2 (March 09, 2023)

- Renamed the scan functions: now use `scan()` for performing single-trait scans, and use `bulkscan_...()` for performing multiple-trait scans.

- Permutation testing for single-trait scans are now through the same interface `scan(...)` by specifying the optional argument `permutation_testing = true` and also specifying how the permutation testing should be done.

    For example, 
    ```julia
    scan(y, G, K; permutation_testing = true, nperms = 1000, rndseed = 0, original = true)
    ```

    will return the matrix of LOD scores done on permutated copies of the original trait including itself, where each column is a vector of length $p-$the number of tested markers, of the LOD scores after performing LMM scans on each permutated copy.

- `bulkscan_...()`: there are three choices of algorithms of doing multiple-trait scans (using multi-threading). 

    - `bulkscan_null()` will estimate the heritability parameter independently for each trait supplied under the null model. Based on the estimated heritability for each trait, the scan (over all markers) will be done independently for each trait, using multi-threaded loops.

    Example:
    ```julia
    results = bulkscan_null(Y, G, K) # results is a Julia object

    ## L: the array of LOD scores for all traits in Y and all markers in G
    results.L
    ``` 

    - `bulkscan_null_grid()` will estimate the heritability parameter under the null by approximating the possible values on a discrete grid for all traits, and then group the traits with the "same" estimates together for downstream scans to get LOD scores. This method can be seen as an approximation of the `bulkscan_null()` method. (This is the fastest algorithm and empirically gives much faster performance than the other two multi-threaded methods for multiple-trait scans).

    Example: (need to supply a grid of candidated heritabilities as an array)
    ```julia
    grid = collect(0.0:0.05:0.95) # using a grid from 0.0 to 0.95 with step-size of 0.05
    results = bulkscan_null_grid(Y, G, K, grid) # results is a Julia object

    ## L: the array of LOD scores for all traits in Y and all markers in G
    results.L
    ``` 

    - `bulkscan_alt_grid()` will estimate the heritability parameter independently for every combination of trait and marker but on a discrete grid. This method can be seen as an appriximation of performing `scan(y, G, K; assumption = alt)` iteratively for every trait $y$.

    Example: (need to supply a grid of candidated heritabilities as an array)
    ```julia
    grid = collect(0.0:0.05:0.95) # using a grid from 0.0 to 0.95 with step-size of 0.05
    results = bulkscan_alt_grid(Y, G, K, grid)

    ## the output `results` is the array of LOD scores for all traits in Y and all markers in G
    results
    ``` 

- All scan functions now can model genome-wide associations after adjusting for additional covariates that are independent of the tested marker. 
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


Version 0.0.1
