
## eQTL example: application on BXD spleen expression data

We demonstrate the usage of `BulkLMM.jl` for eQTLs through an example applying
the package on the BXD mouse strains data.

First, after successfully installing the package, load it to the
current *Julia* session by

```@example
using BulkLMM, BigRiverQTLPlots, Plots
using CSV, DelimitedFiles, DataFrames, Statistics
```

The BXD data are accessible through our published [github
repo](https://github.com/senresearch/BulkLMM.jl) of the `BulkLMM.jl`
package as .csv files under the `data/bxdData` directory.

The original data for BXD spleen traits `BXDtraits_with_missing.csv` contains missing
values. We saved the data after removing any missing values to the file named "spleen-pheno_nomissing.csv" under the same directory. 

```@example
bulklmmdir = dirname(pathof(BulkLMM));
pheno_file = joinpath(bulklmmdir,"..","data/bxdData/spleen-pheno-nomissing.csv");
pheno = readdlm(pheno_file, ',', header = false);
pheno_processed = pheno[2:end, 2:(end-1)].*1.0; # exclude the header, the first (transcript ID)and the last columns (sex)
```

Required data format for traits should be .csv or .txt files with
values separated by `','`, with each column being the observations of
$n$ BXD strains on a particular trait and each row being the
observations on all $m$ traits of a particular mouse strain.

Also load the BXD genotypes data. The raw BXD genotypes file
`BXDgeno_prob.csv` contains even columns that each contains the
complement genotype probabilities of the column immediately preceded
(odd columns). Calling the function `readBXDgeno` will read the BXD
genotype file excluding the even columns.

```@example
geno_file = joinpath(bulklmmdir,"..","data/bxdData/spleen-bxd-genoprob.csv");
geno = readdlm(geno_file, ',', header = false);
geno_processed = geno[2:end, 1:2:end] .* 1.0;
```

Compute the kinship matrix $K$ from the genotype probabilities using the function `calcKinship`. 

```@example
kinship = calcKinship(geno_processed); # calculate K
```
Also, read in the `gmap.csv` and the `phenocovar.csv` under `data/bxdData/` directory as

```@example
gmap_file = joinpath(bulklmmdir,"..","data/bxdData/gmap.csv");
gInfo = CSV.read(gmap_file, DataFrame);
phenocovar_file = joinpath(bulklmmdir,"..","data/bxdData/phenocovar.csv");
pInfo = CSV.read(phenocovar_file, DataFrame);
```
To get LODs for multiple traits, for better runtime performance, first
start *julia* with multiple threads following [Instructions for
starting Julia REPL with
multi-threads](https://docs.julialang.org/en/v1/manual/multi-threading/)
or switch to a multi-threaded *julia* kernel if using Jupyter
notebooks.

Then, run the function `bulkscan()` with the matrices of the
traits of interest, genome markers, and the kinship. Type `?bulkscan()` for more 
detailed description of the function.

```@example
@time multiple_results_allTraits = bulkscan(pheno_processed, geno_processed, kinship);
```

      0.888991 seconds (2.13 k allocations: 5.098 GiB, 10.25% gc time)

Please Note: the default method and modeling options for `bulkscan()` takes an approximated approach for the best runtime performance. The user may choose to use other methods and options provided for more precision but longer runtime, following the detailed instructions in `?bulkscan()`.

The output `multiple_results_allTraits` is an object containing our model results:
- the matrix of LOD scores $L_{p \times m}$, where $p$ is the number of markers and $m$ is number of traits; each column corresponds to the LOD scores resulting from performing GWAS on each given trait.
- variance components (heritability) results will be returned in various formats depending on the specific method and other options by the user. For more details, enter `?bulkscan()`.


```@example
size(multiple_results_allTraits.L)
```

    (7321, 35554)

To visualize the multiple-trait scan results, we can use the plotting function `plot_eQTL` from `BigRiverQTLPlots.jl` to generate the eQTL plot. In the following example, we only plot the LOD scores that are above 5.0 by calling the function and specifying in the optional argument `threshold = 5.0`:

```@example
plot_eQTL(multiple_results_allTraits.L, pInfo, gInfo; threshold = 5.0)
```