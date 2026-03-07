### QTL Example: application on BXD spleen expression data

We demonstrate basic usage of `BulkLMM.jl` for QTLs through an example applying
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

### Single trait scanning:

For example, to conduct genome-wide associations mapping on the
1112-th trait, we can run the function `scan()` with inputs of the trait (as
a 2D-array of one column), geno matrix, and the kinship matrix. Type `?scan()` for more 
detailed description of the function.

```@example
traitID = 1112;
pheno_y = reshape(pheno_processed[:, traitID], :, 1);
```


```@example
@time single_results = scan(pheno_y, geno_processed, kinship);
```

      0.111507 seconds (183.73 k allocations: 48.849 MiB, 88.97% gc time)

The output `single_results` is an object containing model results about the variance components (residual variance and the heritability parameter) estimated under the null baseline model, and the lod scores, as the fields named respectively as "sigma2_e", "h2_null", and "lod". By default, variance components are estimated from maximum-likelihood (ML). The user may choose REML for estimating by specifying in the input "reml = true".

```@example
# VCs: residual variance, heritability which is the proportion of genetic variance to total variance
(single_results.sigma2_e, single_results.h2_null)
```

    (0.09448827756304541, 0.850090732186436)


```@example
# LOD scores calculated for a single trait under VCs estimated under the null (intercept model)
single_results.lod; 
```

`BulkLMM.jl` supports permutation testing for a single trait GWAS. Simply run the function `scan()` and set the optional keyword argument `permutation_test = true` with the required number of permutations as `nperms = # of permutations`. For example, to ask the package to do a permutation testing of 1000 permutations, do 

```@example
@time single_results_perms = scan(pheno_y, geno_processed, kinship; permutation_test = true, nperms = 1000);
```

      0.115869 seconds (2.85 k allocations: 144.585 MiB, 74.08% gc time)

Similar to the results of the single-trait scan with no permutation, `single_results_perms` contains the fields `sigma2_e`, `h2_null`, and `lod` for the original trait. Additionally, we report the results of permutation tests as the raw LOD scores computed for each permuted copies, which are stored in a matrix named as `L_perms` of dimension $p \times n_perms$, where each column contains the LOD scores corresponding to $p$ markers on one permuted copy, and each row are the LOD scores for a particular marker fitted on all 1000 permuted copies.


```@example
size(single_results_perms.L_perms)
```

    (7321, 1000)

Based on the results of the permutation test, we can use the function `get_thresholds()` to obtain the LOD thresholds according to the quantile probabilities, based on the significance levels requested. 

For example, if we would like to see the significant LOD scores with significance levels of 0.10 and 0.05, we can run the function `get_thresholds()` and give raw results of LOD scores from permutation testing and the desired significance (0.10, 0.05). The user can ask for results of as many significance levels as they want. In this case, the function reports the 90th and the 95th quantiles among LOD scores testing all 1000 permuted copies.

```@example
lod_thresholds = get_thresholds(single_results_perms.L_perms, [0.10, 0.05]);
```


    (probs = [0.9, 0.95], thrs = [3.4497165009563524, 3.798623630569368])

Finally, let's plot the BulkLMM LOD scores of the 1112-th trait using the QTL plotting function from the package  [`BigRiverQTLPlots.jl`](https://github.com/senresearch/BigRiverQTLPlots.jl):

```@example
plot_QTL(single_results_perms, gInfo, significance = [0.10, 0.05])
```