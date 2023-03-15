#=
`visuals_utils.jl` contains functions to generate eQTL plots.
- Functions for eQTL plots of multiple-trait scan results (beta version)
=#


############## 
# eQTL PLOT  #
##############
"""
**pseudotick** -*Function*.

    pseudotick(mytick::Vector{Float64}) => Vector{Float64}

Returns coordinates of the new ticks. 

"""
function pseudotick(tick)
    minortick = zeros(size(tick,1))
    for i in 1:size(tick,1)
        if i == 1
            minortick[i] = tick[i]/2
        else
            minortick[i] = tick[i-1] + (tick[i] - tick[i-1])/2
        end 
    end   
    return minortick
end


"""
    Recipe for eQTL plots.
"""
mutable struct EQTLPlot{AbstractType}
    args::Any                                      
end

eQTLplot(args...; kw...) = RecipesBase.plot(EQTLPlot{typeof(args[1])}(args); kw...)

@recipe function f(h::EQTLPlot) 
    # check types of the input arguments
    if length(h.args) != 5 || !(typeof(h.args[1]) <: AbstractVector) ||
        !(typeof(h.args[2]) <: AbstractVector) || !(typeof(h.args[3]) <: AbstractVector) ||
        !(typeof(h.args[4]) <: AbstractVector) || !(typeof(h.args[5]) <: AbstractVector) 
        error("eQTL Plots should be given three vectors.  Got: $(typeof(h.args))")
    end
    # Note: if confidence or interval not symmetric, then ε should be a vector of tuple.
    # collect(zip(vCI1, vCI2));
    
    
    x, y, lod, steps, chr_names = h.args
    
    # set a default value for an attribute with `-->`
    xlabel --> "eQTL Position (Chromosome)"
    ylabel --> "Transcript Position (Chromosome)"
    
    marker --> 6
    markerstrokewidth --> 0.3
    
    bottom_margin --> 0mm
    right_margin --> 0mm
    
    guidefontsize --> 15
    fontfamily --> "Helvetica"
    
    size --> (650, 550)
        
    # set up the subplots
    legend := false
    link := :both
    # framestyle := [:none :axes :none]
    # yaxis := false 
    xlims := (0, steps[end])
    ylims := (0, steps[end])
    grid := false
    

    tickfontsize := 8
    tick_direction := :out

    xticks := (pseudotick(steps), chr_names) 
    yticks := (pseudotick(steps), chr_names)
    
    
    # vertical lines
    @series begin
        seriestype := :vline
        linecolor := :lightgrey
        primary := false
        # alpha := 0.5
        steps
        
    end
    
    # horizontal lines
    @series begin
        seriestype := :hline
        linecolor := :lightgrey
        primary := false
        # alpha := 0.5
        steps
    end
    

    # main confidence plot
    @series begin
        seriestype := :scatter
        marker_z := lod
        framestyle := :box
        linecolor := :black#nothing
        # get the seriescolor passed by the user
        color -->  cgrad(:blues)
        cbar --> true

        x, y
    end  
end


"""

sortnatural(x::Vector{String}) => Vector(::String)

Natural sort a string vector accounting for numeric sorting.

## Example
```julia
julia> myvec = ["12", "2", "1", "Y", "10", "X"];
julia> sort(myvec)
6-element Vector{String}:
 "1"
 "10"
 "12"
 "2"
 "X"
 "Y"
 julia> sortnatural(myvec)
 6-element Vector{String}:
  "1"
  "2"
  "10"
  "12"
  "X"
  "Y"
```
"""
function sortnatural(x::Vector{String})
    f = text -> all(isnumeric, text) ? Char(parse(Int, text)) : text
    sorter = key -> join(f(m.match) for m in eachmatch(r"[0-9]+|[^0-9]+", key))
    sort(x, by=sorter)
end


"""

function get_eQTL_accMb(mlodmax::Matrix{Float64}, dfpInfo::DataFrame, dfgInfo::DataFrame; 
    chrColname::String = "Chr", mbColname::String = "Mb", thrs::Float64 = 5.0)

Natural sort a string vector accounting for numeric sorting.

## Arguments
- `mlodmax` is the matrix containing the maximum value of LOD score of each phenotype and its corresponding index
- `dfpInfo` is a dataframe containing the phenotype informnation such as probeset, chromosomes names and Mb distance
- `dfgInfo` is a dataframe containing the genotype informnation such as locus, cM distance, chromosomes names and Mb distance  
- `chrColname` column name containing the chromosomes information, default is `"Chr"`, in the dataframes. 
- `mbColname` column name containing the Mb distance information, default is `"Mb"`, in the dataframes.
- `thrs` is the LOD threshold value, default is `5.0``.



"""
function get_eQTL_accMb(mlodmax::Matrix{Float64}, dfpInfo::DataFrame, dfgInfo::DataFrame; 
                        chrColname::String = "Chr", mbColname::String = "Mb", thr::Float64 = 5.0)
    # match chromosomes in pheno dataframe according to chromosomes list in geno dataframe
    dfpInfo_filtered = copy(dfpInfo) #match_chrs_pheno_to_geno(dfpInfo, dfgInfo)

    # prepare filtered pheno dataframe to compute accumulated mb distance for plotting
    rename!(dfpInfo_filtered, Dict(Symbol(mbColname) => :phenocovar_mb));
    rename!(dfpInfo_filtered, Dict(Symbol(chrColname) => :phenocovar_chr)); 
    dfpInfo_filtered[:, :acc_phenocovar_mb] .= dfpInfo_filtered.phenocovar_mb;

    # get a copy of genotype info dataframe
    gmap = copy(dfgInfo) 
    gmap.acc_mb .= gmap[!, mbColname]
    rename!(gmap, Dict(Symbol(mbColname) => :geno_mb));
    rename!(gmap, Dict(Symbol(chrColname) => :geno_chr)); 
    
    # get unique chromosomes names from genotype info dataframe
    vChrNames = unique(gmap.geno_chr);
    vChrNames = sortnatural(string.(vChrNames))

    # initiate steps vector
    steps = Array{Float64}(undef, length(vChrNames))

    # if more than one chromosome
    if length(vChrNames) > 1 
        # compute accumulated mb distance
        for i in 1:length(vChrNames)-1 
            
            # get temp matrix based on a chromosome name
            phenotemp = filter(:phenocovar_chr => x-> x == vChrNames[i], dfpInfo_filtered)
            genotemp = filter(:geno_chr => x-> x == vChrNames[i], gmap)

            # get maximum distance for this chromosome
            steps[i] = max(
                        maximum(genotemp.acc_mb),
                        maximum(phenotemp.acc_phenocovar_mb) 
                    )

            # calculate the accumulated distance for the next chromosome
            nextchr_phenotemp = view(dfpInfo_filtered, dfpInfo_filtered.phenocovar_chr .== vChrNames[i+1], :)
            nextchr_genotemp = view(gmap, gmap.geno_chr .== vChrNames[i+1], :) 

            nextchr_phenotemp.acc_phenocovar_mb .= nextchr_phenotemp.phenocovar_mb .+ steps[i]
            nextchr_genotemp.acc_mb .= nextchr_genotemp.geno_mb .+ steps[i]
        end
    end

    steps[end] = max(
                  maximum(dfpInfo_filtered.acc_phenocovar_mb), 
                  maximum(gmap.acc_mb)
                )

    # get the corresponding index of the maximum values of LOD score of each phenotype
    idxlodmax = trunc.(Int, mlodmax[:,1])
    # concatenate results and gmap info
    pheno_gmap_lod = hcat(dfpInfo_filtered, gmap[idxlodmax,:], DataFrame(idx = idxlodmax, maxlod = mlodmax[:,2]));

    # filter results according to the chromosome names in geno file
    pheno_gmap_lod = filter(:phenocovar_chr => in(vChrNames), pheno_gmap_lod);

    # filter according to LOD threshold
    pheno_gmap_lod = filter(row -> row.maxlod > thr, pheno_gmap_lod);

    return  pheno_gmap_lod.acc_mb, pheno_gmap_lod.acc_phenocovar_mb,  pheno_gmap_lod.maxlod, steps, vChrNames
end



function plot_eQTL(multiLODs::Array{Float64, 2}, 
                   gmap::DataFrame, phenocovar::DataFrame;
                   thr::Float64 = 5.0, kwargs...)

    maxLODs_allTraits = mapslices(x -> findmax(x), multiLODs; dims = 1);
    maxLODs_allTraits = reduce(vcat, vec(maxLODs_allTraits));
    lodc = Array{Float64, 2}(undef, size(multiLODs, 2), 2);

    for i in 1:size(multiLODs, 2)
        lodc[i, 1] = maxLODs_allTraits[i][2];
        lodc[i, 2] = maxLODs_allTraits[i][1];
    end

    x, y, z, mysteps, mychr = get_eQTL_accMb(
                                lodc, 
                                phenocovar,
                                gmap;
                                thr = thr,
                                kwargs...
                              )

    eQTLplot(x, y, z, mysteps, mychr, kwargs...)

end
