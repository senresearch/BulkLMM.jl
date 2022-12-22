# Functions for eQTL plots of multiple-trait scan results (beta version):

using DataFrames, CSV, DelimitedFiles
using Plots, Plots.PlotMeasures, PlotlyBase, PlotlyKaleido
using ColorSchemes, Colors
gr()


## Match pheno and geno info dataframes by common chromosomes:
function matchPhenoAndGenoInfos(pInfo::DataFrame, gInfo::DataFrame;
                                chrColname::String = "Chr", mbColname::String = "Mb")
    
    chrs_g = unique(gInfo[:, Symbol(chrColname)]);
    chrs_p = unique(pInfo[:, Symbol(chrColname)]);
    
    common_chrs = chrs_p[findall(x->x in chrs_g, chrs_p)];
    pInfo_common = filter(Symbol("Chr") => in(common_chrs), pInfo);
    
    return pInfo_common
    
end

function filterLODs_by_Chr(lodc::Array{Float64, 2}, p_names::Array{String, 1}, pInfo::DataFrame)

    idxs = findall(x -> x in unique(pInfo[:, :ProbeSet]), p_names); # PS ids in p_names that exist in pInfo
    lodc[idxs, :];
    
    return p_names[idxs], lodc[idxs, :];
    
end

## Get the information (Mb, Chr) of the marker of maximum LODs:
# gmap - information of all markers
# lodc - results from fitting about the maximum LODs and corresponding markers
function getMaxMarkersInfo(gInfo::DataFrame, lodc::Array{Float64, 2};
                           chrColname::String = "Chr", mbColname::String = "Mb")
    
    chromosomes = unique(gInfo[:, Symbol(chrColname)]); # get the chromosome names;
    gInfo_maxLocus = hcat(gInfo[Int.(lodc[:, 1]), :], DataFrame(lodc, [:LocusID, :LOD]));
    return gInfo_maxLocus;
    
end

function getFinalInfo(pInfo::DataFrame, pNames::Array{String, 1}, results_gInfo::DataFrame)
    
    # filter!()
    p_Info = filter(:ProbeSet => x -> x in pNames, pInfo);
    
    combinedInfo = copy(results_gInfo);
    rename!(combinedInfo, Dict(:Chr => :geno_Chr));
    rename!(combinedInfo, Dict(:Mb => :geno_Mb));
    
    combinedInfo = hcat(p_Info, combinedInfo);
    rename!(combinedInfo, Dict(:Chr => :pheno_Chr));
    rename!(combinedInfo, Dict(:Mb => :pheno_Mb));
    
    sort!(combinedInfo, [:pheno_Chr, :geno_Chr])
    
    return combinedInfo   
end

function getMbChrLengths(finalInfo::DataFrame)
    
    chrs = unique(finalInfo[:, :pheno_Chr]);
    chrMbLengths = Array{Float64, 1}(undef, length(chrs));
    
    pChr_group = groupby(finalInfo, :pheno_Chr);
    pheno_chrMbLengths = combine(pChr_group, :pheno_Mb => maximum);
    rename!(pheno_chrMbLengths, Dict(:pheno_Chr => :Chr));
    
    gChr_group = groupby(finalInfo, :geno_Chr);
    geno_chrMbLengths = combine(gChr_group, :geno_Mb => maximum);
    rename!(geno_chrMbLengths, Dict(:geno_Chr => :Chr));
    
    summary = innerjoin(pheno_chrMbLengths, geno_chrMbLengths, on = :Chr);
    
    summary[:, :Chr_Mb_length] = map(x -> maximum([summary[x, :pheno_Mb_maximum],
                              summary[x, :geno_Mb_maximum]]), collect(1:20))
    
    acc_chr_length = zeros(size(summary, 1));
    
    for i in 1:size(summary, 1)
        if i == 1
            acc_chr_length[i] = summary[i, :Chr_Mb_length];
        else
            acc_chr_length[i] = acc_chr_length[i-1] + summary[i, :Chr_Mb_length];
        end
    end
    
    summary[:, :Chr_Mb_length_acc] = acc_chr_length;
    
    return summary
    
    
    
end

function calcPositionOnChr!(finalInfo::DataFrame, chr_lengths::DataFrame)
    
    pheno_pos = zeros(size(finalInfo, 1));
    geno_pos = zeros(size(finalInfo, 1));
    
    for i in 1:size(finalInfo, 1)
        p_chr_name = finalInfo[i, :pheno_Chr];
        g_chr_name = finalInfo[i, :geno_Chr];
        
        # pheno
        try p_chr_id = parse(Int64, p_chr_name) # if the current trait/marker is on the chromosome 1-19
            
            if p_chr_id == 1
                pheno_pos[i] = finalInfo[i, :pheno_Mb];
            else
                pheno_pos[i] = finalInfo[i, :pheno_Mb] + chr_lengths[p_chr_id-1, :Chr_Mb_length_acc];
            
            end
            
        catch # if the current trait/marker is on the X chromosome
            
            second_last_chr_id = size(chr_lengths, 1)-1;
            pheno_pos[i] = finalInfo[i, :pheno_Mb] + chr_lengths[second_last_chr_id, :Chr_Mb_length_acc];
            
        end
        
        # geno
        try g_chr_id = parse(Int64, g_chr_name) # if the current trait/marker is on the chromosome 1-19
            
            if g_chr_id == 1
                geno_pos[i] = finalInfo[i, :geno_Mb];
            else
                geno_pos[i] = finalInfo[i, :geno_Mb] + chr_lengths[g_chr_id-1, :Chr_Mb_length_acc];
            end
            
        catch # if the current trait/marker is on the X chromosome
            
            second_last_chr_id = size(chr_lengths, 1)-1;
            geno_pos[i] = finalInfo[i, :geno_Mb] + chr_lengths[second_last_chr_id, :Chr_Mb_length_acc];
            
        end
        
    end
    
    finalInfo[:, :pheno_Pos] = pheno_pos;
    finalInfo[:, :geno_Pos] = geno_pos;
    
end

function ploteQTL_fromMax(maxLODs::Array{Float64, 2}, gmap::DataFrame, phenocovar::DataFrame; 
                          thr::Float64 = 5.0)

    pInfo = matchPhenoAndGenoInfos(phenocovar, gmap);

    pheno_file = string(@__DIR__, "/../data/bxdData/spleen-pheno-nomissing.csv");
    pheno = readdlm(pheno_file, ',', header = false);
    p_names = String.(SubString.(pheno[1, 2:(end-1)], 2, 9));

    filter_phenos = filterLODs_by_Chr(maxLODs, p_names, pInfo);
    p_names_filtered = filter_phenos[1];
    lodc = filter_phenos[2];
    results_gInfo = getMaxMarkersInfo(gInfo, lodc);

    finalInfo = getFinalInfo(pInfo, p_names_filtered, results_gInfo);
    finalInfo = dropmissing!(finalInfo);

    chr_lengths = getMbChrLengths(finalInfo);
    calcPositionOnChr!(finalInfo, chr_lengths);


    ## Plot:
    chrnum = unique(chr_lengths.Chr);
    testdf = filter(:LOD => x -> x > thr, finalInfo)

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

    steps = chr_lengths[:, :Chr_Mb_length_acc];
    xlim = steps[end];
    ylim = xlim
    x = collect(0:xlim)
    y = collect(0:ylim)


    gcolor = cgrad(:blues)
    Plots.vline(steps, color="lightgrey",label="")
    Plots.hline!(steps, color="lightgrey",label="")

    eQTLplot = scatter!(testdf.geno_Pos, testdf.pheno_Pos,
                  framestyle=:box, 
                  marker_z = testdf.LOD, color=gcolor, 
                  legend=false, size=(650, 550), 
                  xticks=(pseudotick(steps), chrnum), 
                  yticks=(pseudotick(steps), chrnum),
                  tickfontsize=8, tick_direction=:out,
                  xlims=(0,xlim), ylims=(0,ylim),
                  xlab="eQTL Position (Chromosome)",
                  ylab="Transcript Position (Chromosome)",
                  grid=false,
                  guidefontsize=15, fontfamily = "Helvetica",
                  marker=6, markerstrokewidth=0.3,
                  bottom_margin=0mm,right_margin=0mm)

    plot(eQTLplot)

end

function ploteQTL(multiLODs::Array{Float64, 2}, gmap::DataFrame, phenocovar::DataFrame;
                  thr::Float64 = 5.0)

    maxLODs_allTraits = mapslices(x -> findmax(x), multiLODs; dims = 1);
    maxLODs_allTraits = reduce(vcat, vec(maxLODs_allTraits));
    lodc = Array{Float64, 2}(undef, size(multiLODs, 1), 2);
    for i in 1:size(multiLODs, 1)
        lodc[i, 1] = maxLODs_allTraits[i][2];
        lodc[i, 2] = maxLODs_allTraits[i][1];
    end

    ploteQTL_fromMax(lodc, gmap, phenocovar; thr = thr)

end