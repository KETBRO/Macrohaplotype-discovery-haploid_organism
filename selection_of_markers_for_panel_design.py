import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import allel
import xarray as xr
from importlib.metadata import version

# --- Setup ---
print(f'Numpy version: {version("numpy")}')
print(f'Pandas version: {version("pandas")}')
print(f'Matplotlib version: {version("matplotlib")}')
print(f'scikit-allel version: {version("scikit-allel")}')

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

windows_df = pd.read_csv("genome_stats.csv", index_col=0)
windows_df = windows_df.sort_values('chrom')
# Add a supplementary column containing the window midpoint, useful for plotting
windows_df = windows_df.assign(midpoint=windows_df.loc[:, ["window_start", "window_end"]].mean(axis=1))
windows_df.head()

pv_regions = pd.read_csv("pv4_core.bed",
                         sep="\t", comment="t", # this avoids very long strings
                         names=["chrom", "chromStart", "chromEnd", "name"])

# Coordinates are stored as 0-based in the file and need to be adjusted
pv_regions[["chromStart", "chromEnd"]] += 1
# Out of convenience, we add a 'length' column and filter out all non-core regions
pv_regions["length"] = pv_regions["chromEnd"] - pv_regions["chromStart"]
pv_core = pv_regions[pv_regions["name"]=="Core"]
pv_core.head()

def get_n_muhap_per_core(df, n_muhap_total):
    core_length = sum(df["length"])
    # First try with a naive distribution proportional to the region size
    n_muhap_per_core = np.round((df["length"]/core_length)*n_muhap_total).astype('int')

    # Check whether the number obtained is what is required (i.e. if there are rounding effects)
    delta = n_muhap_total - sum(n_muhap_per_core)

    # Supplement the smallest (or deplete the largest) regions if an adjustment is required
    if delta > 0: # Less markers than expected, add to the smallest
        smallest = n_muhap_per_core.nsmallest(delta).index
        n_muhap_per_core[smallest] += 1
    elif delta < 0: # More markers than expected, remove from the largest
        largest = n_muhap_per_core.nlargest(-delta).index
        n_muhap_per_core[largest] -= 1
    
    return n_muhap_per_core
	
regions_pal = {"Cen": "blue", "Core":"None", "In":"orange" , "Sub":"red"}
def plot_markers(df, markers_list, regions):

    rcParams = plt.rcParams
    rcParams['font.size'] = 12
    rcParams['axes.labelsize'] = 10

    #plt.rcParams['xtick.labelsize']=10
    #plt.rcParams['ytick.labelsize']=10
    
    fig, ax = plt.subplots(nrows=14, sharex = True, sharey = True, figsize=(10, 10))

    ax[0].set_xlim([0,regions["chromEnd"].max()])
    ax[0].set_ylim([0,1])
    ax[0].get_xaxis().get_major_formatter().set_scientific(False)
    ax[0].get_xaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    markers = df.loc[markers_list]

    for chr_num in range(0,15):
        chr_name = f'PvP01_{chr_num:02}_v1'
    
        chr_regions = regions[regions["chrom"] == chr_name]
        chr_windows = df[df["chrom"] == chr_name]
        chr_markers = markers[markers["chrom"] == chr_name]

        ax[chr_num-1].spines['right'].set_visible(False)
        ax[chr_num-1].spines['top'].set_visible(False)
        ax[chr_num-1].spines['left'].set_visible(False)

        ax[chr_num-1].hlines(0.5,0,np.max(chr_regions["chromEnd"]), color="#CCCCCC",
                              linestyle="dotted", linewidth=0.8, zorder=-1)

        for _, r in chr_regions.iterrows():
            p = mpl.patches.Rectangle((r["chromStart"], -1), r["length"], 2, edgecolor='None', facecolor=regions_pal[r["name"]])
            ax[chr_num-1].add_patch(p)

        ax[chr_num-1].vlines(chr_windows["window_start"],-1,chr_windows["het"], color="#BBBBBB", linewidth=0.8)
        ax[chr_num-1].vlines(chr_markers["window_start"],-1,chr_markers["het"], color="black") 
        ax[chr_num-1].set_ylabel(chr_name, rotation='horizontal', ha='right')
        ax[chr_num-1].set_yticks([])
		
def plot_panel_stats(windows):

    rcParams = plt.rcParams
    rcParams["font.size"] = 10
    rcParams["axes.labelsize"] = 12

    plt.rcParams["xtick.labelsize"]=10
    plt.rcParams["ytick.labelsize"]=10

    fig, ax = plt.subplots(ncols=2, figsize=(10, 5), sharey=True)

    windows.sort_values("window_start", inplace=True) # Guarantee they are sorted by position
    
    # Left - Heterozygosity per window and average of the candidate panel
    ax[0].spines["right"].set_visible(False)
    ax[0].spines["top"].set_visible(False)

    ax[0].set_xlabel("Heterozygosity")
    ax[0].set_ylabel("")

    het_values = windows["het"] 
    
    ax[0].hist(het_values,bins=10)
    ax[0].axvline(het_values.mean(), color="darkred", linestyle="dashed", linewidth=1)
    ax[0].axvline(het_values.median(), color="navy", linestyle="dashed", linewidth=1)
    ax[0].set_xlim(0.5,1)

    # Right - Distance between consecutive markers
    ax[1].spines["right"].set_visible(False)
    ax[1].spines["top"].set_visible(False)

    ax[1].set_xlabel("Distance between consecutive windows")
    ax[1].set_ylabel("")

    # Calculate distance between widnows on each chromosome separately
    dist_values = windows.groupby(["chrom"])["midpoint"].diff()
    dist_values.dropna(inplace=True)
    
    ax[1].hist(dist_values,bins=10)
    ax[1].axvline(dist_values.mean(), color="darkred", linestyle="dashed", linewidth=1)
    ax[1].axvline(dist_values.median(), color="navy", linestyle="dashed", linewidth=1)
    ax[1].set_xlim(0,1.5e6)

    # Add text
    _, max_ylim = plt.ylim()
    ax[0].text(het_values.mean()*1.01, max_ylim*0.97, "Mean: {:.2f}".format(het_values.mean()), color="darkred")
    ax[0].text(het_values.median()*1.01, max_ylim*1.01, "Median: {:.2f}".format(het_values.median()), color="navy")
    ax[1].text(dist_values.mean()*1.1, max_ylim*0.97, "Mean: {:,.0f}".format(dist_values.mean()), color="darkred")
    ax[1].text(dist_values.median()*1.1, max_ylim*1.01, "Median: {:,.0f}".format(dist_values.median()), color="navy")
	
def get_greedy_muhap_list(df, core, n_muhap_per_core, min_distance=1_000):
    
    muhap_list = []

    # Iterate through the core regions one by one
    for (_, row), n_core in zip(core.iterrows(), n_muhap_per_core):
        # Select all windows in the current region and sort them
        core_windows = df[ (df["chrom"] == row["chrom"]) &
                           (df["window_start"] >= row["chromStart"]) &
                           (df["window_end"] <= row["chromEnd"]) ].copy()
        
        local_list = []

        while len(local_list) < n_core:
            # Sort by heterozygosity and pick the top - the "greedy" part
            core_windows.sort_values("het", ascending=False, inplace=True)
            if not np.isnan(core_windows.iloc[0]["midpoint"]):
                local_list += [core_windows.index[0].tolist()] # Add the top window to the list
                midpoint = core_windows.iloc[0]["midpoint"]
                # Mask (i.e. ignore in the next iteration) all windows within min_distance from the current one
                # While the greedy approach doesn't explicitly consider spacing, this small tweak allows to avoid
                # considering clusters of variants multiple times
                core_windows.mask( abs(core_windows["midpoint"] - midpoint) < min_distance, inplace=True)
            else: # If the top marker is NaN, then we are out of options
                print(f'Only {len(local_list)} out of {n_core} windows found in \
{row["chrom"]}:{row["chromStart"]}-{row["chromEnd"]} - try decreasing min_distance')
                break
   
        muhap_list += local_list
    
    return muhap_list
	

windows_subset = windows_df[windows_df["variant_counts"].between(3,10)]
print(len(windows_subset))

n_muhap_total = 100
n_muhap_per_core = get_n_muhap_per_core(pv_core, n_muhap_total)
greedy_muhap_list = get_greedy_muhap_list(windows_subset, pv_core, n_muhap_per_core, 10000)

plot_markers(windows_subset,greedy_muhap_list, pv_regions.sort_values(["chrom","chromStart"]))

plot_panel_stats(windows_subset.loc[greedy_muhap_list])

def get_max_idx(df):
    return df.idxmax()

def get_even_muhap_list(df, core, n_muhap_per_core, spacing=1000, verbose=False):

    muhap_list = []

    # Iterate through the core regions one by one
    for (_, row), n_core in zip(core.iterrows(), n_muhap_per_core):
        # Select all windows in the current region and sort them
        core_windows = df[ (df["chrom"] == row["chrom"]) &
                           (df["window_start"] >= row["chromStart"]) &
                           (df["window_end"] <= row["chromEnd"]) ].copy()

        core_windows.sort_values("window_start", inplace=True)
        
        local_list = []

        if n_core > 0: # Check that the current region requires at least 1 marker
            # Calculate the size of the sliding window based on the size of the region, the number of markers,
            # and the required minimum buffer spacing between them
            win_size = np.max([0, np.ceil(row["length"]/n_core).astype("int") - spacing])
            # Run a sliding window in the region and for each window get the index of the highest heterozygosity marker
            local_list, a, NN = allel.windowed_statistic(core_windows["midpoint"].astype("int"), core_windows["het"], get_max_idx,
                                                         size=win_size, step=win_size+spacing,
                                                         start=row["chromStart"], stop=row["chromEnd"])
            if np.isnan(local_list).any(): # Check if at least one sliding window didn't find any marker
                print(f'Only {np.count_nonzero(NN)} out of {n_core} windows found in \
{row["chrom"]}:{row["chromStart"]}-{row["chromEnd"]}')
                if verbose:
                    print(f'L:{row["length"]}, W:{win_size}, N:{n_core}')
                    print(a)
                    print(NN)
                local_list = local_list[~np.isnan(local_list)].astype("int")

        muhap_list = [*muhap_list, *local_list]
    
    return muhap_list
	

windows_subset = windows_df[windows_df["variant_counts"].between(3,10)]
n_muhap_total = 100
n_muhap_per_core = get_n_muhap_per_core(pv_core, n_muhap_total)
even_muhap_list = get_even_muhap_list(windows_subset, pv_core, n_muhap_per_core, 25000)
len(even_muhap_list)
plot_markers(windows_subset,even_muhap_list, pv_regions.sort_values(["chrom","chromStart"]))
plot_panel_stats(windows_subset.loc[even_muhap_list])

#=====================================================================
variant_coords = pd.read_csv("variants_selected.tsv", sep="\t")
variant_coords.head()

variant_dataset = xr.open_zarr("variant_dataset.zarr")
print(variant_dataset)

# function below to map variants to windows they belong to and return the position (absolute and relative) of all the variants it containes.
# Here is the "variants" inputs from variants_selected.tsv and
# windows from "genome_stats.csv".
def win_to_variants(windows, variants):

    # Intialise output dataframe
    df = pd.DataFrame(columns=["id","start","end","length","number","variants"])
    
    for idx, win in windows.iterrows(): # For each selected window
        # Find all variants that fall within the window
        var_idx = np.where((variants["variant_chrom"] == win["chrom"]) &
                           (variants["variant_position"].between(win["window_start"],win["window_end"])))[0]

        n_var = len(var_idx)
        # Throw an error in case the number of variants found doesn't match the expectation
        assert n_var==win["variant_counts"], "Unexpected number of variants - are all files in sync?"
        
        # The microhaplotype unique ID will have the following format:
        # <crhomosome>:<position of variant #1>:[<position of variant #2 relative to #1>, ...]
        relative_pos = variants.loc[var_idx[1:]]["variant_position"] - variants.loc[var_idx[0]]["variant_position"]
        mhap_id = f'{":".join(str(x) for x in variant_coords.loc[var_idx[0]])}:{",".join(str(x) for x in relative_pos)}'
        
        mhap_start = variants.loc[var_idx[0]]["variant_position"]
        mhap_end = variants.loc[var_idx[-1]]["variant_position"]
        mhap_length = mhap_end - mhap_start

        # For convenience, also include the absolute position of all variants in the window
        var_list=[]
        for v in var_idx:
            var_list += [":".join(str(x) for x in variant_coords.loc[v])]

        # Add a new row to the dataframe. The index matches the one in the 'windows' dataframe, useful for merging
        df.loc[idx] = {"id":mhap_id, "start":mhap_start, "end":mhap_end, "length":mhap_length,
                       "number":n_var, "variants":";".join(var_list)}
        

    return df

# Paneljudge treats each microhaplotype as a single variant rather than a collection of individual ones. The microhaplotype frequencies are a required input to run the simulations and the function below calculates them.
# Here is the variant input from variant_dataset.
def haplotype_freq(amplicon, variants, seed=250523):
    
    muhap = amplicon.split(";")
    va=variants.sel(variants=muhap)
    gt = allel.GenotypeArray(va["call_genotype"].data)
    
    np.random.seed(seed)
    # haplo = gt.haploidify_samples().astype(str)
    haplo = gt[..., 0].astype(str) #We use this for manual genotype calcualtion
    haplo_str = ["".join(x) for x in haplo.transpose()]
    print(haplo_str)
    haplo_non_missing = [h for h in haplo_str if "-" not in h]
    values, counts = np.unique(haplo_non_missing, return_counts=True)
    freqs = np.sort(counts/np.sum(counts))[::-1]
    freqs = np.round(freqs,decimals=5)
    
    return freqs

windows_subset = windows_df[(windows_df["variant_counts"].between(3,10))&(windows_df["het"] >= 0.6)]
len(windows_subset)

n_muhap_per_core = get_n_muhap_per_core(pv_core, 50)
muhap_list = get_even_muhap_list(windows_subset, pv_core, n_muhap_per_core, spacing=10000)
len(muhap_list)
plot_markers(windows_subset,muhap_list, pv_regions.sort_values(["chrom","chromStart"]))
plot_panel_stats(windows_subset.loc[muhap_list])
panel=win_to_variants(windows_subset.loc[muhap_list], variant_coords)
panel.head()
panel["number"].sum()


variables_to_remove = list(variant_dataset.keys()) # Add all variable names to remove list...
variables_to_remove.remove("call_genotype") # ...but we need to keep the genotypes
reduced_variant_set = variant_dataset.drop_vars(variables_to_remove) # Reduce the dataset

# Pre-load the data to avoid continuous query to the cloud. Not strictly speaking necessary.
ids = [f"{c}:{p}" for c,p in zip(variant_dataset.coords["variant_chrom"].values,
                                 variant_dataset.coords["variant_position"].values)]

print(ids)
print("Sample ids:", ids[:20])
reduced_variant_set["id"]=("variants",ids)
all_variants = reduced_variant_set.set_index(variants="id")
all_variants
print(all_variants.variants.values[:20])
len(ids) == len(set(ids))

# Here panel is the output from "win_to_variants" function 
def create_pj_files(panel, variants):

    # Initialise the dataframe
    pj = pd.DataFrame(columns=["Amplicon_name","Chr","Start","Stop","length","pos","chrom","distances","freqs"],index=panel["id"])

    # Interate through each window/microhaplotype
    for _, amplicon in panel.iterrows():

        # Convert chromosome names to number (e.g. 'PvP01_07_v1' -> 13)
        chr_name = amplicon["id"].split(":")[0]
        chr_num = int(chr_name.split("_")[1])

        # Fill in all fields required by paneljudge
        pj.loc[amplicon["id"]]["Amplicon_name"] = amplicon["id"]
        pj.loc[amplicon["id"]]["Chr"] = chr_name
        pj.loc[amplicon["id"]]["Start"] = amplicon["start"]
        pj.loc[amplicon["id"]]["Stop"] = amplicon["end"]
        pj.loc[amplicon["id"]]["length"] = amplicon["end"]-amplicon["start"]
        pj.loc[amplicon["id"]]["pos"] = np.floor(np.mean([amplicon["start"],amplicon["end"]])).astype("int")
        pj.loc[amplicon["id"]]["chrom"] = chr_num
        # This step is the most demanding as it requires access to the genomic data
        pj.loc[amplicon["id"]]["freqs"] = haplotype_freq(amplicon["variants"], variants)

        # Finally, calculate the forward intra-markers distance
        for i, _ in enumerate(pj.iterrows()):
            if i < (len(pj)-1):
                if pj.iloc[i]["chrom"] == pj.iloc[i+1]["chrom"]:
                    pj.iloc[i]["distances"] = pj.iloc[i+1]["pos"]-pj.iloc[i]["pos"]
                else:
                    pj.iloc[i]["distances"] = "Inf"
            else:
                pj.iloc[i]["distances"] = "Inf"
    
    return pj

pj_df = create_pj_files(panel,all_variants)
# Save the markers file

pj_df.drop(columns="freqs").to_csv("./precomputed/markers.tsv", index_label="", sep="\t")
# paneljudge requires the frequencies to be linearised (one row per microhaplotype, one column per allele)
# The next lines calculates how many columns are required (max number of alleles across all microhaplotypes)
pj_df.to_csv("marker.tsv", index_label="", sep="\t")
ff = pj_df["freqs"].to_list()
ff = [np.atleast_1d(x) for x in ff] #(We use this if manual genotype calcualtion in frquency function)
max_alleles = np.max([i.size for i in ff])
# Create and save the frequencies dataframe
pj_freqs = pd.DataFrame.from_records(ff,index=pj_df["Amplicon_name"],
                                    columns=[f"Allele.{i}" for i in np.arange(max_alleles)]).fillna(0)
pj_freqs.to_csv("./precomputed/frequencies.tsv", index_label="", sep="\t")
pj_freqs.head()

#=============================================
#=============================================

# wi=windows_subset.loc[muhap_list]
# print(wi)
#     # Intialise output dataframe
# df = pd.DataFrame(columns=["id","start","end","length","number","variants"])
    
# for idx, win in wi.iterrows(): # For each selected window
#         # Find all variants that fall within the window
#         var_idx = np.where((variant_coords["variant_chrom"] == win["chrom"]) &
#                            (variant_coords["variant_position"].between(win["window_start"],win["window_end"])))[0]

#         print(var_idx)
#         n_var = len(var_idx)
#         print(n_var)
#         print(win["variant_counts"])
#         print(win["window_start"],win["window_end"])
#         # Throw an error in case the number of variants found doesn't match the expectation
#         assert n_var==win["variant_counts"], "Unexpected number of variants - are all files in sync?"
        
#         # The microhaplotype unique ID will have the following format:
#         # <crhomosome>:<position of variant #1>:[<position of variant #2 relative to #1>, ...]
#         relative_pos = variant_coords.loc[var_idx[1:]]["variant_position"] - variant_coords.loc[var_idx[0]]["variant_position"]
#         mhap_id = f'{":".join(str(x) for x in variant_coords.loc[var_idx[0]])}:{",".join(str(x) for x in relative_pos)}'
        
#         mhap_start = variant_coords.loc[var_idx[0]]["variant_position"]
#         mhap_end = variant_coords.loc[var_idx[-1]]["variant_position"]
#         mhap_length = mhap_end - mhap_start

#         # For convenience, also include the absolute position of all variants in the window
#         var_list=[]
#         for v in var_idx:
#             var_list += [":".join(str(x) for x in variant_coords.loc[v])]

#         # Add a new row to the dataframe. The index matches the one in the 'windows' dataframe, useful for merging
#         df.loc[idx] = {"id":mhap_id, "start":mhap_start, "end":mhap_end, "length":mhap_length,
#                        "number":n_var, "variants":";".join(var_list)}
#         print(df["variants"])
#         df.to_csv("filename.csv", index=False)
        
# # Load CSV of amplicon variant IDs
# amplicon1 = pd.read_csv("df.csv", delimiter=",")

# # If your CSV has a column called "variants" that contains entries like:
# # "PvP01_01_v1:116703;PvP01_01_v1:116734;PvP01_01_v1:116755"
# # then you want to loop over those entries
# for i, row in amplicon1.iterrows():
#     amplicon = row["variants"]   # this is a semicolon-separated string
#     print(f"\n▶ Processing amplicon {i}: {amplicon}")

#     # Split variant IDs
#     muhap = amplicon.split(";")

#     # Select corresponding variants from your dataset
#     va = all_variants.sel(variants=muhap)

#     # Convert genotypes
#     gt = allel.GenotypeArray(va["call_genotype"].data)
#     print(gt)

#     # Haploidify (flatten diploid genotypes into haplotypes)
#     # haplo = gt.haploidify_samples().astype(str)
#     # print("haplo", haplo)
    
#     haplo = gt[..., 0].astype(str)

#     # Convert to strings per sample (e.g. "010101")
#     haplo_str = ["".join(x) for x in haplo.transpose()]
#     print("tranpo_haplo", haplo_str)

#     # Remove missing haplotypes (those containing "-")
#     haplo_non_missing = [h for h in haplo_str if "-" not in h]
#     print(haplo_non_missing)

#     # Count unique haplotypes and frequencies
#     values, counts = np.unique(haplo_non_missing, return_counts=True)
#     freqs = np.sort(counts / np.sum(counts))[::-1]
#     freqs = np.round(freqs, decimals=5)

#     print("Haplotypes:", values)
#     print("Frequencies:", freqs)



