#!/usr/bin/env python3

import os
import glob
import logging
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
import allel
from importlib.metadata import version

# Check package compatibility (useful for debugging)
try:
    print(f'Numpy version: {version("numpy")}')
    print(f'Pandas version: {version("pandas")}')
    print(f'scikit-allel version: {version("scikit-allel")}')
    print(f'Matplotlib version: {version("matplotlib")}')
    print(f'Seaborn version: {version("seaborn")}')
except:
    pass # Ignore if version check fails

# -----------------------------
# Config
# -----------------------------
zarr_path = "india72.PASS.core.zarr"       # path to your Zarr store
PRECOMPUTED_DIR = "./precomputed"
RESULTS_DIRECTORY = os.path.join(PRECOMPUTED_DIR, "sliding_window_results")
FINAL_STATS_FILE = os.path.join(PRECOMPUTED_DIR, "genome_stats.csv")

os.makedirs(PRECOMPUTED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIRECTORY, exist_ok=True)

# Window parameters
window_length = 200
step = 50

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logging.info("Starting Microhaplotype Pipeline.")

# -----------------------------
# Load arrays directly from Zarr (lazy with Dask)
# -----------------------------
logging.info(f"Loading data from Zarr store: {zarr_path}")

try:
    # Variant/site arrays
    variant_chrom = da.from_zarr(zarr_path, component="variants/CHROM")
    variant_position = da.from_zarr(zarr_path, component="variants/POS")
    variant_numalt = da.from_zarr(zarr_path, component="variants/numalt")
    variant_is_snp = da.from_zarr(zarr_path, component="variants/is_snp")
    variant_filter_pass = da.from_zarr(zarr_path, component="variants/FILTER_PASS")

    # Genotypes
    call_genotype = da.from_zarr(zarr_path, component="calldata/GT")

    # Sample IDs as a lazy Dask object coordinate
    def _to_object_str(block, encoding="utf-8", errors="strict"):
        if block.dtype.kind == "S":  # fixed-width bytes
            return np.char.decode(block, encoding=encoding, errors=errors).astype(object)
        return block.astype(object)

    sample_id_da = da.from_zarr(zarr_path, component="samples").map_blocks(_to_object_str, dtype=object)

except Exception as e:
    logging.error(f"Failed to load Zarr components. Check if {zarr_path} exists and is structured correctly.")
    logging.error(f"Details: {e}")
    exit()

# -----------------------------
# Build Dataset
# -----------------------------
variant_dataset = xr.Dataset(
    data_vars=dict(
        variant_numalt=xr.DataArray(variant_numalt, dims=("variants",)),
        variant_is_snp=xr.DataArray(variant_is_snp.astype(bool), dims=("variants",)),
        variant_filter_pass=xr.DataArray(variant_filter_pass.astype(bool), dims=("variants",)),
        call_genotype=xr.DataArray(call_genotype, dims=("variants", "samples", "ploidy")),
    ),
    coords=dict(
        variant_position=("variants", variant_position),
        variant_chrom=("variants", variant_chrom),
        sample_id=("samples", sample_id_da),
    ),
)

print(variant_dataset)

#==============================================================================

va = variant_dataset.reset_coords(["variant_chrom","variant_position"])
va[["variant_chrom","variant_position"]].to_pandas().to_csv("variants_selected.tsv", index=False, sep="\t")
#==============================================================================

def filter_variants(variant_dataset, field, value):
    logging.info(f"Filtering variants: '{field}=={value}'")
    filter_values = (variant_dataset[field] == value).data
    variant_dataset_filtered = variant_dataset.isel(variants=filter_values)
    return variant_dataset_filtered


def variant_positions(positions):
    return list(positions)


def unique_allele_counts_in_window(gt):
    unique, index, counts = np.unique(gt, axis=1, return_counts=True, return_index=True)
    # Find index with the missing or het
    alleles_with_missing = []
    alleles_with_het = []
    for i in range(len(index)):
        if -1 in gt[:, index[i]].compute():
            alleles_with_missing.append(i)
        if gt[:, int(index[i])].is_het().compute().any():
            alleles_with_het.append(i)
            
        # Convert to numpy arrays with consistent types
    counts_array = np.array(counts, dtype=object)
    missing_array = np.array(alleles_with_missing, dtype=object)
    het_array = np.array(alleles_with_het, dtype=object)

    return counts_array, missing_array, het_array


def calculate_stats(variant_dataset, window_length, step, chrom):
    pos = np.array(list(variant_dataset.indexes['variants']))

    # Count variants that are in each window
    logging.info(f"Counting variants per-windows in {chrom}")    
    n_variants, windows = allel.windowed_count(
        pos, size=window_length, step=step, start=1
    )

    # Filter out windows that don't contain any variant
    logging.info(f"Filtering and de-duplicating windows in {chrom}")
    index_with_variants = [i for i, var in enumerate(n_variants) if var != 0]
    window_with_variants = [list(windows[i]) for i in index_with_variants]

    # Find windows with unique variants - skip any windows that have replicate set of variants
    positions_list = []
    for window in window_with_variants:
        mask = (pos >= window[0]) & (pos <= window[1])
        positions_in_window = pos[mask]
        positions_list.append(list(positions_in_window))

    seen = set()
    unique_indices = []
    for i, p in enumerate(positions_list):
        p_tuple = tuple(p)
        if p_tuple not in seen:
            seen.add(p_tuple)
            unique_indices.append(i)
    unique_windows = [window_with_variants[i] for i in unique_indices]
        
    logging.info(f"Number of unique windows: {len(unique_windows)}")
    
    # Count occurrences of each unique allele
    logging.info(f"Genotyping windows in {chrom}")
    values = allel.GenotypeDaskArray(variant_dataset["call_genotype"].data.astype(int))
    

    allele_counts_list = []
    n_variants_list = []
    for window in unique_windows:
        mask = (pos >= window[0]) & (pos <= window[1])
        gt_window = values[mask]
        counts, missing, het = unique_allele_counts_in_window(gt_window)
        allele_counts_list.append((counts, missing, het))
        n_variants_list.append(np.sum(mask))

    n_variants = np.array(n_variants_list)
    allele_counts = allele_counts_list
    windows = np.array(unique_windows)

    return n_variants, allele_counts, windows


def evaluate_marker_options(variant_dataset, chrom, window_length=200, step=50):

    # Filter variants to chromosome and set index
    variant_dataset = filter_variants(variant_dataset, "variant_chrom", chrom)
    variant_dataset = variant_dataset.set_index(
        variants="variant_position", samples="sample_id"
    )

    logging.info(f"Starting window scanning for {chrom}")

    # Calculate per-window statistics
    n_variants, allele_counts, windows = calculate_stats(
        variant_dataset, window_length, step, chrom
    )

    logging.info(f"Window scanning completed for {chrom}")
    
    # Format results
    window_start = list(windows[:, 0])
    window_end = list(windows[:, 1])
    variant_counts = list(n_variants)
    unique_allele_counts = [x[0] for x in allele_counts]
    unique_alleles_with_missing = [x[1] for x in allele_counts]
    unique_alleles_with_het = [x[2] for x in allele_counts]

    return (
        variant_counts,
        unique_allele_counts,
        unique_alleles_with_missing,
        unique_alleles_with_het,
        window_start,
        window_end,
    )


def run_per_chromosome(chromosomes, variants, results_dir):

    for chrom in chromosomes:
    
        logging.info(f"Starting chromosome {chrom}")
        
        # Calculate window stats
        (
            variant_counts,
            unique_allele_counts,
            unique_alleles_with_missing,
            unique_alleles_with_het,
            window_start,
            window_end,
        ) = evaluate_marker_options(variants, chrom)
    
        # Format data
        df = pd.DataFrame(
            data={
                "chrom": chrom,
                "window_start": window_start,
                "window_end": window_end,
                "variant_counts": variant_counts,
                "unique_allele_counts": unique_allele_counts,
                "unique_alleles_with_missing_index": unique_alleles_with_missing,
                "unique_alleles_with_het_index": unique_alleles_with_het,
            }
        )
    
        # Calculate entropy and heterozygosity
        unique_allele_count = []
        unique_allele_freqs = []
        entropy = []
        het = []
        df_with_stats = df.copy()
        
        for index, row in df.iterrows():
            gt_counts = np.asarray(row.unique_allele_counts)
            n_alleles = len(gt_counts)
            unique_allele_count.append(n_alleles)
            if n_alleles == 0:
                unique_allele_freqs.append([])
                entropy.append(0.0)
                het.append(0.0)
            else:
                gt_freqs = gt_counts / np.sum(gt_counts)
                unique_allele_freqs.append(list(gt_freqs))
                entropy_val = 0.0
                for f in gt_freqs:
                    if f > 0:
                        entropy_val -= f * np.log(f)
                entropy.append(entropy_val)
                het.append(1.0 - np.sum(gt_freqs ** 2))
            
        df_with_stats["unique_allele_frequencies"] = unique_allele_freqs
        df_with_stats["unique_allele_count"] = unique_allele_count
        df_with_stats["entropy"] = entropy
        df_with_stats["het"] = het
        
        # Output to csv
        logging.info(f"Writing results for chromosome {chrom}")
        df_with_stats.to_csv(os.path.join(results_dir,f"{chrom}_windowed_stats.csv"))
        
    return True

    
results_directory = "./precomputed/sliding_window_results"

# Initialise the logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

logging.info("Start")
logging.info(f'Output directory: {results_directory}')

chrom_list = [f'PvP01_{chr_num+1:02}_v1' for chr_num in range(14)]

logging.info(f'Chromosomes: {chrom_list}')

# Reduce the size and pre-load the variants table to speedup computation
variables_to_remove = list(variant_dataset.keys()) # Add all variable names to remove list...
variables_to_remove.remove("call_genotype") # ...but we need to keep the genotypes
reduced_variant_set = variant_dataset.drop_vars(variables_to_remove).load() # Reduce the dataset
print(reduced_variant_set)

va = reduced_variant_set.reset_coords(["variant_chrom","variant_position"])
va[["variant_chrom","variant_position"]].to_pandas().to_csv("variants_selected.tsv", index=False, sep="\t")


run_per_chromosome(chrom_list, reduced_variant_set, results_directory)

# Collate all the chromosome-level files into a single file
logging.info("Collating results")

all_files = glob.glob(os.path.join(results_directory , "*_windowed_stats.csv"))
li = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=0)
    li.append(df)
results_df = pd.concat(li, ignore_index=True)
results_df.to_csv("./precomputed/genome_stats.csv")

logging.info("Completed!")

# # Save the reduced variant dataset as a Zarr store
# reduced_variant_set_path = "./precomputed/reduced_variant_set.zarr"

# logging.info(f"Saving reduced variant dataset to {reduced_variant_set_path}")
# reduced_variant_set.to_zarr(reduced_variant_set_path, mode="w")





















  

