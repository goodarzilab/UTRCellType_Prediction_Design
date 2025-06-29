import pandas as pd
import glob
import numpy as np
from statsmodels.stats.multitest import multipletests
import re
from Bio import SeqIO
from scipy import stats
import os
import matplotlib.pyplot as  plt
import seaborn as sns

import pymc as pm

# Function to read and combine count files for a specific cell line
def create_counts_df(cell_line):
    fraction_dfs = {}

    for i in range(1, 5):
        file_path = f'counts/121924_{cell_line}_Fraction{i}_counts.txt'
        df = pd.read_csv(file_path, sep='\t', header=None, index_col=0)

        if i == 1:
            fraction_dfs[f'Fraction{i}'] = df
        else:
            count_col = df.columns[-1]
            fraction_dfs[f'Fraction{i}'] = df[count_col]

    # Combine all fractions into one dataframe
    combined_df = pd.concat(fraction_dfs, axis=1)

    # Clean up the column names to remove the MultiIndex
    combined_df.columns = [f'Fraction{i}' for i in range(1, 5)]

    return combined_df

def normalize_counts(df):
    """
    Normalize counts similar to DESeq2's method:
    1. Calculate geometric means per gene
    2. Calculate size factors
    3. Normalize counts by size factors
    
    Parameters:
    df: pandas DataFrame where rows are genes and columns are samples
    
    Returns:
    normalized_df: DataFrame with normalized counts
    size_factors: array of size factors used for normalization
    """
    # Replace 0s with NaN to handle log calculations
    df_temp = df.replace(0, np.nan)
    
    # Calculate geometric means for each gene (row)
    geometric_means = np.exp(np.nanmean(np.log(df_temp), axis=1))
    
    # Calculate ratios of each count to geometric mean
    ratios = df_temp.div(geometric_means, axis=0)
    
    # Calculate size factors (median of ratios for each sample)
    size_factors = np.nanmedian(ratios, axis=0)

    # Scale size factors so minimum is 1
    size_factors = size_factors / np.min(size_factors)
    
    # Normalize counts by size factors
    normalized_df = df.div(size_factors, axis=1)
    
    return normalized_df, size_factors

def split_normal_shuffled(df):
    """
    Split dataframe into normal tiles and shuffled tiles based on pattern matching.
    
    Parameters:
    df: pandas DataFrame where index contains tile identifiers
    
    Returns:
    normal_df: DataFrame containing only normal tiles (ending in _Tile_XX_5P)
    shuffled_df: DataFrame containing only shuffled tiles (ending in _Tile_XX_ushuffle_5P)
    """
    # Create boolean masks for filtering
    normal_mask = df.index.str.contains(r'_Tile_\d+_5P$')
    shuffled_mask = df.index.str.contains(r'_Tile_\d+_ushuffle_5P$')
    
    # Split dataframes using masks
    normal_df = df[normal_mask]
    shuffled_df = df[shuffled_mask]

    # Remove '_ushuffle' from shuffled df index
    shuffled_df.index = shuffled_df.index.str.replace('_ushuffle', '')
    
    return normal_df, shuffled_df

def analyze_distribution(df):
    """
    Calculate center of mass and test for deviation from uniform distribution
    for each row, followed by multiple testing correction.
    
    Parameters:
    df: pandas DataFrame with fraction counts (columns are fractions)
    
    Returns:
    result_df: DataFrame with columns:
        - center_of_mass: weighted average position
        - z_score: test statistic
        - p_value: uncorrected p-value
        - adjusted_p: FDR-corrected p-value
        - total_counts: sum of counts across fractions
    """
    # Positions for each fraction (1-based)
    positions = np.array([1, 2, 3, 4])
    
    def analyze_row(row):
        # Calculate center of mass and stats
        total_counts = row.sum()
        if total_counts == 0:
            return pd.Series({
                'center_of_mass': np.nan,
                'z_score': np.nan,
                'total_counts': 0
            })
        
        # Center of mass calculation
        com = np.sum(row * positions) / total_counts
        
        # Statistical test
        # Under null, Var(X) = 1.25 for single observation
        se = np.sqrt(1.25 / total_counts)
        z_score = (com - 2.5) / se
        
        return pd.Series({
            'center_of_mass': com,
            'z_score': z_score,
            'total_counts': total_counts
        })
    
    # Apply analysis to each row
    result_df = df.apply(analyze_row, axis=1)
    
    return result_df

def read_fasta_to_df(fasta_file):
    """
    Read FASTA file and extract sequence and transcript IDs.
    
    Parameters:
    fasta_file: path to FASTA file
    
    Returns:
    DataFrame with sequence IDs, sequences, and transcript IDs
    """
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        # First try ENST pattern
        match = re.match(r'(ENST\d+)', record.id)
        if match:
            transcript_id = match.group(1)
        else:
            # For non-ENST IDs, use the part before _5P or _ushuffle_5P
            # Remove _5P, _ushuffle_5P, and any variant descriptors
            transcript_id = record.id.split('_5P')[0]
            transcript_id = transcript_id.split('_ushuffle')[0]
            # For Frag sequences, keep only the Frag## part
            if transcript_id.startswith('Frag'):
                transcript_id = re.match(r'(Frag\d+)', transcript_id).group(1)
            
        sequences.append({
            'sequence_id': record.id,
            'sequence': str(record.seq),
            'transcript_id': transcript_id
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(sequences)
    
    return df

def read_bed_coordinates(bed_file):
    """
    Read BED file with tile coordinates.
    
    Parameters:
    bed_file: path to BED file
    
    Returns:
    DataFrame with tile coordinates
    """
    # Read BED file
    bed_df = pd.read_csv(bed_file, sep='\t', header=None,
                        names=['chrom', 'start', 'end', 'name', 'score', 'strand'])

    # Add 'chr' prefix to chromosome names if not already present
    bed_df['chrom'] = bed_df['chrom'].apply(lambda x: f'chr{x}' if not str(x).startswith('chr') else str(x))
    
    # Set name as index to match with results dataframe
    bed_df = bed_df.set_index('name')
    
    return bed_df

def combine_overlapping_fragments(merged_df):
    """
    Split overlapping fragments into non-overlapping segments and combine z-scores.
    Uses adjusted z-scores for combination and corrects inflation using genomic control.
    """
    segments = []
    
    # Process each chromosome separately
    for chrom in merged_df['chrom'].unique():
        chrom_df = merged_df[merged_df['chrom'] == chrom]
        breakpoints = sorted(set(chrom_df['start'].tolist() + chrom_df['end'].tolist()))
        
        for i in range(len(breakpoints)-1):
            start = breakpoints[i]
            end = breakpoints[i+1]
            
            overlapping = chrom_df[
                (chrom_df['start'] <= start) & (chrom_df['end'] >= end)
            ]
            
            if len(overlapping) > 0:
                n = len(overlapping)
                combined_z = overlapping['adjusted_z_score'].sum() / np.sqrt(n)
                
                segment = {
                    'chrom': chrom,
                    'start': start,
                    'end': end,
                    'n_fragments': n,
                    'fragment_ids': ','.join(overlapping.index),
                    'combined_z_score': combined_z,
                    'strand': overlapping['strand'].iloc[0]
                }
                segments.append(segment)
    
    segments_df = pd.DataFrame(segments)
    
    # Calculate genomic inflation factor (lambda)
    chi_sq = segments_df['combined_z_score']**2
    lambda_gc = np.median(chi_sq) / 0.456
    
    # Adjust z-scores using genomic control
    segments_df['adjusted_combined_z'] = segments_df['combined_z_score'] / np.sqrt(lambda_gc)
    
    # Add lambda_gc to output
    segments_df.attrs['lambda_gc'] = lambda_gc
    
    return segments_df

def analyze_distribution_with_permutation(df, n_permutations=4, random_state=None):
    """
    Calculate center of mass and z-scores, then adjust using permutation-based null.

    Parameters:
    df: pandas DataFrame with fraction counts
    n_permutations: number of permutations for null distribution
    random_state: int or None, seed for reproducibility

    Returns:
    result_df: DataFrame with original results plus adjusted z-scores
    """
    # Set the random seed
    rng = np.random.default_rng(random_state)

    # Get original results
    orig_results = analyze_distribution(df)

    # Generate null distribution through permutations
    null_zscores = []
    for _ in range(n_permutations):
        # Shuffle fractions within each row
        shuffled_df = df.copy()
        for idx in shuffled_df.index:
            shuffled_df.loc[idx] = rng.permutation(shuffled_df.loc[idx])  # Use rng for shuffling

        # Calculate z-scores for shuffled data
        perm_results = analyze_distribution(shuffled_df)
        null_zscores.extend(perm_results['z_score'])

    # Calculate mean and std of null distribution
    null_mean = np.mean(null_zscores)
    null_std = np.std(null_zscores)

    # Adjust original z-scores
    orig_results['adjusted_z_score'] = (orig_results['z_score'] - null_mean) / null_std

    return orig_results

def merge_neighboring_fragments(segments_df, z_score_threshold=0.5, distance_threshold=40):
    """
    Merge nearby fragments with similar z-scores.
    
    Parameters:
    segments_df: DataFrame with segment information
    z_score_threshold: maximum difference in z-scores to be considered similar
    distance_threshold: maximum distance between fragments to be considered nearby
    
    Returns:
    DataFrame with merged segments
    """
    merged_segments = []
    current_group = None
    
    # Sort by chromosome and start position
    sorted_df = segments_df.sort_values(['chrom', 'start'])
    
    for idx, row in sorted_df.iterrows():
        if current_group is None:
            current_group = {
                'chrom': row['chrom'],
                'start': row['start'],
                'end': row['end'],
                'z_scores': [row['adjusted_combined_z']],
                'fragment_ids': [row['fragment_ids']],
                'strand': row['strand']
            }
        else:
            # Check if this segment should be merged with current group
            same_chrom = row['chrom'] == current_group['chrom']
            close_enough = (row['start'] - current_group['end']) <= distance_threshold
            similar_score = abs(row['adjusted_combined_z'] - np.mean(current_group['z_scores'])) <= z_score_threshold
            
            if same_chrom and close_enough and similar_score:
                # Extend current group
                current_group['end'] = row['end']
                current_group['z_scores'].append(row['adjusted_combined_z'])
                current_group['fragment_ids'].append(row['fragment_ids'])
            else:
                # Save current group and start new one
                if len(current_group['z_scores']) > 0:
                    n = len(current_group['z_scores'])
                    merged_z = sum(current_group['z_scores']) / np.sqrt(n)
                    merged_segments.append({
                        'chrom': current_group['chrom'],
                        'start': current_group['start'],
                        'end': current_group['end'],
                        'merged_z_score': merged_z,
                        'n_fragments': n,
                        'fragment_ids': ','.join(current_group['fragment_ids']),
                        'strand': current_group['strand']
                    })
                current_group = {
                    'chrom': row['chrom'],
                    'start': row['start'],
                    'end': row['end'],
                    'z_scores': [row['adjusted_combined_z']],
                    'fragment_ids': [row['fragment_ids']],
                    'strand': row['strand']
                }
    
    # Don't forget to add the last group
    if current_group is not None and len(current_group['z_scores']) > 0:
        n = len(current_group['z_scores'])
        merged_z = sum(current_group['z_scores']) / np.sqrt(n)
        merged_segments.append({
            'chrom': current_group['chrom'],
            'start': current_group['start'],
            'end': current_group['end'],
            'merged_z_score': merged_z,
            'n_fragments': n,
            'fragment_ids': ','.join(current_group['fragment_ids']),
            'strand': current_group['strand']
        })
    
    merged_df = pd.DataFrame(merged_segments)
    
    # Calculate genomic inflation factor and adjust z-scores
    chi_sq = merged_df['merged_z_score']**2
    lambda_gc = np.median(chi_sq) / 0.456
    
    merged_df['p_value'] = 2 * stats.norm.sf(abs(merged_df['merged_z_score']))
    merged_df['adjusted_p'] = multipletests(merged_df['p_value'], method='fdr_bh')[1]
    
    return merged_df

def compare_real_shuffled_distributions(real_df, shuffled_df):
    """
    Compare distribution patterns between real and shuffled sequences using linear regression.
    Tests for interaction between fraction number and shuffle status.
    Also calculates individual slopes for real and shuffled sequences.
    """
    import statsmodels.api as sm
    
    # Get common indices
    common_idx = real_df.index.intersection(shuffled_df.index)
    
    results = []
    
    # For each sequence that exists in both dataframes
    for idx in common_idx:
        # Get real and shuffled counts
        real_counts = real_df.loc[idx]
        shuffled_counts = shuffled_df.loc[idx]
        
        # Create regression dataframe
        data = []
        for frac in range(1, 5):
            data.append({
                'count': real_counts[f'Fraction{frac}'],
                'fraction': frac,
                'shuffle': 0
            })
            data.append({
                'count': shuffled_counts[f'Fraction{frac}'],
                'fraction': frac,
                'shuffle': 1
            })
        reg_df = pd.DataFrame(data)
        
        # Skip if any counts are NaN
        if reg_df['count'].isna().any():
            continue
            
        # Add interaction term
        reg_df['fraction_shuffle'] = reg_df['fraction'] * reg_df['shuffle']
        
        try:
            # Fit full model
            X = sm.add_constant(reg_df[['fraction', 'shuffle', 'fraction_shuffle']])
            model = sm.OLS(reg_df['count'], X)
            fit = model.fit()
            
            # Fit separate models for real and shuffled
            real_data = reg_df[reg_df['shuffle'] == 0]
            shuffled_data = reg_df[reg_df['shuffle'] == 1]
            
            # Get slope for real sequences
            X_real = sm.add_constant(real_data['fraction'])
            real_fit = sm.OLS(real_data['count'], X_real).fit()
            real_slope = real_fit.params['fraction']
            
            # Get slope for shuffled sequences
            X_shuffled = sm.add_constant(shuffled_data['fraction'])
            shuffled_fit = sm.OLS(shuffled_data['count'], X_shuffled).fit()
            shuffled_slope = shuffled_fit.params['fraction']
            
            results.append({
                'sequence_id': idx,
                'interaction_coef': fit.params['fraction_shuffle'],
                'interaction_pvalue': fit.pvalues['fraction_shuffle'],
                'interaction_tstat': fit.tvalues['fraction_shuffle'],
                'r_squared': fit.rsquared,
                'model_pvalue': fit.f_pvalue,
                'real_slope': real_slope,
                'shuffled_slope': shuffled_slope,
                'slope_difference': real_slope - shuffled_slope
            })
        except:
            continue
    
    results_df = pd.DataFrame(results)
    results_df = results_df.set_index('sequence_id')
    
    # Filter out any NaN p-values before adjustment
    valid_pvals = results_df['interaction_pvalue'].dropna()
    if len(valid_pvals) > 0:
        adjusted_pvals = multipletests(valid_pvals, method='fdr_bh')[1]
        results_df.loc[valid_pvals.index, 'adjusted_p'] = adjusted_pvals
    
    # Print summary statistics
    print(f"Total sequences analyzed: {len(common_idx)}")
    print(f"Sequences with valid results: {len(results_df)}")
    print(f"Sequences with valid p-values: {len(valid_pvals)}")
    
    return results_df

# Function to read and combine count files for a specific cell line
def create_counts_df_custom(cell_line, counts_dir, utr_type = '3UTR', date_str = '030725'):
    """
    Read and combine count files for a specific cell line.
    Uses the naming format: 021425_5UTR_CELLLINE-Fraction{i}_R1_001_counts.txt
    """
    fraction_dfs = {}
    
    for i in range(1, 5):
        # Define file path based on the established pattern
        file_path = f"{counts_dir}/{date_str}_{utr_type}_{cell_line}-Fraction{i}_R1_001_counts.txt"
        
        if not os.path.exists(file_path):
            print(f"Warning: No count file found at {file_path}")
            # Try alternative pattern with glob
            file_pattern = f"*{cell_line}*Fraction{i}*counts.txt"
            matching_files = list(Path(counts_dir).glob(file_pattern))
            
            if not matching_files:
                print(f"Warning: No count file found for {cell_line} Fraction {i}")
                continue
                
            file_path = str(matching_files[0])
        
        print(f"Reading file: {file_path}")
        try:
            df = pd.read_csv(file_path, sep='\t', header=None, index_col=0)
            print(f"Successfully read file with shape: {df.shape}")
            
            # Exclude the '*' entry which represents unaligned reads
            if '*' in df.index:
                print(f"Removing unaligned reads entry ('*') from {file_path}")
                df = df[~df.index.isin(['*'])]
            
            # Rename the column to a standard name
            df.columns = [1]
            
            if i == 1:
                fraction_dfs[f'Fraction{i}'] = df
            else:
                fraction_dfs[f'Fraction{i}'] = df[1]
                
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")
            continue
    
    if not fraction_dfs:
        raise ValueError(f"No count files loaded for cell line {cell_line}")
    
    # Combine all fractions into one dataframe
    print(f"Combining {len(fraction_dfs)} fractions for {cell_line}")
    combined_df = pd.concat(fraction_dfs, axis=1)
    
    # Clean up the column names
    combined_df.columns = [f'Fraction{i}' for i in range(1, len(fraction_dfs) + 1)]
    print(f"Combined dataframe shape: {combined_df.shape}")
    
    return combined_df

# Function to filter rows with missing values
def filter_missing_values(df, cell_line):
    """
    Filter out rows with missing values in any fraction
    Parameters:
    df: DataFrame with count data
    cell_line: Name of cell line (for display)
    Returns:
    Filtered DataFrame
    """
    initial_rows = df.shape[0]
    filtered_df = df.dropna()
    missing_rows = initial_rows - filtered_df.shape[0]
    print(f"{cell_line}: Removed {missing_rows} rows with missing values ({missing_rows/initial_rows:.1%} of data)")
    return filtered_df
# Function to filter based on normalized median read counts
def filter_by_median_norm_counts(norm_df, cell_line, output_dir, threshold = 100):
    """
    Filter normalized count data based on median normalized read counts.
    Creates a plot of median counts and prompts the user for a threshold.
    Parameters:
    norm_df: DataFrame with normalized count data
    cell_line: Name of cell line (for display)
    Returns:
    Filtered DataFrame
    """
    # Calculate median normalized read count per gene/entry
    norm_df_with_median = norm_df.copy()
    norm_df_with_median['median_norm_count'] = norm_df.median(axis=1)
    # Plot median normalized read count distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(norm_df_with_median['median_norm_count'], log_scale=True, kde=False)
    plt.title(f"{cell_line} - Distribution of Median Normalized Read Counts", fontsize=14)
    plt.xlabel("Median Normalized Read Count (log scale)", fontsize=12)
    plt.ylabel("Number of Entries", fontsize=12)
    # Add line for median value
    median_value = norm_df_with_median['median_norm_count'].median()
    plt.axvline(x=median_value, color='red', linestyle='--',
                label=f"Median: {median_value:.1f}")
    # Suggest potential thresholds at different percentiles
    percentiles = [10, 25, 50]
    for p in percentiles:
        threshold_value = norm_df_with_median['median_norm_count'].quantile(p/100)
        plt.axvline(x=threshold_value, color='green', alpha=0.5, linestyle=':',
                    label=f"{p}th percentile: {threshold_value:.1f}")
    plt.legend()
    plt.tight_layout()
    # Save the plot
    output_file = f"{output_dir}/{cell_line}_median_norm_counts.png"
    plt.savefig(output_file)
    print(f"Plot saved to: {output_file}")
    plt.show()
    # Interactive input for threshold
    print(f"\nSuggested thresholds for {cell_line}:")
    print(f"- Median: {median_value:.1f}")
    for p in percentiles:
        threshold_value = norm_df_with_median['median_norm_count'].quantile(p/100)
        print(f"- {p}th percentile: {threshold_value:.1f}")
        
    # Apply the threshold
    filtered_df = norm_df_with_median[norm_df_with_median['median_norm_count'] >= threshold]
    removed_rows = norm_df.shape[0] - filtered_df.shape[0]
    print(f"{cell_line}: Removed {removed_rows} rows below threshold {threshold} ({removed_rows/norm_df.shape[0]:.1%} of data)")
    print(f"Remaining entries: {filtered_df.shape[0]}")
    # Remove the median_count column before returning
    filtered_df = filtered_df.drop(columns=['median_norm_count'])
    return filtered_df

# Define a function to process each cell line
def process_cell_line_timo(cell_line, fasta_df, counts_dir, output_dir, utr_type = '3UTR', date_str = '030725'):
    """
    Process a single cell line with all filtering steps
    """
    try:
        print("\n" + "="*50)
        print(f"Processing {cell_line}")
        print("="*50)
        
        # 1. Load and combine count data
        counts_df = create_counts_df_custom(cell_line, counts_dir, utr_type, date_str)
        print(f"{cell_line} initial shape: {counts_df.shape}")
        
        # 2. Filter out rows with missing values
        filtered_df = filter_missing_values(counts_df, cell_line)
        print(f"{cell_line} shape after filtering missing values: {filtered_df.shape}")
        
        # 3. Normalize counts
        norm_df, factors = normalize_counts(filtered_df)
        print(f"{cell_line} size factors: {factors}")
        
        # 4. Filter by median normalized counts
        filtered_norm_df = filter_by_median_norm_counts(norm_df, cell_line, output_dir)
        print(f"{cell_line} shape after all filtering: {filtered_norm_df.shape}")
        
        # 5. Calculate center of mass and z-scores
        results_df = analyze_distribution_with_permutation(
            filtered_norm_df, n_permutations=4, random_state=42
        )
        print(f"{cell_line} results shape: {results_df.shape}")
        results_df = pd.merge(fasta_df, results_df, left_on = 'ID', right_index = True)
        results_df['cell_type'] = cell_line
        
        # 6. Save results
        output_file = f"{output_dir}/{cell_line.replace('-', '_')}_5UTR_analysis.csv"
        results_df.to_csv(output_file)
        print(f"Results saved to {output_file}")
        
        return results_df
        
    except Exception as e:
        print(f"Error processing {cell_line}: {str(e)}")
        return None

def process_cell_line_matt(counts_df, cell_line, output_dir, utr_type = '3UTR', date_str = '030725'):
    """
    Process a single cell line with all filtering steps
    """
    try:
        print("\n" + "="*50)
        print(f"Processing {cell_line}")
        print("="*50)
        
        # 1. Load and combine count data
        print(f"{cell_line} initial shape: {counts_df.shape}")
        
        # 2. Filter out rows with missing values
        filtered_df = filter_missing_values(counts_df, cell_line)
        print(f"{cell_line} shape after filtering missing values: {filtered_df.shape}")
        
        # 3. Normalize counts
        norm_df, factors = normalize_counts(filtered_df)
        print(f"{cell_line} size factors: {factors}")
        
        # 4. Filter by median normalized counts
        filtered_norm_df = filter_by_median_norm_counts(norm_df, cell_line, output_dir)
        print(f"{cell_line} shape after all filtering: {filtered_norm_df.shape}")
        
        # 5. Calculate center of mass and z-scores
        results_df = analyze_distribution_with_permutation(
            filtered_norm_df, n_permutations=4, random_state=42
        )
        print(f"{cell_line} results shape: {results_df.shape}")

        results_df = results_df.reset_index()
        results_df.rename(columns = {'seq':'sequence'}, inplace = True)

        results_df['cell_type'] = cell_line
        
        # 6. Save results
        output_file = f"{output_dir}/{cell_line.replace('-', '_')}_5UTR_analysis.csv"
        results_df.to_csv(output_file)
        print(f"Results saved to {output_file}")
        
        return results_df
        
    except Exception as e:
        print(f"Error processing {cell_line}: {str(e)}")
        return None

def fasta_to_dataframe(fasta_file):
    """
    读取fasta文件并转换为DataFrame，包含'ID'和'seq'两列
    
    参数:
    fasta_file (str): fasta文件的路径
    
    返回:
    pandas.DataFrame: 包含ID和序列的数据框
    """
    ids = []
    sequences = []
    
    with open(fasta_file, 'r') as f:
        current_id = None
        current_seq = ""
        
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # 发现新的序列头，保存之前的序列
                if current_id is not None:
                    ids.append(current_id)
                    sequences.append(current_seq)
                
                # 开始新的序列
                current_id = line[1:]  # 移除'>'前缀
                current_seq = ""
            else:
                # 累积序列
                current_seq += line
        
        # 保存最后一个序列
        if current_id is not None:
            ids.append(current_id)
            sequences.append(current_seq)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'ID': ids,
        'sequence': sequences
    })
    
    return df

def fit_bayesian_model_3utr(library):
    basic_model = pm.Model()

    with basic_model:
        # Priors for unknown model parameters
        # Mixture weights
        psi = pm.Uniform("psi", lower=0.0, upper=1.0)
        p = pm.Beta("p", alpha=3, beta=1)
        n = pm.Uniform("n", lower=0.0, upper=10.0)

        # Likelihood (sampling distribution) of observations
        likelihood = pm.ZeroInflatedNegativeBinomial('zinb', psi=psi, p=p, n=n, observed=library)

        # draw 1000 posterior samples
        idata = pm.find_MAP()
    return idata

def fit_bayesian_model_5utr(library):
    basic_model = pm.Model()

    with basic_model:
        # Priors for unknown model parameters

        # Mixture weights
        wd = pm.Dirichlet("wd", a=np.ones(2))
        psi = pm.Uniform("psi", lower=0.0, upper=1.0)

        # ZINB (ZEROS)
        p1 = pm.Beta("p1", alpha=3, beta=1)
        n1 = pm.Uniform("n1", lower=0.0, upper=10.0)

        # NEGBIN (POSITIVES)
        p2 = pm.Beta("p2", alpha=1, beta=3)
        n2 = pm.Uniform("n2", lower=0.0, upper=10.0)

        # Mixture
        components = [
            pm.ZeroInflatedNegativeBinomial.dist(psi=psi, p=p1, n=n1),
            pm.ZeroInflatedNegativeBinomial.dist(psi=psi, p=p2, n=n2),
        ]

        # Likelihood (sampling distribution) of observations
        likelihood = pm.Mixture('mixture', w=wd, comp_dists=components, observed=library)

        # draw 1000 posterior samples
        idata = pm.find_MAP()
    return idata
    
def get_reference_rv(psi, p, n):
    mixture_dist = pm.ZeroInflatedNegativeBinomial.dist(psi=psi, p=p, n=n)
    return mixture_dist

def get_discrete_quantile_function(rv, min_value=0, max_value=100_000):
    if min_value != 0:
        raise NotImplementedError
    logcdf = pm.logcdf(rv, np.arange(0, max_value + 1)).eval()
    cdf = np.exp(logcdf)

    def quantile(q: np.array):
        q = np.asarray(q)
        values = np.abs(cdf - q[:, None]).argmin(axis=1)
        return values

    return quantile

def geom_mean_1d(inp_array):
    x = np.log(inp_array[inp_array > 0]).sum() / inp_array.shape[0]
    return np.exp(x)


def normalize_median_of_ratios(inp_df, do_print = False):
    geom_means = inp_df.apply(geom_mean_1d, axis=1)
    ratios_df = inp_df.divide(geom_means, axis=0)
    norm_factors = ratios_df.median(axis=0)
    if do_print:
        print('Normalization factors: ')
        print(norm_factors)
    normalized_df = inp_df.divide(norm_factors, axis=1)
    return normalized_df


def create_dataframes_from_multiindex(matt_data):
    """
    从MultiIndex列的DataFrame创建多个子DataFrame
    
    Args:
        matt_data: 具有MultiIndex列的DataFrame
        
    Returns:
        dict: 包含各个cell_type-replicate组合的DataFrame字典
    """
    matt_results = {}
    
    # 获取所有唯一的cell_type和replicate组合
    cell_rep_combinations = matt_data.columns.droplevel('bin').unique()
    
    for cell_type, replicate in cell_rep_combinations:
        # 选择特定cell_type和replicate的所有bin列
        selected_cols = matt_data.loc[:, (cell_type, replicate, slice(None))]
        
        # 重命名列为Fraction1, Fraction2, Fraction3, Fraction4
        new_column_names = [f'Fraction{bin_num}' for bin_num in selected_cols.columns.get_level_values('bin')]
        selected_cols.columns = new_column_names
        
        # 生成键名
        key_name = f'{cell_type}_Rep{replicate}'
        matt_results[key_name] = selected_cols
    
    return matt_results