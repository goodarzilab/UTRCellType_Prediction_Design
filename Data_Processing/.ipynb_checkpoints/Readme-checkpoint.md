# UTR Activity Prediction from MPRA Data

This repository contains code and data for preprocessing Massively Parallel Reporter Assay (MPRA) data to predict UTR (Untranslated Region) activity across different cell lines.

## Dataset Overview

### 3' UTR Data
- **Mattvei's dataset (2023)**: `SourceData/UTR3_sequence_counts_05_23_23.tsv`
- **Timo's dataset (2025)**: 
 - Count data: `SourceData/Timo_3UTR_count_data/`
 - Sequence reference: `UTR_3_library_4_20_22_modified_revcomp.fasta`

### 5' UTR Data
- **Mattvei's dataset (2023)**: `SourceData/UTR5_sequence_counts_05_23_23.tsv`
- **Timo's dataset (2025)**: 
 - Count data: `SourceData/Timo_5UTR_count_data/`
 - Sequence reference: `SourceData/UTR_5_library_5_15_22_modified.fasta`

## Data Preprocessing Pipeline

### Scripts
- **Main notebooks**: 
 - `MPRA_preprocessing_UTR3.ipynb` - Processes 3' UTR data
 - `MPRA_preprocessing_UTR5.ipynb` - Processes 5' UTR data
- **Utilities**: `utils/` - Supporting modules for preprocessing

### Processing Steps
1. **Data Loading**: Import MPRA data from both Mattvei's and Timo's experiments
2. **NA Filtering**: Remove entries with missing values
3. **Quality Control**: Filter low-quality sequences based on read count thresholds
  - Default threshold: count < 100
  - Alternative threshold: 50% percentile
4. **Sequence Annotation**: Map sequences to Timo's data using FASTA files
5. **Data Integration**: Merge Mattvei's and Timo's datasets
6. **Feature Calculation**: Compute Mass of Center (**Mass**) for each sequence per cell line
7. **Data Splitting**: Create train/validation sets (90:10 ratio)
8. **Differential Analysis**: Calculate activity differences (**Diff**) across cell lines
9. **Visualization**: Generate plots for data exploration

## Output Files

### Processed Datasets
- **50% threshold filtering**:
 - `vJun25_3UTR_Timo_Matt_Merged_data.csv`
 - `vJun25_5UTR_Timo_Matt_Merged_data.csv`
- **Count < 100 filtering**:
 - `vJun26_3UTR_Timo_Matt_Merged_data_mincount100.csv`
 - `vJun26_5UTR_Timo_Matt_Merged_data_mincount100.csv`

### Visualizations
- `*.pdf` files containing dataset visualizations

### Cell Line Analysis
Individual analysis results organized by dataset and UTR type:
- `Timo_5UTR_Data/`
- `Timo_3UTR_Data/`
- `Matt_5UTR_Data/`
- `Matt_3UTR_Data/`

## Usage

1. Ensure all source data files are in the `SourceData/` directory
2. Run the appropriate preprocessing notebook for your UTR type
3. Processed data will be saved to the output directories
4. Check the PDF files for quality control visualizations