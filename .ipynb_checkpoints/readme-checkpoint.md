# UTR Cell Type-Specific Activity Prediction and Generation

This repository contains a complete pipeline for analyzing, predicting, and generating cell type-specific UTR (Untranslated Region) sequences with tailored activity levels.

## Repository Structure
├── Data_Processing/          # MPRA data preprocessing
├── Parade_Prediction/        # PARADE model training and prediction
└── Diffusion_Model_Generation/   # Sequence generation using diffusion models

## Pipeline Overview

### 1. Data Processing (`Data_Processing/`)
Preprocesses Massively Parallel Reporter Assay (MPRA) data to prepare training datasets.

**Key Features:**
- Merges multiple MPRA datasets (Mattvei's 2023 & Timo's 2025)
- Quality control filtering (count threshold: 100 or 50%)
- Calculates Center of Mass (CoM) for activity measurement
- Computes activity differences across cell lines
- Train/validation split (90:10)

**Main Scripts:**
- `MPRA_preprocessing_UTR3.ipynb` - 3' UTR processing
- `MPRA_preprocessing_UTR5.ipynb` - 5' UTR processing

### 2. PARADE Prediction (`Parade_Prediction/`)
Trains deep learning models to predict UTR activity across different cell types.

**Key Features:**
- LegNet-based architecture for sequence activity prediction
- Cell type-specific predictions for 20+ cell lines
- Performance metrics: R², Pearson, and Spearman correlations
- Activity level classification (4 or 5 levels)

**Main Scripts:**
- `vJun26_regression_utr[3/5]_Mass_Diff_mincount100.ipynb` - Model training
- `vJun26_Parade_TrainingData_mincount100_Prediction.ipynb` - Predictions

### 3. Diffusion Model Generation (`Diffusion_Model_Generation/`)
Generates novel UTR sequences with desired cell type specificity and activity levels.

**Key Features:**
- Conditional diffusion model (DDPM)
- Generates sequences for specific cell types and activity levels
- 4-level or 5-level activity classification
- Sequence overlap analysis and filtering

**Main Scripts:**
- `vJun22_UTR_CellLine_DifferentLevel_DM.py` - Diffusion model training
- `vJun26_5UTR_Generate2Parade_4Level_remove50percent.ipynb` - Sequence analysis

## Quick Start

### 1. Prepare Data
```bash
cd Data_Processing/
# Run preprocessing notebooks for your UTR type
```

### 2. Train PARADE Model
```bash
cd ../Parade_Prediction/
# Run training notebooks with your processed data
```

### 3. Generate Sequences
```bash
cd ../Diffusion_Model_Generation/
python vJun22_UTR_CellLine_DifferentLevel_DM.py \
    --file_name /path/to/leveled_data.csv \
    --category_col combined_5_category \
    --prefix experiment_name
```

# Key Outputs

- **Processed Data**: Normalized MPRA measurements with activity levels
- **Trained Models**: PARADE checkpoints for activity prediction
- **Generated Sequences**: FASTA files with cell type-specific UTRs
- **Analysis Results**: Performance metrics, overlap statistics, and visualizations

# Requirements
requirements.txt
Support Python 3.9 and Python 3.12