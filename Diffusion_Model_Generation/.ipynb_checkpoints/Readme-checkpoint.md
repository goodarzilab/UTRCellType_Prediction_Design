# Generate Level Data for Diffusion Model

## Scripts by Quality Threshold

### Count < 100 Threshold
- `vJun26_3UTR_mincount100_DiffusionModel_LevelData.ipynb`
- `vJun26_5UTR_mincount100_DiffusionModel_LevelData.ipynb`

### 50% Threshold
- `vJun25_3UTR_DiffusionModel_LevelData_remove50percent.ipynb`
- `vJun25_5UTR_DiffusionModel_LevelData_remove50percent.ipynb`

## Loading Custom Dataset

You can load your own dataset by modifying cell [2] in the notebook:

```python
utr5_df = pd.read_csv('../Data_Processing/vJun25_5UTR_Timo_Matt_Merged_data.csv', index_col = 0)
```

## Level Assignment

These notebooks assign activity levels to each sequence based on:
- **Center of mass** (activity measurement)
- **Difference** across cell lines

### 4-Level Classification
- **Category 0**: Low mass (<2.5) + Non-positive diff (≤0)
- **Category 1**: Low mass (<2.5) + Positive diff (>0)
- **Category 2**: Medium mass (2.5-3.0) + Non-positive diff (≤0)
- **Category 3**: Medium/High mass (≥2.5) + Positive diff (>0)

### 5-Level Classification
- **Category 0**: Low mass (<2.5) + Non-positive diff (≤0)
- **Category 1**: Low mass (<2.5) + Positive diff (>0)
- **Category 2**: Medium mass (2.5-3.0) + Non-positive diff (≤0)
- **Category 3**: Medium/High mass (≥2.5) + Positive diff (>0), but not meeting Category 4 criteria
- **Category 4**: HIGH ACTIVITY - Either mass > 3.0 OR diff > 0.5 (or both)

## Combined Level-Cell Type Classification

The notebooks also create combined classifications that incorporate both activity level and cell type:

### 4-Level Combined Classification
- **Field**: `combined_4_category_with_cell_type`
- **Unique values**: 80 (4 levels × 20 cell lines)
- **Code range**: `combined_4_category_code` from 0 to 79

### 5-Level Combined Classification
- **Field**: `combined_5_category_with_cell_type`
- **Unique values**: 94
- **Code range**: `combined_5_category_code` from 0 to 93

## Output Data
saved in ./Data/*.csv

# UTR Cell Line Diffusion Model

A diffusion model for generating cell type-specific UTR (Untranslated Region) sequences with different activity levels.

This script implements a conditional diffusion model that generates synthetic UTR sequences based on:
- Cell type specificity
- Activity levels (categorized by center of mass and difference metrics)

## Usage

### Basic Command

```bash
# Training a general model
python vJun22_UTR_CellLine_DifferentLevel_DM.py --file_name /home/yanyichu/1_UTR_Cell_Type/UTR_celltype_github/Dffusion_Model_Generation/Data/vJun26_5UTR_4or5Level_CellType.csv --category_col combined_5_category_with_cell_type_code --prefix vJun26_5UTR_DM --cell_type ''
```

### Cell Type Specific Training
```bash
python vJun22_UTR_CellLine_DifferentLevel_DM.py --file_name /home/yanyichu/1_UTR_Cell_Type/UTR_celltype_github/Dffusion_Model_Generation/Data/vJun26_5UTR_4or5Level_CellType.csv --category_col combined_5_category --prefix vJun26_5UTR_DM --cell_type K562
```

## Parameters

### Required Arguments
- `--file_name`: Path to input CSV file containing UTR sequences and activity levels
- `--category_col`: Column name for category classification
 - Options: `combined_4_category`, `combined_5_category`, `combined_4_category_with_cell_type_code`, `combined_5_category_with_cell_type_code`
- `--prefix`: Prefix for output files

### Optional Arguments
- `--cell_type`: Specific cell type to train on (default: '' for all cell types)
- `--device`: GPU device ID (default: '0')
- `--epochs`: Number of training epochs (default: 1608)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--batch_size`: Batch size (default: 64)
- `--image_size`: UTR sequence length (default: 50)
- `--timesteps`: Diffusion timesteps (default: 100)
- `--beta_scheduler`: Beta scheduler type ['linear', 'cosine', 'quadratic', 'sigmoid'] (default: 'linear')
- `--gen_num`: Number of sequences to generate per class (default: 5000)

## Output Files

### Models
- `./models/{prefix}_best_model.pt`: Best model checkpoint based on validation loss

### Generated Sequences
- `./results/{prefix}_Class{N}_num{count}.fasta`: Generated sequences for each class
 - N: Class number
 - count: Number of unique sequences generated

### Training Metrics
- `./results/{prefix}_training_curves.png`: Training and validation loss curves
 - NLL (Negative Log-Likelihood) loss
 - Diffusion loss
 - Total loss

## Model Architecture
- **Base Model**: U-Net with attention mechanisms
- **Conditioning**: Class-conditional generation using embedding layers
- **Diffusion Process**: Denoising diffusion probabilistic model (DDPM)
- **Time Embedding**: Learned sinusoidal position embeddings
- **Classifier-Free Guidance**: Support for unconditional training (p_uncond=0.1)

## Features
- **Class Balancing**: Automatic class weight calculation for imbalanced datasets
- **Early Stopping**: Patience-based early stopping (default: 50 epochs)
- **EMA**: Exponential Moving Average for stable generation
- **Time Warping**: Optional time step adjustment based on loss values
- **Multi-level Generation**: Supports both 4-level and 5-level activity classifications

  
## Notes
- The model uses one-hot encoding for DNA sequences (A, C, T, G)
- Supports sequences of length 50nt by default (adjustable via --image_size), and may need to change the Upsampling and Downsampling class in the code to fit the image_size.

# Analysis of Generated Sequences

This section analyzes the sequences generated by the diffusion model using PARADE predictions and overlap analysis.

## Script

- `vJun26_5UTR_Generate2Parade_4Level_remove50percent.ipynb`

### Modify Generated Sequence Path (Cell [7])

Update the `filename` variable to point to your generated sequence FASTA files:

```python
filename = f'./results/vJun25_{args.utr_type}_DM_{cell_type}__combined_4_category__Epoch1608_L50_Batch64_TimeSteps100_linearBETA_lr0.0001_Class{level}_num5056.fasta'
```

The filename pattern should match your diffusion model output format:
- `{args.utr_type}`: 5UTR or 3UTR
- `{cell_type}`: Specific cell type name
- `{level}`: Activity level class (0-3 for 4-level classification)
- `num5056`: Number of sequences in the file (will vary)
  
## Configuration Parameters

Configure the analysis by modifying the following arguments:

```python
parser.add_argument('--prefix', type=str, default='vJun26_DM_5UTR_Generate')
parser.add_argument('--utr_type', type=str, default='5UTR')
parser.add_argument('--train_file', type=str, default='../Data_Processing/vJun25_5UTR_Timo_Matt_Merged_data.csv')
parser.add_argument('--modelfile', type=str, default='../Parade_Prediction/vJun25_5UTR_saved_models/epoch=9-step=2590.ckpt')
parser.add_argument('--train_stat', type=str, default='../Parade_Prediction/prediction_results/vJun26_5UTR_training_stats.json')
parser.add_argument('--pred_stat', type=str, default='../Parade_Prediction/prediction_results/vJun26_5UTR_prediction_stats.json')
```

### Parameter Descriptions
- `--prefix`: Output file prefix for results
- `--utr_type`: Type of UTR being analyzed ('5UTR' or '3UTR')
- `--train_file`: Path to training data CSV file
- `--modelfile`: Path to trained PARADE model checkpoint
- `--train_stat`: Path to training statistics JSON file
- `--pred_stat`: Path to prediction statistics JSON file

## Analysis Pipeline

### 1. Load PARADE Model
Loads the pre-trained PARADE model to predict activity levels for generated sequences.

### 2. Sequence Overlap Analysis
The analysis provides three filtering modes for handling sequence overlaps:
- Analyze Only (No Filtering): Analyzes overlap patterns without removing any sequences, Useful for understanding the extent of sequence duplication

- Filter Within Cell Type: Removes sequences that appear in multiple activity levels within the same cell type, Ensures each sequence is unique within its cell type context

- Filter All Overlaps: Removes all sequences that appear in multiple categories, Ensures global uniqueness across all cell types and activity levels

### 3. Predict Center of Mass and Difference Across Cell Lines

Predicts activity metrics for each generated sequence in each cell line.

#### Example Output:

Matching cell types found: ['K562']
=== Processing Level 0 ===
Processing K562 - Level 0
Loaded 5056 sequences from K562 - level_0
Sample sequences:
Sequence 1: GCAGCGCCGAGGAAGCCGGGGCCACCGCCGCCGCGAGGCGGGGTGCGGCG...
Sequence 2: AGCCGAGCCGCCGCCGCGTCCATTGAAGCCAGAGCGTGGCCTGCAGTACA...
Sequence 3: ACTTCCCTGTATAGATTTGCACAGTGGGTCGCTGTAAGGGTTTTTTTCGG...
Getting predictions...
Available cell types in predictions: ['A549', 'Colo320', 'H23', 'H9-Rep1', 'H9-Rep2', 'HCT116', 'K562', 'MCF7', 'MP2', 'PC3', 'c17_Rep1', 'c17_Rep2', 'c1_Rep1', 'c1_Rep2', 'c2_Rep1', 'c2_Rep2', 'c4_Rep1', 'c4_Rep2', 'c6_Rep1', 'c6_Rep2']
Found exact match: K562
Target cell type found: K562
=== Other Cell Diff Calculation Process ===
Target cell type: K562
Other cell types: ['A549', 'Colo320', 'H23', 'H9-Rep1', 'H9-Rep2', 'HCT116', 'MCF7', 'MP2', 'PC3', 'c17_Rep1', 'c17_Rep2', 'c1_Rep1', 'c1_Rep2', 'c2_Rep1', 'c2_Rep2', 'c4_Rep1', 'c4_Rep2', 'c6_Rep1', 'c6_Rep2']
--- Example Sequence Other Cell Diff Calculation ---
Sequence: AAAAGGCGCCCGGCCTGCAGGCGGGCGACC...
Target cell type K562 predicted diff: 0.0196
Other cell types predicted diff:
A549: 0.0096
Colo320: -0.0069
H23: -0.0197
H9-Rep1: 0.0041
H9-Rep2: 0.0038
HCT116: -0.0044
MCF7: 0.0206
MP2: -0.0039
PC3: -0.1046
...
Maximum predicted diff in other cell types: 0.0206
Other cell type with maximum diff: MCF7
--- Other Cell Diff Advantage Calculation ---
Formula: other_cell_diff_advantage = target_cell_diff - max_other_cell_diff
Example: 0.0196 - 0.0206 = -0.0010
Interpretation:

If other_cell_diff_advantage > 0: Sequence performs better in target cell type than any other
If other_cell_diff_advantage < 0: Sequence performs better in another cell type
Higher values indicate stronger specificity for target cell type

=== K562 Advantage Analysis Statistics ===
Training data - diff_max: 0.5714, diff_mean: 0.0104
Prediction data - diff_max: 0.3430, diff_mean: 0.0087
Sequences exceeding training max: 0/5056 (0.0%)
Sequences exceeding prediction max: 0/5056 (0.0%)
Sequences more specific to target cell type: 606/5056 (12.0%)
Total sequences obtained for level 0: 5056
