# PARADE Model Training and Evaluation

This section describes the training of PARADE models using preprocessed MPRA data and evaluation of model performance.

## Model Training

### Training Scripts by Quality Threshold

#### Count < 100 Threshold
- **5' UTR**: `vJun26_regression_utr5_Mass_Diff_mincount100.ipynb`
- **3' UTR**: `vJun26_regression_utr3_Mass_Diff_mincount100.ipynb`

#### 50% Threshold
- **5' UTR**: `vJun25_regression_utr5_Mass_Diff_remove50percent.ipynb`
- **3' UTR**: `vJun25_regression_utr3_Mass_Diff_remove50percent.ipynb`

### Configuration Parameters

Before running the training scripts, update the following parameters:

1. **Input Data Path**:
  ```python
  PATH_FROM = "path/to/your/training/file.csv"
```
  The training file must contain these columns:
- `sequence`: UTR sequences
- `center_of_mass`: Activity measurements
- `diff`: Activity differences across cell lines
- `cell_type`: Cell line identifiers
- `fold`: Train/validation split indicator

### 1. Model Checkpoint Directory:

```python
checkpoint_callback = pl.callbacks.ModelCheckpoint(
   dirpath="path/to/your/output/folder/"
)
```
### 2. Best Model Loading:
```python
model = RNARegressor.load_from_checkpoint(
    'path/to/your/output/best_model'
) 
```
### 3. Performance Metrics Output:
```python
metrics_readable.to_csv('path/to/your/output/performance_metrics')
```

## Model Performance Metrics

The training scripts evaluate models using:
- **R²** (coefficient of determination)
- **Pearson correlation**
- **Spearman correlation**

Metrics are calculated for both training and validation sets.

## Model Prediction

### Prediction Scripts by Quality Threshold

#### Count < 100 Threshold
- `vJun26_Parade_TrainingData_mincount100_Prediction.ipynb`

#### 50% Threshold
- `vJun26_Parade_TrainingData_Prediction_remove50percent.ipynb`

### Utilities
- **`Parade_Predict.py`**: Core module containing:
 - Model loading functions
 - Dataset loader
 - Prediction pipeline

### Usage Example
The prediction .ipynb scripts support argparse.ArgumentParser() to self-define the train_file and parade modelfile.
The default output folder is ./prediction_results, you could change by the parameter **output_prefix**

## Output Files

All prediction results are saved in the `prediction_results/` directory:

### 1. Complete Prediction Results
- **5' UTR**: `vJun26_5UTR_mincount100_prediction_results.csv`
- **3' UTR**: `vJun26_3UTR_mincount100_prediction_results.csv`

These files contain the original data with added **predictions** for:
- Center of mass values
- Difference values across cell lines

### 2. Statistical Summaries
- `{output_prefix}training_stats.json`
- `{output_prefix}prediction_stats.json`

JSON files containing statistics for each cell line:
- `diff_mean`: Mean difference value
- `diff_max`: Maximum difference value
- `mass_mean`: Mean center of mass
- `mass_max`: Maximum center of mass

These files can be used for evaluating generation method. 

### 3. Performance Metrics
- `vJun26_3UTR_mincount100_Performancecorrelation.csv`
- `vJun26_5UTR_mincount100_Performancecorrelation.csv`

CSV files with per-cell-line metrics:
- R² score
- Pearson correlation
- Spearman correlation
- Sample size

### 4. Visualizations
- `*.pdf`: Various plots showing model performance and predictions

## Quick Start

1. Prepare your preprocessed MPRA data with required columns
2. Update configuration parameters in the training notebook
3. Run the training script for your UTR type and threshold
4. Use the trained model for predictions with the prediction notebook
5. Check the `prediction_results/` folder for outputs