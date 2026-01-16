# Harmonic Pattern Classification – Data Mining Project

Our machine learning project for classifying harmonic candlestick patterns in financial time series data using multiple deep learning and traditional ML approaches.

## Project Overview

This project implements and compares several machine learning models to classify 12 harmonic patterns in financial market data:

**Bullish Patterns:** BullBat, BullButterfly, BullCrab, BullCypher, BullGartley, BullShark  
**Bearish Patterns:** BearBat, BearButterfly, BearCrab, BearCypher, BearGartley, BearShark

### Key Features
- **Multi-resolution data:** 1-minute, 1-hour, 4-hour, and 1-day candlestick resolutions
- **Multiple sigma values:** Varying noise levels (0.001 to 0.1) for robustness testing
- **Diverse models:** LSTM, CNN+BiLSTM, SVM, and binary classification approaches
- **Ablation studies:** Component analysis for CNN+LSTM architecture
- **Preprocessing pipeline:** Data cleaning, normalization, and sequence generation

## Project Structure

```
data_mining/
├── data/
│   └── multi_resolution/          # 15 CSV files with different resolutions & sigma values
├── notebooks/
│   ├── lstm.ipynb                 # LSTM baseline model
│   ├── cnn+bilstm.ipynb           # CNN + Bidirectional LSTM
│   ├── svm.ipynb                  # Linear SVM classifier
│   ├── binary.ipynb               # Binary pattern classification
│   ├── deterministic.ipynb        # Deterministic pattern rules
│   └── ablation/                  # Component ablation studies
│       ├── wo_module7_cnn+lstm_experiment.ipynb
│       ├── wo_module10_cnn+lstm_experiment.ipynb
│       └── wo_module15_cnn+lstm_experiment.ipynb
└── src/
    ├── config.py                  # Global configuration (features, classes, paths)
    ├── data_io.py                 # File listing and I/O utilities
    ├── preprocess.py              # CSV loading, cleaning, feature extraction
    ├── sequences.py               # Time series to sequence conversion
    ├── scaling.py                 # StandardScaler normalization
    ├── splits.py                  # Train/eval split utilities
    ├── datasets.py                # Data loader functions
    ├── evaluation.py              # Metrics, confusion matrices, visualization
    ├── train_loops.py             # Training procedures
    ├── weights.py                 # Class weight computation
    └── models/
        ├── lstm_baseline.py       # Simple LSTM model
        ├── cnn_liu_si.py          # CNN feature extractor
        ├── build_liu_si_cnn_bilstm.py  # CNN+BiLSTM architecture
        └── svm.py                 # SVM model wrapper
```


## Models

### 1. LSTM Baseline ([notebooks/lstm.ipynb](notebooks/lstm.ipynb))
Simple LSTM model for sequence classification. Good baseline for temporal pattern recognition.

### 2. CNN + Bidirectional LSTM ([notebooks/cnn+bilstm.ipynb](notebooks/cnn+bilstm.ipynb))
Combines convolutional feature extraction with bidirectional LSTM. Improved performance on directional patterns.

### 3. Linear SVM ([notebooks/svm.ipynb](notebooks/svm.ipynb))
Traditional machine learning approach. Flattens sequences and uses LinearSVC for multiclass classification.

### 4. Binary Classification ([notebooks/binary.ipynb](notebooks/binary.ipynb))
Simplified task: discriminate between bullish vs. bearish patterns.

### 5. Deterministic Rules ([notebooks/deterministic.ipynb](notebooks/deterministic.ipynb))
Rule-based classification using harmonic pattern geometric constraints.


## Ablation Studies

The `notebooks/ablation/` directory contains experiments removing specific CNN+LSTM components to measure their individual contribution:

- **wo_module7:** CNN+LSTM without module 7
- **wo_module10:** CNN+LSTM without module 10
- **wo_module15:** CNN+LSTM without module 15

## Results & Evaluation

Models are evaluated on held-out test sets using:
- **Classification Report:** Precision, Recall, F1-score per class
- **Confusion Matrix:** Visualization of misclassifications
- **Train/Eval Overlap Check:** Ensures no data leakage


## Contact

Bilgesu & Barış
