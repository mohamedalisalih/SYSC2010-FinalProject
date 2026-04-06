# SYSC2010 Signal Analyzer

A Python GUI application for loading, preprocessing, filtering, analyzing, and visualizing sensor-based time-series data from CSV files.

## Features

- Load CSV files
- Support for ECG, Respiration, Temperature.
- Preprocess signals
- Apply FIR and IIR filters
- View raw and processed signals
- View FFT plots
- Compute statistics:
  - mean
  - standard deviation
  - RMS
  - peak-to-peak
- Extract features:
  - ECG heart rate
  - Respiration rate
  - Temperature trend slope

## Files

- `main.py` — runs the app
- `gui_app.py` — GUI and plotting
- `io_loader.py` — CSV loading
- `preprocess.py` — preprocessing functions
- `processing.py` — filter design and application
- `analysis.py` — FFT, statistics, and feature extraction

## Requirements

- Python 3
- NumPy
- Matplotlib
- Pandas
- SciPy
- Tkinter

