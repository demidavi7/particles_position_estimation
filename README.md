# Particles position estimation in Resistive Silicon Detectors

This repository contains the code, datasets, and report for a project focused on estimating the 2-D position of subatomic particles passing through Resistive Silicon Detectors (RSDs). The project leverages machine learning techniques to solve a multi-output regression task, using Random Forests to predict particle positions from noisy detector readings.

## Project Structure

├── `datasets.txt`  (Link to the datasets used in the project)  
├── `position_estimation.py`  (Python implementation of position estimation using Random Forests)    
├── `report.pdf`  (Detailed project report)  
├── `requirements.txt`  (Python dependencies)  

## Overview

The objective of this project is to:
- Accurately estimate the x and y coordinates of subatomic particles based on sensor readings from RSDs.
- Leverage the noise resistance and multi-output regression capabilities of Random Forests to improve estimation accuracy.
- Analyze the performance of the model and optimize hyperparameters for better results.

## Dataset Details

The dataset includes:
- A **development set** with 385,500 labeled events.
- An **evaluation set** with 128,500 events.
- Each event contains readings from 18 signals across 5 features: `pmax`, `negpmax`, `area`, `tmax`, and `rms`.

Key preprocessing steps include:
- Identifying and discarding noisy readings based on feature correlation.
- Creating new features, such as `rangemax` (difference between `pmax` and `negpmax`).

## Requirements

To run the code, ensure you have Python installed and the necessary dependencies specified in `requirements.txt`. Install them using:
```bash
pip install -r requirements.txt
```

## Usage
Run the `position_estimation.py` script to train the Random Forest model and evaluate its performance.
Ensure the datasets described in datasets.txt are available in the appropriate directory before executing the script.

## Report
For an in-depth explanation of the methodology, dataset details, preprocessing, hyperparameter tuning, and results, refer to the `report.pdf`.

## Highlights
The model achieves an average estimation error of ~1%, significantly outperforming baseline solutions.
Random Forests were selected for their robustness to noise and support for multi-output regression.
Preprocessing and feature engineering (e.g., rangemax) played a critical role in enhancing model performance.

## Authors
- Davide Elio Stefano Demicheli
- Davide Fassio [@Davidefassio](https://github.com/Davidefassio)
