# Perceptron for Breast Cancer Classification

This folder contains a simple implementation of the perceptron algorithm to classify breast cancer as either benign or malignant based on the widely-used Wisconsin breast cancer dataset.

## Overview

The project is structured as a cohesive pipeline that:

1. Downloads the breast cancer dataset.
2. Processes the data to obtain training, validation, and test sets.
3. Trains a perceptron model on the training data.
4. Validates the trained model on the validation set.
5. Tests the trained model on the test set.
6. Uses the trained model to predict the class of new data samples.

## Dependencies

- Python 3.x
- `urllib`
- `csv`
- `random`

## Usage
``` bash
cd Perceptron
python perceptron_example.py