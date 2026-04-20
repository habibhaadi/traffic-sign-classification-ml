# Traffic Sign Prediction with Machine Learning

This project focuses on classifying traffic sign images into 43 categories using engineered image features rather than raw pixels.

Instead of going down the usual deep learning from image route, this task was approached using structured features already extracted from the images. The goal was to see how far strong classical machine learning models could be pushed on a multi-class image classification task.

## What’s in the project

The notebook walks through a full machine learning workflow:

- loading and combining extracted image features
- preprocessing with feature scaling
- stratified train/validation split
- training multiple classifiers
- tuning the strongest model
- evaluating performance
- generating a Kaggle submission file

## Models tested

I trained and compared:

- k-Nearest Neighbours
- Support Vector Machine
- Random Forest
- Multi-Layer Perceptron

The final model selected was a tuned Random Forest, which gave the best balance between validation performance and generalisation on unseen test data.

## Results

Validation accuracy:

- kNN: 60.3%
- SVM: 74.2%
- Random Forest (untuned): 78.9%
- Random Forest (tuned): 79.9%
- MLP: 80.7%

Kaggle test accuracy:

- MLP: 33.6%
- Random Forest (tuned): 55.7%

The MLP looked strongest on validation, but it clearly did not generalise well. The tuned Random Forest was more stable and ended up being the best final model.

## Features used

Each image was represented using 123 engineered features:

- 97 colour histogram features
- 21 HOG features reduced with PCA
- 5 additional handcrafted image features

These were scaled before training to keep the feature ranges consistent across models.

## Key takeaway

One of the biggest takeaways from this project was that the model with the highest validation score is not always the one that performs best on unseen data. The Random Forest ended up being the better choice because it generalised more reliably.

## Files

- `Assignment2.ipynb` — full notebook with training, tuning, evaluation, and submission code
- `Report in pdf format - final.pdf` — written report discussing methodology, results, and model behaviour

## Tech used

- Python
- pandas
- numpy
- scikit-learn
- matplotlib

## Note

The dataset itself is not included here unless distribution is permitted. This repository focuses on the modelling workflow and results.
