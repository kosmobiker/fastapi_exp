# ML + FastAPI

## Summary
This repository contains code for training and deploying machine learning models with FastAPI. It includes data preprocessing, model building, and API endpoints for prediction and model management. Mian goal is to train test building and learn basics of FastAPI.


## Data 

The dataset used in this project is the [Bank Account Fraud Dataset](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) from Kaggle. This dataset was used for the NeurIPS 2022 competition.

The dataset contains transaction data that can be used to detect fraudulent bank transactions. Each transaction is represented by a number of features, such as the amount of the transaction, the type of transaction, and information about the account involved in the transaction.

The goal of the project is to use this dataset to train a model that can predict whether a given transaction is fraudulent or not. Code for training was taken from this [notebook](https://www.kaggle.com/code/bolouki/bank-account-fraud-detection-eda-and-model).

## Model

* 1. Get data
* 2. Split data into train, validate and test part
* 3. Build a model (LogReg and LightGBM)
* 4. Model must predict if transaction is fraud or not
* 5. Save preprocessor and model to the disk
* 6. Add a record to the model registry with models' metadata

## Endpoints
* [x] `/models`: Returns list of available models and their properties
* [x] `/predict`: Accepts a transaction data and returns the prediction of whether it is a fraud or not.
* [x] `/history`: Retrieves the history of transactions and results for given period.
* [ ] `/train`: Train new model with specified parameters


