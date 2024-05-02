# ML + FastAPI

## Data 

https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022/data

## Model

* 1. Get data
* 2. Split data into train, validate and test part
* 3. Build a model
* 4. Model must predict if transaction is fraud or not
* 5. Deploy model using FastAPI. This is a list of endpoint

## Endpoints
* `/predict`: Accepts a transaction data and returns the prediction of whether it is a fraud or not.
* `/models`: Returns list of available models and their properties
* `/history`: Retrieves the history of transactions and results.
* `/docs`: Provides the Swagger UI documentation for the API.
* `/health`: Returns the health status of the API.
* `/status`: Returns the status of the API.
* `/info`: Returns information about the API.
* `/train`: Train new model with specified parameters


