# ML + FastAPI


## Data 

Data was taken from this [page](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022/data). Code for training was taken from this [notebook](https://www.kaggle.com/code/bolouki/bank-account-fraud-detection-eda-and-model).

## Model

* 1. Get data
* 2. Split data into train, validate and test part
* 3. Build a model (LogReg and LightGBM)
* 4. Model must predict if transaction is fraud or not
* 5. Save preprocessor and model to the disk
* 6. Add a record to the model registry with models' metadata

## Endpoints
* `/models`: Returns list of available models and their properties
* `/predict`: Accepts a transaction data and returns the prediction of whether it is a fraud or not.
* `/history`: Retrieves the history of transactions and results for given period.
* `/status`: Returns the status of the API.
* `/info`: Returns information about the API.
* `/train`: Train new model with specified parameters


