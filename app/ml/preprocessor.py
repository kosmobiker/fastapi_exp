import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    A class to preprocess data for machine learning tasks.
    It handles loading, cleaning, and transforming data.
    """

    def __init__(self, data_path: str, save_path: str):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"The specified data path {self.data_path} does not exist."
            )
        self.save_path = save_path

    def load_data(self) -> pd.DataFrame:
        """
        Load data from the specified path.

        Returns:
            pd.DataFrame: Loaded data as a DataFrame.
        """
        try:
            data = pd.read_csv(self.data_path)
            return data
        except Exception as e:
            raise IOError(f"Error loading data: {e}")

    def replace_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace null values in the DataFrame with the mean of each column.
        """
        cols_to_replace = [col for col in df.columns if col != "credit_risk_score"]
        df[cols_to_replace] = df[cols_to_replace].replace(-1, np.nan)

        return df

    def split_data(
        self, df: pd.DataFrame, target_col: str = "fraud_bool"
    ) -> dict[str, pd.DataFrame]:
        """
        Split the DataFrame into features and target variable.
        """
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
        )

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        }

    def preprocess(self, data: dict[str, pd.DataFrame]) -> None:
        """
        Preprocess the data by loading, replacing nulls, and splitting.
        """
        X = pd.concat([data.get("X_train"), data.get("X_val"), data.get("X_test")])
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_cols),
                ("cat", categorical_pipeline, categorical_cols),
            ],
            remainder="passthrough",
        )
        X_train = data["X_train"]
        preprocessor.fit(X_train)
        try:
            dump(preprocessor, f"{self.save_path}/preprocessor.pkl")
        except Exception as e:
            raise IOError(f"Error saving preprocessor: {e}")

    def transform(self, data: dict[str, pd.DataFrame]) -> None:
        """
        Transform the data using the fitted preprocessor.
        """
        preprocessor = load(f"{self.save_path}/preprocessor.pkl")
        X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
        y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]
        X_train = preprocessor.transform(data["X_train"])
        X_val = preprocessor.transform(data["X_val"])
        X_test = preprocessor.transform(data["X_test"])

        np.save(f"{self.save_path}/X_train.npy", X_train)
        np.save(f"{self.save_path}/X_val.npy", X_val)
        np.save(f"{self.save_path}/X_test.npy", X_test)
        np.save(f"{self.save_path}/y_train.npy", y_train)
        np.save(f"{self.save_path}/y_val.npy", y_val)
        np.save(f"{self.save_path}/y_test.npy", y_test)


if __name__ == "__main__":
    preprocessor = Preprocessor(
        data_path="data/raw/Base.csv", save_path="data/model_artifacts"
    )
    data = preprocessor.load_data()
    logger.info("Data loaded successfully 😊")
    data = preprocessor.replace_nulls(data)
    logger.info("Null values replaced successfully 🙂")
    split_data = preprocessor.split_data(data)
    logger.info("Data split into training, validation, and test sets 😎")
    preprocessor.preprocess(split_data)
    logger.info("Preprocessing pipeline fitted and saved 👌")
    preprocessor.transform(split_data)
    logger.info("Data transformed and saved successfully 🤖")
