import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from CSV file and perform initial preprocessing."""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


def preprocess_data(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Preprocess the data for XGBoost training."""
    # Validate target column exists
    if target_column not in df.columns:
        available_columns = ", ".join(df.columns)
        raise ValueError(f"Target column '{target_column}' not found in the dataset. Available columns: {available_columns}")
    
    # Separate features and target
    features = df.drop(columns=[target_column])
    target = df[target_column]

    # Identify numeric and categorical columns
    numeric_cols = features.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = features.select_dtypes(include=['object', 'category']).columns

    # Handle missing values
    numeric_imputer = SimpleImputer(strategy='mean')
    categorical_imputer = SimpleImputer(strategy='most_frequent')

    # Impute numeric columns
    if len(numeric_cols) > 0:
        features[numeric_cols] = numeric_imputer.fit_transform(features[numeric_cols])

    # Impute categorical columns
    if len(categorical_cols) > 0:
        features[categorical_cols] = categorical_imputer.fit_transform(features[categorical_cols])

    # Encode categorical variables
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        features[col] = le.fit_transform(features[col])
        encoders[col] = le

    # Scale numeric features
    scaler = StandardScaler()
    if len(numeric_cols) > 0:
        features[numeric_cols] = scaler.fit_transform(features[numeric_cols])

    return features, target


def split_data(features: pd.DataFrame, target: pd.Series, train_ratio: float = 0.6,
               val_ratio: float = 0.2) -> tuple[tuple[pd.DataFrame, pd.Series], 
                                              tuple[pd.DataFrame, pd.Series], 
                                              tuple[pd.DataFrame, pd.Series]]:
    """Split the data into training, validation, and test sets."""
    # Shuffle the data
    np.random.seed(42)  # For reproducibility
    shuffle_idx = np.random.permutation(len(features))
    features = features.iloc[shuffle_idx].reset_index(drop=True)
    target = target.iloc[shuffle_idx].reset_index(drop=True)

    # Calculate split indices
    n = len(features)
    train_idx = int(n * train_ratio)
    val_idx = int(n * (train_ratio + val_ratio))

    # Split the data
    features_train = features[:train_idx]
    target_train = target[:train_idx]
    
    features_val = features[train_idx:val_idx]
    target_val = target[train_idx:val_idx]
    
    features_test = features[val_idx:]
    target_test = target[val_idx:]

    return (features_train, target_train), (features_val, target_val), (features_test, target_test)


def save_data(x: pd.DataFrame, y: pd.Series, prefix: str, output_dir: str) -> None:
    """Save features and target data to separate files."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save features
        features_path = os.path.join(output_dir, f"{prefix}_features.csv")
        x.to_csv(features_path, index=False)
        print(f"Saved features to {features_path}")
        
        # Save target
        target_path = os.path.join(output_dir, f"{prefix}_target.csv")
        y.to_csv(target_path, index=False)
        print(f"Saved target to {target_path}")
    except Exception as e:
        print(f"Error saving data: {e}")
        sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print("Usage: python preprocess_data.py [input_file] [target_column]")
        sys.exit(1)

    input_file = sys.argv[1]

    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found")
        sys.exit(1)

    # Load data first to show available columns
    print("Loading data...")
    df = load_data(input_file)
    
    # If target column is not provided, show available columns and exit
    if len(sys.argv) != 3:
        print("\nAvailable columns in the dataset:")
        for i, col in enumerate(df.columns, 1):
            print(f"{i}. {col}")
        print("\nPlease run the script again with one of these columns as target column:")
        print(f"python preprocess_data.py {input_file} <target_column_name>")
        sys.exit(1)

    target_column = sys.argv[2]

    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    df = load_data(input_file)

    # Preprocess data
    print("Preprocessing data...")
    x, y = preprocess_data(df, target_column)

    # Split data
    print("Splitting data...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_data(x, y)

    # Save splits
    print("Saving splits...")
    save_data(x_train, y_train, "train", output_dir)
    save_data(x_val, y_val, "val", output_dir)
    save_data(x_test, y_test, "test", output_dir)

    # Print split sizes
    print("\nData split summary:")
    print(f"Total samples: {len(df)}")
    print(f"Training samples: {len(x_train)} ({len(x_train)/len(df)*100:.1f}%)")
    print(f"Validation samples: {len(x_val)} ({len(x_val)/len(df)*100:.1f}%)")
    print(f"Test samples: {len(x_test)} ({len(x_test)/len(df)*100:.1f}%)")

    # Print feature information
    print("\nFeature information:")
    print(f"Number of features: {x_train.shape[1]}")
    print("\nFeature names:")
    print(", ".join(x_train.columns))


if __name__ == "__main__":
    main()
