import os
import sys

import numpy as np


def load_data(file_path: str) -> np.ndarray:
    try:
        return np.genfromtxt(file_path, delimiter=',', skip_header=1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


def split_data(data: np.ndarray, train_ratio: float = 0.6,
               val_ratio: float = 0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Shuffle the data
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(data)

    # Calculate split indices
    n = len(data)
    train_idx = int(n * train_ratio)
    val_idx = int(n * (train_ratio + val_ratio))

    # Split the data
    train_data = data[:train_idx]
    val_data = data[train_idx:val_idx]
    test_data = data[val_idx:]

    return train_data, val_data, test_data


def save_data(data: np.ndarray, file_path: str) -> None:
    try:
        np.savetxt(file_path, data, delimiter=',')
        print(f"Saved data to {file_path}")
    except Exception as e:
        print(f"Error saving data: {e}")
        sys.exit(1)


def main():
    if len(sys.argv) != 2:
        print("Usage: python preprocess_data.py [input_file]")
        sys.exit(1)

    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found")
        sys.exit(1)

    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    data = load_data(input_file)

    # Split data
    print("Splitting data...")
    train_data, val_data, test_data = split_data(data)

    # Save splits
    print("Saving splits...")
    save_data(train_data, os.path.join(output_dir, "train.csv"))
    save_data(val_data, os.path.join(output_dir, "validation.csv"))
    save_data(test_data, os.path.join(output_dir, "test.csv"))

    # Print split sizes
    print("\nData split summary:")
    print(f"Total samples: {len(data)}")
    print(f"Training samples: {len(train_data)} ({len(train_data)/len(data)*100:.1f}%)")
    print(f"Validation samples: {len(val_data)} ({len(val_data)/len(data)*100:.1f}%)")
    print(f"Test samples: {len(test_data)} ({len(test_data)/len(data)*100:.1f}%)")


if __name__ == "__main__":
    main()
