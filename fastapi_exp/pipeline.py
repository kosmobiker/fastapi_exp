#!/usr/bin/env python3
"""
Complete ML Pipeline for Fraud Detection
This script handles the entire pipeline from raw data to trained model:
1. Load and validate raw data (Base.csv)
2. Preprocess data (imputation, encoding, scaling)
3. Split into train/val/test sets
4. Train XGBoost model
5. Evaluate model performance
6. Save model and preprocessors for API deployment

Usage:
    python complete_pipeline.py [--input-file Base.csv] [--target fraud_bool]
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from typing import Tuple, Dict, Any
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb
from xgboost import DMatrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FraudDetectionPipeline:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.preprocessors = {}
        self.feature_info = {}
        self.model = None
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and validate the raw dataset."""
        logger.info(f"📁 Loading data from {file_path}")
        
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            df = pd.read_csv(file_path)
            logger.info(f"✅ Loaded data with shape: {df.shape}")
            
            # Display basic info
            logger.info(f"📊 Dataset info:")
            logger.info(f"   - Total samples: {len(df):,}")
            logger.info(f"   - Total features: {df.shape[1]}")
            logger.info(f"   - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Show missing values
            missing_info = df.isnull().sum()
            if missing_info.sum() > 0:
                logger.info(f"⚠️  Missing values found:")
                for col, missing_count in missing_info[missing_info > 0].items():
                    pct = (missing_count / len(df)) * 100
                    logger.info(f"   - {col}: {missing_count} ({pct:.1f}%)")
            else:
                logger.info("✅ No missing values found")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            raise
    
    def validate_data(self, df: pd.DataFrame, target_column: str) -> None:
        """Validate dataset and target column."""
        logger.info("🔍 Validating dataset...")
        
        # Check if target column exists
        if target_column not in df.columns:
            available_cols = ", ".join(df.columns[:10])  # Show first 10 columns
            raise ValueError(f"Target column '{target_column}' not found. Available: {available_cols}...")
        
        # Check target distribution
        target_dist = df[target_column].value_counts()
        logger.info(f"📈 Target distribution ({target_column}):")
        for value, count in target_dist.items():
            pct = (count / len(df)) * 100
            logger.info(f"   - {value}: {count:,} ({pct:.1f}%)")
        
        # Check for class imbalance
        if len(target_dist) == 2:
            minority_pct = min(target_dist) / len(df) * 100
            if minority_pct < 10:
                logger.warning(f"⚠️  Severe class imbalance detected: {minority_pct:.1f}% minority class")
        
        logger.info("✅ Data validation passed")
    
    def preprocess_data(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Complete preprocessing pipeline."""
        logger.info("🔧 Starting data preprocessing...")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Identify column types
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        boolean_cols = X.select_dtypes(include=['bool']).columns.tolist()
        
        logger.info(f"📋 Feature types identified:")
        logger.info(f"   - Numeric: {len(numeric_cols)} columns")
        logger.info(f"   - Categorical: {len(categorical_cols)} columns") 
        logger.info(f"   - Boolean: {len(boolean_cols)} columns")
        
        # Store column information
        self.feature_info = {
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'boolean_columns': boolean_cols,
            'feature_names': X.columns.tolist(),
            'target_column': target_column
        }
        
        # Handle missing values
        logger.info("🔧 Handling missing values...")
        
        # Numeric imputation
        if numeric_cols:
            numeric_imputer = SimpleImputer(strategy='median')  # More robust than mean
            X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])
            self.preprocessors['numeric_imputer'] = numeric_imputer
            logger.info(f"   ✅ Imputed {len(numeric_cols)} numeric columns with median")
        
        # Categorical imputation
        if categorical_cols:
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            X[categorical_cols] = categorical_imputer.fit_transform(X[categorical_cols])
            self.preprocessors['categorical_imputer'] = categorical_imputer
            logger.info(f"   ✅ Imputed {len(categorical_cols)} categorical columns with mode")
        
        # Encode categorical variables
        logger.info("🔧 Encoding categorical variables...")
        label_encoders = {}
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
            
            n_categories = len(le.classes_)
            logger.info(f"   ✅ Encoded '{col}': {n_categories} categories")
        
        self.preprocessors['label_encoders'] = label_encoders
        
        # Scale numeric features
        if numeric_cols:
            logger.info("🔧 Scaling numeric features...")
            scaler = StandardScaler()
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
            self.preprocessors['scaler'] = scaler
            logger.info(f"   ✅ Scaled {len(numeric_cols)} numeric columns")
        
        # Convert boolean columns to int
        if boolean_cols:
            X[boolean_cols] = X[boolean_cols].astype(int)
            logger.info(f"   ✅ Converted {len(boolean_cols)} boolean columns to int")
        
        logger.info("✅ Preprocessing completed")
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   train_size: float = 0.6, val_size: float = 0.2, test_size: float = 0.2) -> Dict[str, Tuple]:
        """Split data into train/validation/test sets."""
        logger.info("✂️  Splitting data...")
        
        # Validate split ratios
        if abs(train_size + val_size + test_size - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=y
        )
        
        # Second split: separate train and validation
        val_ratio = val_size / (train_size + val_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=self.random_state,
            stratify=y_temp
        )
        
        splits = {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
        
        # Log split information
        total_samples = len(X)
        logger.info(f"📊 Data split completed:")
        for split_name, (X_split, y_split) in splits.items():
            count = len(X_split)
            pct = (count / total_samples) * 100
            fraud_pct = (y_split.sum() / len(y_split)) * 100
            logger.info(f"   - {split_name.capitalize()}: {count:,} samples ({pct:.1f}%) - {fraud_pct:.1f}% fraud")
        
        return splits
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                    X_val: pd.DataFrame, y_val: pd.Series) -> xgb.Booster:
        """Train XGBoost model with hyperparameter optimization."""
        logger.info("🚀 Training XGBoost model...")
        
        # Convert to DMatrix
        dtrain = DMatrix(X_train, label=y_train)
        dval = DMatrix(X_val, label=y_val)
        
        # XGBoost parameters (optimized for fraud detection)
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'eta': 0.1,                # Learning rate
            'max_depth': 6,            # Tree depth
            'min_child_weight': 1,     # Regularization
            'subsample': 0.8,          # Row sampling
            'colsample_bytree': 0.8,   # Column sampling
            'gamma': 0,                # Min split loss
            'alpha': 0,                # L1 regularization
            'lambda': 1,               # L2 regularization
            'scale_pos_weight': 1,     # Handle class imbalance if needed
            'random_state': self.random_state,
            'n_jobs': -1,              # Use all CPU cores
            'verbosity': 1
        }
        
        # Handle class imbalance automatically
        fraud_ratio = y_train.sum() / len(y_train)
        if fraud_ratio < 0.1:  # If less than 10% fraud cases
            scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
            params['scale_pos_weight'] = scale_pos_weight
            logger.info(f"⚖️  Applied class balancing: scale_pos_weight = {scale_pos_weight:.2f}")
        
        # Training with early stopping
        evallist = [(dtrain, 'train'), (dval, 'eval')]
        
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=1000,
            evals=evallist,
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        self.model = model
        logger.info("✅ Model training completed")
        
        return model
    
    def evaluate_model(self, splits: Dict[str, Tuple]) -> Dict[str, Dict]:
        """Comprehensive model evaluation."""
        logger.info("📊 Evaluating model performance...")
        
        evaluation_results = {}
        
        for split_name, (X_split, y_split) in splits.items():
            logger.info(f"   📈 Evaluating on {split_name} set...")
            
            # Make predictions
            dmatrix = DMatrix(X_split)
            y_prob = self.model.predict(dmatrix)
            y_pred = (y_prob > 0.5).astype(int)
            
            # Calculate metrics
            auc_score = roc_auc_score(y_split, y_prob)
            
            # Classification report
            report = classification_report(y_split, y_pred, output_dict=True)
            
            # Confusion matrix
            cm = confusion_matrix(y_split, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            # Custom metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results = {
                'auc': auc_score,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'confusion_matrix': {
                    'true_negative': int(tn),
                    'false_positive': int(fp),
                    'false_negative': int(fn),
                    'true_positive': int(tp)
                },
                'classification_report': report
            }
            
            evaluation_results[split_name] = results
            
            # Log key metrics
            logger.info(f"      🎯 AUC: {auc_score:.4f}")
            logger.info(f"      🎯 Precision: {precision:.4f}")
            logger.info(f"      🎯 Recall: {recall:.4f}")
            logger.info(f"      🎯 F1-Score: {f1_score:.4f}")
        
        return evaluation_results
    
    def save_artifacts(self, output_dir: str, evaluation_results: Dict) -> None:
        """Save model, preprocessors, and evaluation results."""
        logger.info(f"💾 Saving artifacts to {output_dir}...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model
        model_path = output_path / "model.json"
        self.model.save_model(str(model_path))
        logger.info(f"   ✅ Model saved: {model_path}")
        
        # Save preprocessors
        preprocessors_path = output_path / "preprocessors.pkl"
        joblib.dump(self.preprocessors, preprocessors_path)
        logger.info(f"   ✅ Preprocessors saved: {preprocessors_path}")
        
        # Save feature info
        feature_info_path = output_path / "feature_info.pkl"
        joblib.dump(self.feature_info, feature_info_path)
        logger.info(f"   ✅ Feature info saved: {feature_info_path}")
        
        # Save evaluation results
        evaluation_path = output_path / "evaluation_results.pkl"
        joblib.dump(evaluation_results, evaluation_path)
        logger.info(f"   ✅ Evaluation results saved: {evaluation_path}")
        
        # Save a summary report
        self._save_summary_report(output_path, evaluation_results)
        
        logger.info("✅ All artifacts saved successfully")
    
    def _save_summary_report(self, output_path: Path, evaluation_results: Dict) -> None:
        """Save a human-readable summary report."""
        report_path = output_path / "model_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("🎯 FRAUD DETECTION MODEL SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Dataset info
            f.write("📊 DATASET INFORMATION\n")
            f.write("-" * 25 + "\n")
            f.write(f"Features: {len(self.feature_info['feature_names'])}\n")
            f.write(f"Numeric features: {len(self.feature_info['numeric_columns'])}\n")
            f.write(f"Categorical features: {len(self.feature_info['categorical_columns'])}\n")
            f.write(f"Boolean features: {len(self.feature_info['boolean_columns'])}\n\n")
            
            # Model performance
            f.write("📈 MODEL PERFORMANCE\n")
            f.write("-" * 25 + "\n")
            
            for split_name, results in evaluation_results.items():
                f.write(f"\n{split_name.upper()} SET:\n")
                f.write(f"  AUC: {results['auc']:.4f}\n")
                f.write(f"  Precision: {results['precision']:.4f}\n")
                f.write(f"  Recall: {results['recall']:.4f}\n")
                f.write(f"  F1-Score: {results['f1_score']:.4f}\n")
                
                cm = results['confusion_matrix']
                f.write(f"  Confusion Matrix:\n")
                f.write(f"    TN: {cm['true_negative']:,}, FP: {cm['false_positive']:,}\n")
                f.write(f"    FN: {cm['false_negative']:,}, TP: {cm['true_positive']:,}\n")
        
        logger.info(f"   ✅ Summary report saved: {report_path}")

    def prepare_features_for_db(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert processed feature set back to raw-like schema for DB storage."""
        # Reverse label encoding for categorical columns (optional)
        for col, le in self.preprocessors.get("label_encoders", {}).items():
            df[col] = le.inverse_transform(df[col])
        return df
    
    def run_pipeline(self, input_file: str, target_column: str, output_dir: str) -> None:
        """Run the complete ML pipeline."""
        logger.info("🚀 Starting complete ML pipeline...")
        
        try:
            # 1. Load data
            df = self.load_data(input_file)
            
            # 2. Validate data
            self.validate_data(df, target_column)
            
            # 3. Preprocess data
            X, y = self.preprocess_data(df, target_column)
            
            # 4. Split data
            splits = self.split_data(X, y)
            
            # 5. Train model
            X_train, y_train = splits['train']
            X_val, y_val = splits['val']
            self.train_model(X_train, y_train, X_val, y_val)
            
            # 6. Evaluate model
            evaluation_results = self.evaluate_model(splits)
            
            # 7. Save artifacts
            self.save_artifacts(output_dir, evaluation_results)

            # 8 Prepare features for DB storage
            prepared_features = self.prepare_features_for_db(X)
            prepared_features_path = Path(output_dir) / "prepared_features.csv"
            prepared_features.to_csv(prepared_features_path, index=False)

            logger.info("🎉 Pipeline completed successfully!")
            logger.info(f"📁 All artifacts saved in: {output_dir}")
            
            # Show final results
            val_auc = evaluation_results['val']['auc']
            test_auc = evaluation_results['test']['auc']
            logger.info(f"🏆 Final Results:")
            logger.info(f"   - Validation AUC: {val_auc:.4f}")
            logger.info(f"   - Test AUC: {test_auc:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Complete Fraud Detection ML Pipeline")
    parser.add_argument(
        "--input-file", 
        default="Base.csv", 
        help="Input CSV file (default: Base.csv)"
    )
    parser.add_argument(
        "--target", 
        default="fraud_bool", 
        help="Target column name (default: fraud_bool)"
    )
    parser.add_argument(
        "--output-dir", 
        default="data/trained_models", 
        help="Output directory for artifacts (default: data/trained_models)"
    )
    parser.add_argument(
        "--random-state", 
        type=int, 
        default=42, 
        help="Random state for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        logger.error(f"❌ Input file not found: {args.input_file}")
        sys.exit(1)
    
    # Run pipeline
    pipeline = FraudDetectionPipeline(random_state=args.random_state)
    pipeline.run_pipeline(args.input_file, args.target, args.output_dir)

if __name__ == "__main__":
    main()