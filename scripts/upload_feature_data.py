import argparse
import logging
import os
import sys
import uuid

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

# Import your models
from fastapi_exp.models.models import Base, FeatureStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataUploader:
    def __init__(self, database_url: str):
        """Initialize the data uploader with database connection."""
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                logger.info("✅ Database connection successful")
                return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
    def create_tables(self):
        """Create tables if they don't exist."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("✅ Tables created/verified")
        except Exception as e:
            logger.error(f"❌ Error creating tables: {e}")
            raise
    
    def load_and_validate_data(self, csv_path: str, max_records: int | None = None) -> pd.DataFrame:
        """Load and validate the CSV data."""
        try:
            # Load data
            df = pd.read_csv(csv_path)
            logger.info(f"📊 Loaded data with shape: {df.shape}")
            
            # Limit records if specified
            if max_records and len(df) > max_records:
                df = df.head(max_records)
                logger.info(f"🔢 Limited to {max_records} records")
            
            # Validate required columns exist
            required_columns = [
                'income', 'name_email_similarity', 'customer_age', 'days_since_request',
                'zip_count_4w', 'velocity_6h', 'velocity_24h', 'velocity_4w',
                'bank_branch_count_8w', 'date_of_birth_distinct_emails_4w',
                'credit_risk_score', 'proposed_credit_limit', 'device_fraud_count',
                'month', 'payment_type', 'employment_status', 'housing_status',
                'source', 'device_os', 'email_is_free', 'phone_home_valid',
                'phone_mobile_valid', 'has_other_cards', 'foreign_request',
                'keep_alive_session',
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"❌ Missing required columns: {missing_columns}")
                logger.info(f"Available columns: {list(df.columns)}")
                raise ValueError(f"Missing columns: {missing_columns}")
            
            logger.info("✅ Data validation passed")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading/validating data: {e}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for database insertion."""
        df = df.copy()
        
        # Handle missing values (-1 indicators)
        missing_value_columns = [
            'prev_address_months_count', 'current_address_months_count',
            'intended_balcon_amount', 'bank_months_count', 
            'session_length_in_minutes', 'device_distinct_emails'
        ]
        
        for col in missing_value_columns:
            if col in df.columns:
                # Convert -1 to None (NULL in database)
                df[col] = df[col].replace(-1, None)
        
        # Ensure boolean columns are properly typed
        boolean_columns = [
            'email_is_free', 'phone_home_valid', 'phone_mobile_valid',
            'has_other_cards', 'foreign_request', 'keep_alive_session',
        ]
        
        for col in boolean_columns:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        
        # Clean string columns
        string_columns = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        logger.info("✅ Data preprocessing completed")
        return df
    
    def upload_batch(self, batch_df: pd.DataFrame, session) -> int:
        """Upload a batch of records to the database."""
        success_count = 0
        
        for _, row in batch_df.iterrows():
            try:
                # Create FeatureStore record
                feature_record = FeatureStore(
                    id=uuid.uuid4(),
                    income=float(row['income']),
                    name_email_similarity=float(row['name_email_similarity']),
                    prev_address_months_count=row.get('prev_address_months_count'),
                    current_address_months_count=row.get('current_address_months_count'),
                    customer_age=int(row['customer_age']),
                    days_since_request=int(row['days_since_request']),
                    intended_balcon_amount=row.get('intended_balcon_amount'),
                    zip_count_4w=int(row['zip_count_4w']),
                    velocity_6h=float(row['velocity_6h']),
                    velocity_24h=float(row['velocity_24h']),
                    velocity_4w=float(row['velocity_4w']),
                    bank_branch_count_8w=int(row['bank_branch_count_8w']),
                    date_of_birth_distinct_emails_4w=int(row['date_of_birth_distinct_emails_4w']),
                    credit_risk_score=float(row['credit_risk_score']),
                    bank_months_count=row.get('bank_months_count'),
                    proposed_credit_limit=float(row['proposed_credit_limit']),
                    session_length_in_minutes=row.get('session_length_in_minutes'),
                    device_distinct_emails=row.get('device_distinct_emails'),
                    device_fraud_count=int(row['device_fraud_count']),
                    month=int(row['month']),
                    payment_type=str(row['payment_type'])[:50],  # Ensure length limit
                    employment_status=str(row['employment_status'])[:50],
                    housing_status=str(row['housing_status'])[:50],
                    source=str(row['source'])[:20],
                    device_os=str(row['device_os'])[:20],
                    email_is_free=bool(row['email_is_free']),
                    phone_home_valid=bool(row['phone_home_valid']),
                    phone_mobile_valid=bool(row['phone_mobile_valid']),
                    has_other_cards=bool(row['has_other_cards']),
                    foreign_request=bool(row['foreign_request']),
                    keep_alive_session=bool(row['keep_alive_session'])
                )
                
                session.add(feature_record)
                success_count += 1
                
            except Exception as e:
                logger.warning(f"⚠️ Error processing record: {e}")
                continue
        
        return success_count
    
    def upload_data(self, df: pd.DataFrame, batch_size: int = 1000):
        """Upload data in batches with progress tracking."""
        total_records = len(df)
        uploaded_count = 0
        
        logger.info(f"🚀 Starting upload of {total_records} records in batches of {batch_size}")
        
        with tqdm(total=total_records, desc="Uploading") as pbar:
            for start_idx in range(0, total_records, batch_size):
                end_idx = min(start_idx + batch_size, total_records)
                batch_df = df.iloc[start_idx:end_idx]
                
                session = self.SessionLocal()
                try:
                    batch_success = self.upload_batch(batch_df, session)
                    session.commit()
                    uploaded_count += batch_success
                    pbar.update(len(batch_df))
                    
                except Exception as e:
                    session.rollback()
                    logger.error(f"❌ Batch upload failed: {e}")
                    
                finally:
                    session.close()
        
        logger.info(f"✅ Upload completed: {uploaded_count}/{total_records} records uploaded")
        return uploaded_count

def get_database_url() -> str:
    """Get database URL from environment variables."""
    db_url = os.getenv("NEON_DEV_URL")
    if not db_url:
        logger.error("❌ NEON_DEV_URL environment variable not set")
        logger.info("Please set NEON_DEV_URL=postgresql://user:pass@host/dbname")
        sys.exit(1)
    return db_url

def main():
    parser = argparse.ArgumentParser(description="Upload fraud detection data to FeatureStore")
    parser.add_argument("csv_file", help="Path to CSV file containing fraud detection data")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for uploads (default: 1000)")
    parser.add_argument("--max-records", type=int, help="Maximum number of records to upload")
    parser.add_argument("--dry-run", action="store_true", help="Validate data without uploading")
    
    args = parser.parse_args()
    
    # Validate file exists
    if not os.path.exists(args.csv_file):
        logger.error(f"❌ File not found: {args.csv_file}")
        sys.exit(1)
    
    # Get database URL
    database_url = get_database_url()
    
    # Initialize uploader
    uploader = DataUploader(database_url)
    
    # Test connection
    if not uploader.test_connection():
        sys.exit(1)
    
    # Create tables
    uploader.create_tables()
    
    # Load and validate data
    df = uploader.load_and_validate_data(args.csv_file, args.max_records)
    
    # Preprocess data
    df = uploader.preprocess_data(df)
    
    if args.dry_run:
        logger.info("🔍 Dry run completed - data validation passed")
        logger.info(f"Ready to upload {len(df)} records")
        return
    
    # Upload data
    uploaded_count = uploader.upload_data(df, args.batch_size)
    
    if uploaded_count > 0:
        logger.info("🎉 Data upload job completed successfully!")
    else:
        logger.error("❌ No records were uploaded")
        sys.exit(1)

if __name__ == "__main__":
    main()