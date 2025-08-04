# MASTER TASK LIST: ML Model + FastAPI + Feature Store + CI + Docker

### 🛠️ PHASE 0: Setup `uv` environment

### 🧠 PHASE 1: Data & Machine Learning
1. Choose & Download Dataset
- ✅ What: Bank Account Fraud (BAF) Tabular Dataset: https://github.com/feedzai/bank-account-fraud/tree/main
- 📁 Save to: data/raw/heart.csv

2. Exploratory Data Analysis (EDA)
- ✅ What: Understand data, nulls, types, distributions
- 🔧 How: Jupyter Notebook
    - Check for nulls
    - Visualize distributions
    - Label balance

3. Preprocess Data
- ✅ What: Build a sklearn.pipeline.Pipeline to:
    - Impute missing values
    - Encode categoricals (e.g., OneHot or Ordinal)
    - Normalize/scale numerics
- 💾 Save to: data/model_artifacts/preprocessor.pkl

4. Split Data
- ✅ What: Use train_test_split() into:
    - Train (70%)
    - Validation (15%)
    - Test (15%)

5. Train ML Model
- ✅ What: Train a classification model (e.g., XGBoostClassifier)
    - 💾 Save: model.pkl in data/model_artifacts/

6. Evaluate Model
- ✅ What: Evaluate on test set with:
    - Accuracy, Precision, Recall, F1, ROC-AUC
    - Confusion Matrix
    - Feature Importance
- build a script to create preprocesor and model and save it to `data/model_artifacts/`

### 📦 PHASE 2: Backend Project Setup
7. Set Up Project Structure
- ✅ What: Use the structure I gave you earlier
    - 📁 Create folders: app/, tests/, data/, etc.

8. Create Pydantic Schemas
- 📁 app/schemas/predict.py

9. Write Model Inference Logic
- 📁 app/ml/model.py

10. Implement FastAPI Endpoints
- `/healthcheck`: returns "ok"
- `/predict/sync/{user_id}`: sync inference
- `/predict/async/{user_id}`: async inference


### 🗃️ PHASE 3: Feature Store (PostgreSQL)

11. Design Feature Store DB Tables
- ✅ What: Create table for features
- upload data to Neon database
- Configure SQLAlchemy + Alembic
- Initialize Alembic: `alembic init alembic`
- Run migration
- Insert Transformed Features
    - Use SQLAlchemy or asyncpg to insert rows into feature_store

# 🧪 PHASE 4: Testing
12. Write Tests
- ✅ Dummy data, no DB:
- 📁 tests/test_model.py, tests/test_api.py
- Write Integration Tests with Docker DB
    - Use docker-compose service for PostgreSQL
    - Test endpoints: predict/sync, predict/async

# 🐳 PHASE 5: Docker & Deployment
13. Docker is a great way to package your app 
- Create Dockerfile
- Create docker-compose.yml
- Test Docker Build
    - Visit: http://localhost:8000/healthcheck

# 🚀 PHASE 6: CI/CD with GitHub Actions
14. Create .github/workflows/test.yml
