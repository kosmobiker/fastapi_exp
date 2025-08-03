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
✅ What: Evaluate on test set with:
    - Accuracy, Precision, Recall, F1, ROC-AUC
    - Confusion Matrix
    - Feature Importance
- build a script to create preprocesor and model and save it to `data/model_artifacts/`

### 📦 PHASE 2: Backend Project Setup
7. Set Up Project Structure
 - ✅ What: Use the structure I gave you earlier
    - 📁 Create folders: app/, tests/, data/, etc.

8. Create Pydantic Schemas
📁 app/schemas/predict.py

9. Write Model Inference Logic
📁 app/ml/model.py

10. Implement FastAPI Endpoints
/healthcheck: returns "ok"

/predict/sync/{user_id}: sync inference

/predict/async/{user_id}: async inference

📁 In app/api/endpoints/ → Create health.py, predict.py

📁 app/main.py

🗃️ PHASE 3: Feature Store (PostgreSQL)
11. Design Feature Store DB Tables
feature_store: user_id, features (JSON or array), created_at

predictions: user_id, prediction, timestamp

📁 app/db/models/feature.py, prediction.py

12. Configure SQLAlchemy + Alembic
DATABASE_URL in .env

Initialize Alembic: alembic init alembic

Create base.py and session.py

Run migration:

bash
Copy
Edit
alembic revision --autogenerate -m "init"
alembic upgrade head
13. Insert Transformed Features
Use SQLAlchemy or asyncpg to insert rows into feature_store

🧪 PHASE 4: Testing
14. Write Unit Tests
✅ Dummy data, no DB:
📁 tests/test_model.py, tests/test_api.py

Example test:

python
Copy
Edit
def test_dummy_prediction():
    features = [0.1] * 13
    prediction = predict(features)
    assert prediction in [0, 1]
15. Write Integration Tests with Docker DB
Use docker-compose service for PostgreSQL

Test endpoints: predict/sync, predict/async

🐳 PHASE 5: Docker & Deployment
16. Create Dockerfile
dockerfile
Copy
Edit
FROM python:3.11

WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
17. Create docker-compose.yml
Includes:

FastAPI app (web)

PostgreSQL (db)

Use .env for connection string

18. Test Docker Build
bash
Copy
Edit
docker-compose up --build
✅ Visit: http://localhost:8000/healthcheck

🚀 PHASE 6: CI/CD with GitHub Actions
19. Create .github/workflows/test.yml
See earlier answer for template

✅ Runs on each push, with Docker PostgreSQL and pytest

🌐 PHASE 7: Neon (Cloud Feature Store)
20. Create Neon DB
Go to Neon.tech

Create project → DB → copy connection string

21. Update .env for Production
env
Copy
Edit
DATABASE_URL=postgresql+asyncpg://user:pass@<neon>.neon.tech/db
✅ Use this in production only