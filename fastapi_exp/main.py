from fastapi import FastAPI
from fastapi_exp.api.router import api_router

app = FastAPI(
    title="Fraud Detection API",
    description="An API for detecting fraudulent transactions.",
    version="1.0.0"
)

app.include_router(api_router, prefix="/api")

@app.get("/health", tags=["Health Check"])
def health_check():
    return {"status": "ok"}
