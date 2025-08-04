from fastapi import FastAPI

from app.api.router import api_router

app = FastAPI(title="ML Inference API")
app.include_router(api_router)


@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok"}
