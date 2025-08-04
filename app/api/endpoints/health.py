from fastapi import APIRouter

router = APIRouter()


@router.get("/healthcheck")
def health():
    return {"status": "ok"}
