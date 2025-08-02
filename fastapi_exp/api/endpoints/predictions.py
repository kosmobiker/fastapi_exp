from fastapi import APIRouter

router = APIRouter()

@router.post("/")
def predict_fraud():
    # We will add the prediction logic here later
    return {"message": "This is where the fraud prediction will be."}
