from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow
import numpy as np

app = FastAPI(title="NBFC Loan Approval API")

try:
    model = mlflow.sklearn.load_model("mlruns/1/models/m-3daf51e5f53e45eaa7a6f5b0d1c4d73c/artifacts")
except Exception as e:
    model = None
    print(f"Model failed to load: {e}")

class LoanApplicant(BaseModel):
    Age: int = Field(..., gt=17, description="Applicant's age")
    Income_Lakhs: float = Field(..., gt=0, description="Annual income in Lakhs")
    Credit_Score: int = Field(..., ge=300, le=900, description="CIBIL/Credit Score")

@app.post("/predict")
def predict_loan(applicant: LoanApplicant):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    try:
        input_data = np.array([[applicant.Age, applicant.Income_Lakhs, applicant.Credit_Score]])
        prediction = int(model.predict(input_data)[0])
        
        return {
            "status": "success",
            "loan_approved": bool(prediction)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")