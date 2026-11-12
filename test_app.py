# test_app.py
from fastapi.testclient import TestClient
from app import app

# Create a test client to simulate API requests without running the server
client = TestClient(app)

def test_predict_loan_success():
    """Test if a valid applicant gets a successful prediction."""
    payload = {
        "Age": 28,
        "Income_Lakhs": 12.5,
        "Credit_Score": 750
    }
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert type(response.json()["loan_approved"]) == bool

def test_predict_loan_validation_error():
    """Test if Pydantic properly blocks invalid data (e.g., Age < 18)."""
    payload = {
        "Age": 16,  # This violates your gt=17 Pydantic rule
        "Income_Lakhs": 5.0,
        "Credit_Score": 600
    }
    response = client.post("/predict", json=payload)
    
    # FastAPI should automatically throw a 422 Unprocessable Entity error
    assert response.status_code == 422     