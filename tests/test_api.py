from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200

def test_health():
    response = client.get("/health")
    assert response.status_code == 200

def test_prediction():

    payload = {
        "GRE_Score":320,
        "TOEFL_Score":110,
        "University_Rating":4,
        "SOP":4.5,
        "LOR":4,
        "CGPA":9.1,
        "Research":1
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200
    assert "chance_of_admit" in response.json()