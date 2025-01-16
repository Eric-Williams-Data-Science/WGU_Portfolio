#!/usr/bin/env python
# coding: utf-8

# In[3]:


from fastapi.testclient import TestClient
from API import app

client = TestClient(app)

def test_root():
    """Testing the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API is functional!"}

def test_invalid_airport():
    """Testing the /predict/delays endpoint (with an invalid airport)"""
    response = client.get("/predict/delays", params={
        "arrival_airport": "XYZ",  # Invalid airport code
        "departure_airport": "LAX",
        "departure_time": "2024-10-31T14:30:00",
        "arrival_time": "2024-10-31T22:15:00"
    })
    assert response.status_code == 404
    assert response.json() == {"detail": "Arrival airport not found"}

def test_invalid_time_format():
    """Testing the /predict/delays endpoint (with an invalid time format)"""
    response = client.get("/predict/delays", params={
        "arrival_airport": "JFK",
        "departure_airport": "LAX",
        "departure_time": "14:30",  # Invalid format
        "arrival_time": "2024-10-31T22:15:00"
    })
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid time format. Please use 'YYYY-MM-DDTHH:MM:SS'."}

