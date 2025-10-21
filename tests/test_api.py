import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "version" in data

def test_config_status():
    """Test configuration status endpoint"""
    response = client.get("/config/status")
    assert response.status_code == 200
    data = response.json()
    assert "environment" in data
    assert "paper_trading" in data
    assert "api_keys_configured" in data

def test_market_candidates():
    """Test market candidates endpoint"""
    response = client.get("/market/candidates")
    assert response.status_code == 200
    data = response.json()
    assert "candidates" in data
    assert "total_count" in data
    assert isinstance(data["candidates"], list)

def test_market_snapshot():
    """Test market snapshot endpoint"""
    response = client.get("/market/snapshot/SPY")
    assert response.status_code == 200
    data = response.json()
    assert "symbol" in data
    assert "price" in data
    assert "timestamp" in data

def test_spreads_scan():
    """Test spreads scanning endpoint"""
    payload = {
        "symbols": ["SPY"],
        "max_spreads_per_symbol": 1
    }
    response = client.post("/spreads/scan", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "spreads" in data
    assert "total_count" in data

def test_performance_analytics():
    """Test performance analytics endpoint"""
    response = client.get("/analytics/performance?days_back=30")
    assert response.status_code == 200
    data = response.json()
    # Should return either performance data or message about no trades

def test_ml_diagnostics():
    """Test ML diagnostics endpoint"""
    response = client.get("/ml/diagnostics")
    assert response.status_code == 200
    # ML model might not be trained yet, so just check for valid response

if __name__ == "__main__":
    pytest.main([__file__])