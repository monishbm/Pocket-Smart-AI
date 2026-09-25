import importlib
from fastapi.testclient import TestClient

def make_client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH",str(tmp_path/"test.db"))
    monkeypatch.setenv("SECRET_KEY","test-secret-key-long-enough")
    monkeypatch.delenv("GOOGLE_API_KEY",raising=False)
    monkeypatch.delenv("GEMINI_API_KEY",raising=False)
    import database, app
    importlib.reload(database); importlib.reload(app)
    return TestClient(app.app)

def test_health(tmp_path, monkeypatch):
    with make_client(tmp_path,monkeypatch) as client:
        r=client.get("/health")
        assert r.status_code==200 and r.json()["status"]=="ok"

def test_register_login_home_history(tmp_path, monkeypatch):
    with make_client(tmp_path,monkeypatch) as client:
        r=client.post("/register",data={"full_name":"Test User","username":"tester","email":"tester@example.com","password":"password123"},follow_redirects=False)
        assert r.status_code==303
        r=client.post("/login",data={"username":"tester","password":"password123"},follow_redirects=False)
        assert r.status_code==303 and "access_token" in client.cookies
        r=client.post("/home-budget",data={"total_budget":"5000","num_lights":"5","num_fans":"2","num_furniture":"2","num_dining_tables":"1","has_living_room":"true"})
        assert r.status_code==200 and "Your budget plan" in r.text
        r=client.get("/recommendation-history")
        assert r.status_code==200 and len(r.json())==1
