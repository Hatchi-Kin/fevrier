from unittest import TestCase, main
from fastapi.testclient import TestClient
from api import app
import os

class TestDev(TestCase):
# Vérifie que les fichiers sont présents
    
    def test_files(self):
        list_files = os.listdir()
        self.assertIn("api.py", list_files)

    def test_requirements(self):
        list_files = os.listdir()
        self.assertIn("requirements.txt", list_files)

    def test_gitignore(self):
        list_files = os.listdir()
        self.assertIn(".gitignore", list_files)


class TestAPI(TestCase):

    # Vérifie que l'API est bien lancée
    def test_api(self):
        client = TestClient(app)
        response = client.get("/hello")
        self.assertEqual(response.status_code, 200)

    # Vérifie le endpoint predict
    def test_predict(self):
        client = TestClient(app)
        payload = {
            "Gender": 1,
            "Age": 30,
            "Physical_Activity_Level": 30,
            "Heart_Rate": 70,
            "Daily_Steps": 7000,
            "BloodPressure_high": 110,
            "BloodPressure_low": 75,
        }
        response = client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    main(verbosity=2)
