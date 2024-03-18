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
            "PassengerId": 892,
            "Pclass": 3,
            "Sex": 1,
            "Age": 34.5,
            "SibSp": 0,
            "Parch": 0,
            "Fare": 7.8292,
            "Embarked": 1,
        }
        response = client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    main(verbosity=2)
