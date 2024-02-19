# Import des librairies
from unittest import TestCase, main
from fastapi.testclient import TestClient
from api import app
import os
import pickle

# assertEqual(a, b) : Vérifie si a est égal à b.
# assertNotEqual(a, b) : Vérifie si a est différent de b.
        
# assertIn(a, b) : Vérifie si a est dans b.
# assertNotIn(a, b) : Vérifie si a n'est pas dans b.
        
# assertIs(a, b) : Vérifie si a est b.
# assertIsNot(a, b) : Vérifie si a n'est pas b.
        
# assertTrue(x) : Vérifie si x est vrai.
# assertFalse(x) : Vérifie si x est faux.
        
# assertIsNone(x) : Vérifie si x est None.
# assertIsNotNone(x) : Vérifie si x n'est pas None.
        
# assertIsInstance(a, b) : Vérifie si a est une instance de b.
# assertNotIsInstance(a, b) : Vérifie si a n'est pas une instance de b.
        
# assertRaises(exc, fun, *args, **kwargs) : Vérifie si fun(*args, **kwargs) lève une exception de type exc.
# assertRaisesRegex(exc, r, fun, *args, **kwargs) : Vérifie si fun(*args, **kwargs) lève une exception de type exc et dont le message correspond à l'expression régulière r.

# Tests unitaire de l'environnement de développement
class TestDev(TestCase):

    # Vérifie que les fichiers sont présents
    def test_files(self):
        list_files = os.listdir()
        self.assertIn("api.py", list_files)
        self.assertIn("model_1.pkl", list_files)
        self.assertIn("model_2.pkl", list_files)
        self.assertIn("Sleep_health_and_lifestyle_dataset.csv", list_files)

    # Vérifie que les requirements sont présents
    def test_requirements(self):
        list_files = os.listdir()
        self.assertIn("requirements.txt", list_files)
    
    # Vérifie que le gitignore est présent
    def test_gitignore(self):
        list_files = os.listdir()
        self.assertIn(".gitignore", list_files)
    

# Tests unitaire de l'API
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
            "Gender":  1,
            "Age":  30,
            "Physical_Activity_Level":  30,
            "Heart_Rate":  70,
            "Daily_Steps":  7000,
            "BloodPressure_high":  110,
            "BloodPressure_low":  75
            }  
        response = client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)



# Test du modèle individuellement
class TestModel(TestCase):

    # Vérifie que le modèle est bien présent
    def test_model_presence(self):
        list_files = os.listdir()
        self.assertIn("model_1.pkl", list_files)
    

    # Vérifie que le modèle est bien chargé
    def test_model_1_load(self):
        with open("model_1.pkl", "rb") as file:
            model = pickle.load(file)
        self.assertIsNotNone(model)
    

    # Vérifie que le modèle est bien chargé
    def test_model_2_load(self):
        with open("model_2.pkl", "rb") as file:
            model = pickle.load(file)
        self.assertIsNotNone(model)
    

# Démarrage des tests
if __name__== "__main__" :
    main(
        verbosity=2,
    )