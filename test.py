from unittest import TestCase, main
from fastapi.testclient import TestClient
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


if __name__ == "__main__":
    main(verbosity=2)
