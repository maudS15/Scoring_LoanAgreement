import unittest
import json
from app import app, valider_donnees
import pandas as pd

X = pd.read_csv('Cleaned/app_test.csv', index_col='SK_ID_CURR')
features = X.columns.tolist()
l = len(features)

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_predict_route(self):
        # Envoyer une requête POST avec des données fictives
        data_correctes = {'data': [[0] * l],
                          'columns':features}
        response = self.app.post('/predict/', json=data_correctes)

        # Vérifier le code de statut de la réponse
        self.assertEqual(response.status_code, 200)


    def test_valider_donnees(self):
        # Données correctes
        data_correctes = {'data': [[0] * l],
                          'columns': features}
        
        self.assertIsNone(valider_donnees(data_correctes))  # La fonction ne doit pas retourner d'erreur

        # Données manquantes
        data_manquantes = {'columns': features}
        with self.assertRaises(ValueError):
            valider_donnees(data_manquantes)
            
        # Colonnes manquantes
        col_manquantes = {'data': [[0] * l]}
        with self.assertRaises(ValueError):
            valider_donnees(col_manquantes)

        # Données incorrectes (longueur incorrecte)
        data_incorrectes_longueur = {'data': [[0] * 99]}
        with self.assertRaises(ValueError):
            valider_donnees(data_incorrectes_longueur)

if __name__ == '__main__':
    unittest.main()
