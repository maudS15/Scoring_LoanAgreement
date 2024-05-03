from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np


app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

# Chargement du modèle MLflow
model_path = 'GBoost/model.pkl'
with open(model_path, 'rb') as fichier:
    model = pickle.load(fichier)

# Fonction de validation des données
def valider_donnees(data):
    if 'data' not in data:
        raise ValueError("Les données doivent contenir une clé 'data'")
    if len(data['data'][0]) != 234:
        raise ValueError("Les données doivent contenir exactement 234 valeurs")
    if 'columns' not in data:
        raise ValueError("Les données doivent contenir une clé 'columns'")
        

# Route pour la prédiction
@app.route('/predict/', methods=['POST'])
def predict():
    try:
        # Valider les données
        valider_donnees(request.json)

        # Préparer les données pour la prédiction
        input_data = pd.DataFrame([request.json['data'][0]],
                                  columns=[request.json['columns']])
        input_data.replace(to_replace='', value=np.nan, inplace=True)

        # Pour la probabilité de predict_proba, si votre modèle le supporte
        proba = model.predict_proba(input_data)[:,1]

        # Renvoyer la prédiction
        #return jsonify({'probability': [0.1]})
        print('proba', proba)
        return jsonify({'probability': proba.tolist()})
    except ValueError as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run()