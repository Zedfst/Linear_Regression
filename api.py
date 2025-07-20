from fastapi import FastAPI
import mlflow.sklearn
import numpy as np
from dataclasses import dataclass
import sys
from sklearn.preprocessing import StandardScaler
import joblib
import mlflow

scaler = joblib.load("src/scaler.pkl")


@dataclass
class Observations():
    pieds_carres: float
    nombre_chambres: float
    nombre_etages: float
    distance_centre_ville: float
    annee_construction:float

    def compact(self,):
        return np.array([self.pieds_carres,self.nombre_chambres,self.nombre_etages,self.distance_centre_ville,self.annee_construction]).reshape(-1,5)


    
# obs=Observations(12.3,12345678901234567890,1,2.0)
# # obs=Observations(12.3,1,1,2.0)

# print(sys.getsizeof(obs.nombre_chambres))
# print(sys.getsizeof(obs.distance_centre_ville))

api=FastAPI()

mlflow.set_experiment('prediction_loyer')
lr_model = mlflow.sklearn.load_model("models:/model_lr@actual")

@api.get('/model-v0')
def index(pieds_carres: float, nombre_chambres: int, nombre_etages: int, distance_centre_ville: float,annee_construction: float)->float:
    """
        Model: Régression Linéaire.

        Tâche: Prédiction du prix du loyer.

        Version: 0.0.0
    """
    print('______________________')
    obs=Observations(
        pieds_carres,
        nombre_chambres,
        nombre_etages,
        distance_centre_ville,
        annee_construction,

    )
    features=scaler.transform(obs.compact())
    pred=np.round(lr_model.predict(features),2)[0][0]
    #pred*=100000 <=> pred=pred*100000
    pred*=10000
    return pred
