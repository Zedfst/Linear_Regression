from fastapi import FastAPI
import mlflow.sklearn
import numpy as np
from dataclasses import dataclass
import sys


@dataclass
class Observations():
    pieds_carres: float
    nombre_chambres: float
    nombre_etages: float
    distance_centre_ville: float


    
# obs=Observations(12.3,12345678901234567890,1,2.0)
# # obs=Observations(12.3,1,1,2.0)

# print(sys.getsizeof(obs.nombre_chambres))
# print(sys.getsizeof(obs.distance_centre_ville))

api=FastAPI()

mlflow.set_experiment('rent_prediction')
lr_model = mlflow.sklearn.load_model("runs:/0bdf16ae8bb5417184a6da6956390dfb/model")

@api.get('/model-v0')
def index(pieds_carres: float, nombre_chambres: int, nombre_etages: int, distance_centre_ville: float)->float:
    """
        Model: Régression Linéaire.

        Tâche: Prédiction du prix du loyer.

        Version: 0.0.0
    """
    obs=Observations(
        pieds_carres,
        nombre_chambres,
        nombre_etages,
        distance_centre_ville

    )
    pred=np.round(lr_model.predict([[obs.pieds_carres,obs.nombre_chambres,obs.nombre_etages,obs.distance_centre_ville]]),2)[0][0]
    #pred*=100000 <=> pred=pred*100000
    pred*=10000
    return pred
