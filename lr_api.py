from fastapi import FastAPI
import mlflow.sklearn
import numpy as np

api=FastAPI()

mlflow.set_experiment('rent_prediction')
lr_model = mlflow.sklearn.load_model("runs:/289a195b27074060920e9b8ba8f2c5b9/model")

# @api.get('/model-v0')
def index():
    """
        Model: Régression Linéaire.

        Tâche: Prédiction du prix du loyer.

        Version: 0.0.0
    """
    pred=np.round(lr_model.predict([[250.54924519,3.,1.]]),2)[0][0]
    #pred*=100000 <=> pred=pred*100000
    pred*=100000
    return pred
