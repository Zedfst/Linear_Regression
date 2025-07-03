from fastapi import FastAPI

api=FastAPI()

@api.get('/model-v0')
def index():
    """
        Model: Régression Linéaire.

        Tâche: Prédiction du prix du loyer.

        Version: 0.0.0
    """
    return {"message":"U ma rambunra"}
