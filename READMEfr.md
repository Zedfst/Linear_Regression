# Régression Linéaire

Cette formation présente un modèle (algorithme) de machine learning, la régression linéaire. Elle explore les aspects théoriques de modèle et propose une étude de cas pratique.
De plus, elle présente comment gérer et déployer le modèle a des finsprofessionnelles, commerciales ou sociales.

<p align="center">
  <a href="README.md">Anglais</a>
  <a href="READMEfr.md">Français</a>
</p>

# Configuration de l'environnement

## Ubuntu

```bash
git clone https://github.com/Zedfst/Linear_Regression #Impoter le project en local
#Créer un environnement virtuel python
python3 -m venv nom_environement_virtuel # exemple: python -m venv venv
source nom_environement_virtuel/bin/activate # Pour activer l'environnement virtuel. Tapez deactivate pour le désactiver
python3 -m pip install -r requirements.txt #Installer les bibliothèques Python présentes dans le fichier Requirements.txt
```

## Windows

```bash
git clone https://github.com/Zedfst/Linear_Regression #Impoter le project en local
#Créer un environnement virtuel python
python3 -m venv nom_environement_virtuel # exemple: python -m venv venv
Get-ExecutionPolicy # Vérifier la politique d'exécution des scripts PowerShell. Si la valeur retournée est Restricted, entrez la commande ci-dessous
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
nom_environement_virtuel\Scripts\Activate # Pour activer l'environnement virtuel. Tapez deactivate pour le désactiver
python -m pip install -r requirements.txt # Installer les bibliothèques Python présentes dans le fichier Requirements.txt
```

Ouvrez un autre termial pour lancer MLflow (Ubuntu/Windows)

```bash
mlflow server --host 127.0.0.1 --port 8080#ctrl+c pour le stopper
```

# Ressources

[Pdf](src/regression_lineaire.pdf)

[Notebook](rent_prediction.ipynb)
