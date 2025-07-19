# Prédiction du prix du loyer
# Dataset: https://www.kaggle.com/datasets/denkuznetz/housing-prices-regression/data

# Ce notebook introduit  un cas pratique d'utilisation de  régression linéaire pour la prévision des prix de location. Les valeurs observées (utilisées pour faire la prédiction) sont les suivantes :
# - pieds_carres
# - nombre_chambres
# - nombre_douches
# - annee_construction
# - avec_jardin
# - avec_piscine
# - taille_garage
# - score_localisation
# - distance_centre_ville



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import time
from sklearn.preprocessing import StandardScaler
import random
import joblib


## Numpy
# Bibliothèque de calcul numérique sur les matrices.


#Créer une matrice de dimension 3 X 4 avec des valeurs aléatoires
matrice_1=np.random.rand(3,4)
print(matrice_1)


#Création d'une matrice
print('\n')
M1=np.array([
    [12,14,45],[10,5,3]
])
print(f'Matrice 1')
print(M1)

print('\n')

M2=np.array([
    [1,8.0,12],[11,56,73]
])
print(f'Matrice 2')
print(M2)

print('\n')

M3=np.array([
    [2.3,8.0,12],[10.7,9.6,73],[5.7,12,7.4]
])
print(f'Matrice 3')
print(M3)


print('\n')


print(f'Matrice 1 + Matrice 2')
print(M1 + M2)

print('\n')

print(f'Matrice 1 + Matrice 2')
print(M1 * M2)

print('\n')

print(f'Matrice 1 @ Matrice 3')
print(M1@M3)


print('\n')

print(f'2 x Matrice 1')
print(2*M1)


## Excercices
# Calculer
# -  $M_2+M3$
# -  $\alpha * M2$
# -  $M_1@M2$
# $$
# M_1=\begin{bmatrix}
# 12.5 & 5.5 & 3.2 \\
# 7.2 & 6.8 & 2.9 
# \end{bmatrix} ; 
# M_2=\begin{bmatrix}
# 23.3 & 4.2 & 5.6 \\
# 3.3 & 2.5 & 9.6 \\
# 12.3 & 4.7 & 7.6 
# \end{bmatrix} ; 
# M_3=\begin{bmatrix}
# 10.7 & 6.8 & 4.9 \\
# 45.8 & 10.4 & 12 \\
# 3.8 & 2.6 & 2.1 
# \end{bmatrix} ;\alpha=2
# $$

## Pandas
# Permet de manipuler des données structurées.

data=pd.read_csv('src/data1.csv')#Lire le fichier csv.
data.head()#Affiche les 5 premières lignes du dataframe.

#Renommer les colonnes en français
nouveaux_noms_colonnes={
    "Square_Feet":"pieds_carres",#Evitez de nommer les colonnes comme suit: pieds carrés
    "Num_Bedrooms":"nombre_chambres",
    "Num_Bathrooms":"nombre_douches",
    "Num_Floors":"nombre_etages",
    "Year_Built":"annee_construction",
    "Has_Garden":"avec_jardin",
    "Has_Pool":"avec_piscine",
    "Garage_Size":"taille_garage",
    "Location_Score":"score_localisation",
    "Distance_to_Center":"distance_centre_ville",
    "Price":"prix"
}

data.rename(columns=nouveaux_noms_colonnes,inplace=True)
#Inplace permet de modifier directement le dataframe. Sa valeur par défaut est False.
#Si vous auriez défini inplace à False vous auriez   réaffecter la valeur data comme ci-dessous.
# data=data.rename(columns=nouveaux_noms_colonnes,inplace=False)
data.head(2)

print(f'Colonnes du dataframe -> {data.columns}')
print(type(data.columns))
print('\n')
colonnes=data.columns.to_list()
print(f'Colonnes du dataframe en format liste-> {colonnes}')
print(type(colonnes))

data.info()#Info sur les colonnes/observations/features et sur le dataframe.

#Convertir les dates en durée en fonction de la date la plus récente.
max_date=data.annee_construction.max()
print(f'maximum date {max_date}')
data['annee_construction']=data['annee_construction'].apply(lambda x: max_date-x)
data.head(2)


#Diviser les prix par 100000 pour avoir des valeurs plus petites.
data['prix']=data['prix'].apply(lambda x: x/10000)
data.head(2)

# Changons le type de données des colonnes catégorielles/discrètes listé ci-dessous  en type object pour des raisons d'études statistiques: 
# - avec_jardin
# - avec_piscine
# - annee_construction

data[['avec_jardin','avec_piscine']]=data[['avec_jardin','avec_piscine']].astype(object)
data.info()

data.describe()#Statistique des données numériques.

data.describe(include=['object'])#Statistique des données catégorielles.

data.duplicated().values.any()#Vérifier si il y a des linges qui se répètent (doublons/duplicates).

data.isnull().values.any()#Vérifier si le dataframe contient des valeurs manquantes, généralement marquées comme NAN.

# Les valeurs manquantes peuvent être remplacées par :
# - La moyenne. Example $[2,NAN,10]$->$[2,6,10]$; 
# - La médiane;
# - Le mode (valeurs catégorielles)
# - Ou des méthodes plus avancées 

# Supposons que des valeurs soient manquantes dans la colonne « pieds carrés ». Nous aurions remplacé ces valeurs comme ci-dessous.

_ = data[['pieds_carres']].fillna(data[['pieds_carres']].mean(numeric_only=True))#Undescore est généralement utilisé pour nommer une variable que l'on ne souhaite pas utilisée.

## Matplotlib & Seaborn
# Bibliothèques de visualisation

sns.jointplot(x="pieds_carres",y="prix",data=data)

sns.pairplot(data[['pieds_carres','nombre_douches','taille_garage','prix','annee_construction','distance_centre_ville','score_localisation']],kind='scatter',plot_kws={"alpha":0.4})

#Supprimons les valeurs catégorielles
colonnes.remove('ID')
colonnes.remove('avec_piscine')
colonnes.remove('avec_jardin')

colonnes

#Selectionner au hasard les colonnes que vous voulez analyser.
random_colonnes=random.sample(colonnes,k=4)
print(f'Colonnes choisies -> {random_colonnes}')
sns.pairplot(data[random_colonnes],kind='scatter',plot_kws={"alpha":0.4})

## Entraînement du modèle avec Scikit-learn
# Division des données. Une partie pour l'entraînement, une partie pour l'évaluation du modèle (également appelée test du modèles).

# observations=['pieds_carres','nombre_chambres','nombre_etages','distance_centre_ville','annee_construction','taille_garage','score_localisation','avec_piscine']
observations=['pieds_carres','nombre_chambres','nombre_etages','distance_centre_ville']


X=data[observations].values#Observations

print('Example des observations/features avant la standardization.')
print(X[0])

scaler = StandardScaler()
X_ = scaler.fit_transform(X)
print('\n')
print('Example des observations/features après la standardization.')
print(X_[0])

y=data[['prix']].values#Target (cible)
#2% des données (soit 100 échantillons/samples) seront utilisées pour le test.
#random_state permet  de reproduire la distribution des données d'entraînement et de test.
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1234)

X_train=scaler.fit_transform(X_train)#Standardiser les données d'entrainement.

#Enregistrer manuellement l'objet scaler
joblib.dump(scaler, 'src/scaler.pkl')

### MLflow pour tracker et logger les résultats.

mlflow.set_experiment('rent_prediction')
mlflow.sklearn.autolog()#Active le tracking de l'expérience avec mlflow

lr=LinearRegression(n_jobs=3)#Initialisation du modèle
#Tous les résultats relatifs à l'entraînement du modèle seront enregistrés dans Mlflow.
# with mlflow.start_run(run_id='523ec6f6a5944e0e8233fe4342b1e837',run_name='model1') as run:
with mlflow.start_run() as run:
    lr.fit(X_train,y_train)#Entrainement

## Evaluation du modèle

#Standardiser les données en fonction de la moyenne et l'écart type apprise sur les données d'entrainement.
X_test=scaler.transform(X_test)
pred=lr.predict(X_test)

print(f'Dimenssion du vecteur de prédiction {pred.shape}')
print(f'Dimenssion du vecteur de cibles {y_test.shape}')

#Concaténer la prediction et la cible pour avoir une matrice de taille 100 x 2
print(type(y_test))
pred_ytest=np.concatenate([pred,y_test],1)

#Convertir en dataframe pour des raison de visualization.
df_test=pd.DataFrame(pred_ytest)
df_test.head(2)

df_test.rename(columns={0:"prediction",1:"cible"},inplace=True)
df_test.head(2)

## Analyse des résultats

sns.scatterplot(df_test,x='prediction',y='cible')

erreur=mean_squared_error(y_test,pred)
print(f'Erreur {erreur}')

# Condons la fonction d'erreur MSE ci-dessous from scratch avec numpy.

# $MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$

erreur=np.sum(np.square(y_test-pred))/len(y_test)
print(f'Erreur {erreur}')

## Interpretations des des poids

# $lr=y=a_1x_1 + a_2x_2 + a_3x_3 + a_4x_4 + a_5x_5 + b$

# $lr=y=a_1{pieds\_carres} + a_2{nombre\_chambres} + a_3{nombre\_etages} + a_4{distance\_centre_ville} +a_5{annee\_construction}$ + b

print(f'Coefficients du modèle {lr.coef_}')
print(f'Intercept du modèle {lr.intercept_}')

for index,obs in enumerate(observations):
    print(f'{obs} => {lr.coef_[0][index]}')
# lr.coef_
# X=data[].values#Observations

## Enregistrement manuel des poids modèle

np.save('weights/lr_coeficients',lr.coef_)
np.save('weights/lr_intercept',lr.intercept_)

coeficients=np.load('weights/lr_coeficients.npy')
interecpt=np.load('weights/lr_intercept.npy')

lr_=LinearRegression(n_jobs=3)
lr_.coef_=coeficients
lr_.intercept_=interecpt

print(f"Dimension d'une seule observation {X_test[0].shape}")
print(f"Observation redimensionner {X_test[0].reshape(-1,X_test[0].shape[0]).shape}")

#Résultat avec le modèle dont les poids ont été chargés.
pred_=lr_.predict(X_test[0].reshape(-1,X_test[0].shape[0]))
print(pred_)

#Résultat avec le modèle initial.
pred=lr.predict(X_test[0].reshape(-1,X_test[0].shape[0]))
print(pred)

#Une autre façon de vérifier si le chargement des poids est effectué correctement.
assert pred_==pred, "Error"