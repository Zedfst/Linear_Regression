#Import the necessary Python libraries
import pandas as pd
# import numpy as np
import mlflow
from mlflow.models.signature import infer_signature
# import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# from datetime import time
from sklearn.preprocessing import StandardScaler
# import random
import joblib
import argparse
parser=argparse.ArgumentParser()


#==============================================cli parameters=========================
parser.add_argument("-training_size","--training_size",type=float,default=0.2)
parser.add_argument("-n_jobs","--n_jobs",type=str,default='normal')
# parser.add_argument("-dimembs","--dimension_embeddings",type=int)
# parser.add_argument("-batchsize","--batch_size",type=int)
# parser.add_argument("-uf1","--units_fnn1",type=int)
# parser.add_argument("-drp1","--dropout1",type=float,default=0.5)
# parser.add_argument("-drp2","--dropout2",type=float,default=0.5)
# parser.add_argument("-lindim","--lin_dim",type=int,default=5)
# parser.add_argument("-icdv","--icd_version",type=int,required=True,default=10)
args=parser.parse_args()
#=================================================Data loading===================

data=pd.read_csv('src/data1.csv')#Load the csv file
data.info()
print('\n')

#Map each date to a duration based on the most recent date.
max_date=data.Year_Built.max()
data['Year_Built']=data['Year_Built'].apply(lambda x: max_date-x)

#Rescale target values ​​by dividing by 100000
data['Price']=data['Price'].apply(lambda x: x/10000)

##Change the data type of the Has_Pool and Has_Garden features to object so that they can be processed as categorical/discrete features.
data[['Has_Garden','Has_Pool']]=data[['Has_Garden','Has_Pool']].astype(object)

observations=['Square_Feet','Num_Bedrooms','Num_Floors','Year_Built','Distance_to_Center']
X=data[observations].values#Features
y=data[['Price']].values#Target

scaler = StandardScaler()
X_stand = scaler.fit_transform(X)#Standardisation

#Save the scaler object to standardise test data
joblib.dump(scaler, 'src/scaler.pkl')

#2% of the 100 samples are used for testing.
#By default the train_test_split function, splits the data randomly. random_state parameter allows reproduicing the data splitting scenario.
X_train,X_test,y_train,y_test=train_test_split(X_stand,y,test_size=args.training_size,random_state=0)
X_train=scaler.fit_transform(X_train)#Standardise training data (only features).

#MLflow settings
mlflow.end_run()#End the current run if any. 
mlflow.set_tracking_uri('http://127.0.0.1:8080')
mlflow.set_experiment('rent_prediction')
mlflow.start_run(run_name='my_model')

# mlflow.sklearn.autolog()
lr=LinearRegression(n_jobs=3)#Model initialization
lr.fit(X_train,y_train)#Training

#Standardise test data
X_test=scaler.transform(X_test)
pred=lr.predict(X_test)#Predictions
mse=mean_squared_error(y_test,pred)
print(f'Mean Squared Error (MSE) {mse}')
print('\n')
signature = infer_signature(X_test, lr.predict(X_test))
mlflow.log_metric("mse", mse)
#name-> Define the model version.
mlflow.sklearn.log_model(lr, name="model-v1",input_example=X_test[:5],signature=signature)
mlflow.log_artifact('src/scaler.pkl')#Register the scaler object to MLflow. It will be used to process inputs during the deployment phase.







