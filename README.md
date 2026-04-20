![Version](https://img.shields.io/badge/version-v1.0.0-blue)
![License](https://img.shields.io/badge/License-Apache%202.0-red.svg)

# Linear Regression

This course introduces the linear regression algorithm by exploring the theoretical aspect and provides a practical example in python as well as the management and deployment of the model using mlflow and FastAPI.

<p align="center">
  <a href="README.md">English</a>
  <a href="READMEfr.md">French</a>
</p>

## 🚀 Get Started

From environment configuration to model deployment.

### ⚙️ 1. Environment Setting

#### Ubuntu

```bash
git clone https://github.com/Zedfst/Linear_Regression #Clone the repository locally
#Create virtual environment
python3 -m venv virtual_environment_name # exemple: python -m venv myvenv
source virtual_environment_name/bin/activate # Activate the virtaul environment. Enter deactivate to disable it
python3 -m pip install -r requirements.txt #Install the Python libraries present in the Requirements.txt file
```

#### Windows

```bash
git clone https://github.com/Zedfst/Linear_Regression #Import the project locally
cd Linear_Regression
code . #If you use VS code
#Create python virtual environment
python -m venv virtual_environment_name # example: python -m venv venv
Get-ExecutionPolicy # Check the PowerShell script execution policy. If the returned value is Restricted, enter the command below
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
virtual_environment_name\Scripts\Activate # To activate the virtual environment. Enter deactivate to deactivate it
python -m pip install -r requirements.txt # Install the Python libraries listed in the Requirements.txt file
```

### 📊 2. Run tracking engine: MLflow

MLflow is used to log our models and their artefacts for further evaluation, comparison, and deployment. All operations can be performed via an API or a web user interface.

Documentation: https://mlflow.org/docs/latest/ml/

Open a new terminal at the project root to launch MLflow.

```bash
mlflow server --host 127.0.0.1 --port 8080 #ctrl+c to stop it.
#Access your browser and enter http://localhost:8080

```

### 🤖 3. Model training

Open a new terminal at the project root and run the following command to begin training:

```bash
python3 rent_prediction.py -testing_size 0.2
```

Refer to the documentation for a deeper understanding of the model: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

### 🌐 4. Model deployement with FastAPI

FastAPI is a web framwork for building APIs with Python. We use it to obtain predictions from our model via the HTTP protocol.

Documentation: https://fastapi.tiangolo.com/

```bash
uvicorn api:api --host 127.0.0.1 --port 8000 --reload #ctrl+c to stop it.
#Access your browser and enter http://localhost:8000/docs

```

## 📚 Ressources

[Pdf](src/Linear_Regression.pdf)

[Notebook](rent_prediction.ipynb)
