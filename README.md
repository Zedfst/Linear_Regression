# Linear_Regression

This course introduces the linear regression algorithm by exploring the theoretical aspect and provides a practical example in python as well as the management and deployment of the model using mlflow and FastAPI.

<p align="center">
  <a href="README.md">English</a>
  <a href="READMEfr.md">French</a>
</p>

# Environment Setting

## Ubuntu

```bash
git clone https://github.com/Zedfst/Linear_Regression #Clone the repository locally.
#Create virtual environment
python3 -m venv virtual_environment_name # exemple: python -m venv myvenv
source virtual_environment_name/bin/activate # Activate the virtaul environment. Enter deactivate to disable it.
python3 -m pip install -r Requirements.txt #Install the Python libraries present in the Requirements.txt file.
```

Open another terminal to launch MLflow

```bash
mlflow server --host 127.0.0.1 --port 8080#ctrl+c to stop it.

```
