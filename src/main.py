import mlflow
import os
import mlflow
import subprocess
import sys
import platform
import pkg_resources
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

# Set MLflow tracking URI from environment variable or default
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment("s3_bucket")

mlflow.autolog()

db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# Create and train models.
rf = RandomForestRegressor(n_estimators=160, max_depth=5, max_features=3)
with mlflow.start_run():
    mlflow.log_param("python_version", sys.version)
        # OS info
    mlflow.log_param("os", platform.platform())

    rf.fit(X_train, y_train)

    # Use the model to make predictions on the test dataset.
    predictions = rf.predict(X_test)
    
import subprocess

# Save git status to a file
with open("git_status.txt", "w") as f:
    subprocess.run(["git", "status"], stdout=f)
    
import boto3
try:
    gpu_info = subprocess.check_output("nvidia-smi --query-gpu=name --format=csv", shell=True).decode()
    mlflow.log_param("gpu", gpu_info.strip().split('\n')[1])
except Exception as e:
    mlflow.log_param("gpu", "N/A")

        # Installed libraries
with open("requirements_logged.txt", "w") as f:
    subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=f)
    mlflow.log_artifact("requirements_logged.txt")

s3 = boto3.client(
    "s3",
    endpoint_url=os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
    aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("MINIO_SECRET_ACCESS_KEY"),
    region_name="us-east-1"
)

bucket_name = "mlflow"
s3.upload_file("git_status.txt", bucket_name, "git_status.txt")

import psycopg2

with open("git_status.txt", "r") as f:
    git_status_content = f.read()

conn = psycopg2.connect(
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
    host="postgres"
)
cur = conn.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS git_logs (id SERIAL PRIMARY KEY, content TEXT);")
cur.execute("INSERT INTO git_logs (content) VALUES (%s);", (git_status_content,))
conn.commit()
cur.close()
conn.close()