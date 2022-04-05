# Databricks notebook source
# Create widget for parameter passing into the notebook
dbutils.widgets.text("run_id", "")
dbutils.widgets.text("path", "")
dbutils.widgets.text("n_estimators", "10")
dbutils.widgets.text("max_depth", "20")
dbutils.widgets.text("max_features", "auto")

# COMMAND ----------

import os
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
local_dir = "/tmp/artifact_downloads"
if not os.path.exists(local_dir):
    os.mkdir(local_dir)
local_path = client.download_artifacts(dbutils.widgets.get("run_id").strip(), dbutils.widgets.get("path").strip(), local_dir)
print("Artifacts downloaded in: {}".format(local_path))
print("Artifacts: {}".format(os.listdir(local_path)))

# COMMAND ----------

# Read from the widget
artifact_URI = local_path + "/" + os.listdir(local_path)[0]
n_estimators = int(dbutils.widgets.get("n_estimators"))    # Cast to Int
max_depth = int(dbutils.widgets.get("max_depth"))          # Cast to Int
max_features = dbutils.widgets.get("max_features").strip()

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

with mlflow.start_run() as run:
  # Import the data
  df = pd.read_csv(artifact_URI)
  X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df[["price"]].values.ravel(), random_state=42)
    
  # Create model, train it, and create predictions
  rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)
  rf.fit(X_train, y_train)
  predictions = rf.predict(X_test)

  # Log model
  model_path = "random-forest-model"
  mlflow.sklearn.log_model(rf, model_path)
    
  # Log params
  mlflow.log_param("n_estimators", n_estimators)
  mlflow.log_param("max_depth", max_depth)
  mlflow.log_param("max_features", max_features)

  # Log metrics
  mlflow.log_metric("mse", mean_squared_error(y_test, predictions))
  mlflow.log_metric("mae", mean_absolute_error(y_test, predictions))  
  mlflow.log_metric("r2", r2_score(y_test, predictions)) 
  
  #artifactURI = mlflow.get_artifact_uri()
  model_output_path = "runs:/" + run.info.run_id + "/" + model_path

# COMMAND ----------

# Report the results back to the parent notebook
import json

dbutils.notebook.exit(json.dumps({
  "status": "OK",
  "model_output_path": model_output_path, #.replace("dbfs:", "/dbfs")
  "data_path": artifact_URI
}))


# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
