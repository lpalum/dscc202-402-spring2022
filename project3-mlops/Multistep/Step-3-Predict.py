# Databricks notebook source
# Create widget for parameter passing into the notebook
dbutils.widgets.text("model_path", "")
dbutils.widgets.text("data_path", "")

# COMMAND ----------

# Read from the widget
model_path = dbutils.widgets.get("model_path").strip()
data_path = dbutils.widgets.get("data_path").strip()

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
import tempfile

with mlflow.start_run() as run:
  # Import the data
  df = pd.read_csv(data_path).drop(["price"], axis=1)
  model = mlflow.sklearn.load_model(model_path)

  predictions = model.predict(df)
  
  temp = tempfile.NamedTemporaryFile(prefix="predictions_", suffix=".csv")
  temp_name = temp.name
  try:
    pd.DataFrame(predictions).to_csv(temp_name)
    mlflow.log_artifact(temp_name, "predictions.csv")
  finally:
    temp.close() # Delete the temp file
    
  #artifactURI = mlflow.get_artifact_uri()
  #predictions_output_path = artifactURI + "/predictions.csv"
  run_id = run.info.run_id
  data_dir = temp_name
  

# COMMAND ----------

# Report the results back to the parent notebook
import json

dbutils.notebook.exit(json.dumps({
  "status": "OK",
  "run_id": run_id,
  "data_dir": data_dir
}))


# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
