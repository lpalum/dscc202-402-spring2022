# Databricks notebook source
# Create widget for parameter passing into the notebook
dbutils.widgets.text("data_input_path", "/dbfs/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.csv")

# COMMAND ----------

# Read from the widget
data_input_path = dbutils.widgets.get("data_input_path").strip()

# COMMAND ----------

# Log an artifact from the input path
import mlflow
name = 'multistep'
with mlflow.start_run(run_name=name) as run:
  # Log the data
  data_path = "data-csv"
  mlflow.log_artifact(data_input_path, data_path)
  
  run_id = run.info.run_id
  path = data_path
  #artifactURI = mlflow.get_artifact_uri()
  #full_path = dbutils.fs.ls(artifactURI + "/" + data_path)[0].path

# COMMAND ----------

# Report the results back to the parent notebook
import json

dbutils.notebook.exit(json.dumps({
  "status": "OK",
  "data_input_path": data_input_path,
  "run_id":run_id,
  "path":path,
  #"data_output_path": full_path.replace("dbfs:", "/dbfs")
}))


# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
