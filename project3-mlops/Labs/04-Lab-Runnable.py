# Databricks notebook source
# TODO
dbutils.widgets.text("n_estimators", "10")
dbutils.widgets.text("learning_rate", "0.1")
dbutils.widgets.text("max_depth", "1")

# Create 3 widgets for parameter passing into the notebook:
#   - n_estimators with a default of 100
#   - learning_rate with a default of .1
#   - max_depth with a default of 1 
# Note that only strings can be used for widgets

# COMMAND ----------

# TODO
# Read from the widgets to create 3 variables.  Be sure to cast the values to numeric types
n_estimators = int(dbutils.widgets.get("n_estimators"))
learning_rate = float(dbutils.widgets.get("learning_rate"))
max_depth = int(dbutils.widgets.get("max_depth"))

# COMMAND ----------

# TODO
# Train and log the results from a model.  Try using Gradient Boosted Trees
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
data_path="/dbfs/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.csv"
with mlflow.start_run() as run:
    # Import the data
    df = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df[["price"]].values.ravel(), random_state=42)

    # Create model, train it, and create predictions
    gb= GradientBoostingRegressor(learning_rate=learning_rate,max_depth=max_depth,n_estimators=n_estimators)
    gb.fit(X_train, y_train)
    predictions = gb.predict(X_test)

    # Log model
    model_path = "gradient-boost"
    mlflow.sklearn.log_model(gb, model_path)

    # Log params
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("max_depth", max_depth)

    # Log metrics
    mlflow.log_metric("mse", mean_squared_error(y_test, predictions))
    mlflow.log_metric("mae", mean_absolute_error(y_test, predictions))  
    mlflow.log_metric("r2", r2_score(y_test, predictions)) 

    #artifactURI = mlflow.get_artifact_uri()
    model_output_path = "runs:/" + run.info.run_id + "/" + model_path

# COMMAND ----------

# TODO
# Report the model output path to the parent notebook
import json

dbutils.notebook.exit(json.dumps({
  "status": "OK",
  "model_output_path": model_output_path, #.replace("dbfs:", "/dbfs")
#   "data_path": artifact_URI
}))


# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
