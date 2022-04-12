# Databricks notebook source
# MAGIC %md
# MAGIC # Lab: Grid Search with MLflow
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lab you:<br>
# MAGIC  - Import the housing data
# MAGIC  - Perform grid search using scikit-learn
# MAGIC  - Log the best model on MLflow
# MAGIC  - Load the saved model
# MAGIC  
# MAGIC ## Prerequisites
# MAGIC - Web browser: Chrome
# MAGIC - A cluster configured with **8 cores** and **DBR 7.0 ML**

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Classroom-Setup
# MAGIC 
# MAGIC For each lesson to execute correctly, please make sure to run the **`Classroom-Setup`** cell at the<br/>
# MAGIC start of each lesson (see the next cell) and the **`Classroom-Cleanup`** cell at the end of each lesson.

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Import
# MAGIC 
# MAGIC Load in same Airbnb data and create train/test split.

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("/dbfs/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.csv")
X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df[["price"]].values.ravel(), random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Perform Grid Search using scikit-learn
# MAGIC 
# MAGIC We want to know which combination of hyperparameter values is the most effective. Fill in the code below to perform <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV" target="_blank"> grid search using `sklearn`</a> over the 2 hyperparameters we looked at in the 02 notebook, `n_estimators` and `max_depth`.

# COMMAND ----------

# TODO
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# dictionary containing hyperparameter names and list of values we want to try
# hyperparameters from 02 tracking notebook
parameters = {'n_estimators': [100, 1000] , 
              'max_depth': [5, 10] }

rf = RandomForestRegressor()
grid_rf_model = GridSearchCV(rf, parameters, cv=3)
grid_rf_model.fit(X_train, y_train)

best_rf = grid_rf_model.best_estimator_
for p in parameters:
  print("Best '{}': {}".format(p, best_rf.get_params()[p]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log Best Model on MLflow
# MAGIC 
# MAGIC Log the best model as `grid-random-forest-model`, its parameters, and its MSE metric under a run with name `RF-Grid-Search` in our new MLflow experiment.

# COMMAND ----------

# TODO
from sklearn.metrics import mean_squared_error

parameters = {'n_estimators': [1000] , 
              'max_depth': [10] }

with mlflow.start_run(run_name= "RF-Grid-Search") as run:
    # Create predictions of X_test using best model
    rf = RandomForestRegressor()
    rf_grid_model = GridSearchCV(rf, parameters, cv=3)
#     grid_rf_model = best_rf
    rf_grid_model.fit(X_train, y_train)
    predictions = rf_grid_model.predict(X_test)
  
  
  # Log model with name
    mlflow.sklearn.log_model(rf_grid_model, "grid-random-forest-model")
  
  # Log params
    mlflow.log_params(parameters)
  
  # Create and log MSE metrics using predictions of X_test and its actual value y_test
    mse = mean_squared_error(y_test, predictions)
    mlflow.log_metrics({"mse": mse})
    
    experimentID = run.info.experiment_id
    runID = run.info.run_uuid
    rfURI = run.info.artifact_uri
    print("Inside MLflow Run with id {}".format(runID))

# COMMAND ----------

# MAGIC %md
# MAGIC Check on the MLflow UI that the run `RF-Grid-Search` is logged has the best parameter values found by grid search.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Load the Saved Model
# MAGIC 
# MAGIC Load the trained and tuned model we just saved. Check that the hyperparameters of this model matches that of the best model we found earlier.
# MAGIC 
# MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** Use the `artifactURI` variable declared above.

# COMMAND ----------

# TODO
from  mlflow.tracking import MlflowClient
client = MlflowClient()
# runs = client.search_runs(experimentID, max_results=1)
# # runs[0].data.metrics
# # runs[0] = client.search_runs(experimentID)
# model = mlflow.sklearn.load_model('runs:/'+runs[0].info.run_id+"/grid-random-forest-model")
# model.feature_importances_

model = rf_grid_model.best_estimator_
for p in parameters:
  print("Best '{}': {}".format(p, best_rf.get_params()[p]))
  
print(rfURI)
model = mlflow.sklearn.load_model(rfURI)
print(model.feature_importances_)

# COMMAND ----------

# MAGIC %md
# MAGIC Time permitting, continue to grid search over a wider number of parameters and automatically save the best performing parameters back to `mlflow`.

# COMMAND ----------

# TODO
parameters = {'n_estimators': [100, 500, 1000, 10000] , 
              'max_depth': [5, 10, 15, 20] }

with mlflow.start_run(run_name= "RF-Grid-Search2") as run:
    # Create predictions of X_test using best model
    rf = RandomForestRegressor()
    rf_grid_model = GridSearchCV(rf, parameters, cv=3)
#     grid_rf_model = best_rf
    rf_grid_model.fit(X_train, y_train)
    predictions = rf_grid_model.predict(X_test)
  
  
  # Log model with name
    mlflow.sklearn.log_model(rf_grid_model, "grid-random-forest-model2")
  
  # Log params
    mlflow.log_params(parameters)
  
  # Create and log MSE metrics using predictions of X_test and its actual value y_test
    mse = mean_squared_error(y_test, predictions)
    mlflow.log_metrics({"mse": mse})
    
    experimentID2 = run.info.experiment_id
    runID = run.info.run_uuid
    print("Inside MLflow Run with id {}".format(runID))
    
    best_rf = rf_grid_model.best_estimator_

# COMMAND ----------

# MAGIC %md
# MAGIC Time permitting, use the `MlflowClient` to interact programatically with your run.

# COMMAND ----------

# TODO
runs = client.search_runs(experimentID2, max_results=1)
# runs[0].data.metrics
# # runs[0] = client.search_runs(experimentID)
model = mlflow.sklearn.load_model('runs:/'+runs[0].info.run_id+"/grid-random-forest-model2")
model.feature_importances_

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> See the solutions folder for an example solution to this lab.

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Classroom-Cleanup<br>
# MAGIC 
# MAGIC Run the **`Classroom-Cleanup`** cell below to remove any artifacts created by this lesson.

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Cleanup"

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Next Steps
# MAGIC 
# MAGIC Start the next lesson, [Packaging ML Projects]($../03-Packaging-ML-Projects ).

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
