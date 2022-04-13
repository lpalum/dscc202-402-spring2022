# Databricks notebook source
# MAGIC %md
# MAGIC # Model Management
# MAGIC 
# MAGIC An MLflow model is a standard format for packaging models that can be used on a variety of downstream tools.  This lesson provides a generalizable way of handling machine learning models created in and deployed to a variety of environments.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Introduce model management best practices
# MAGIC  - Store and use different flavors of models for different deployment environments
# MAGIC  - Apply models combined with arbitrary pre and post-processing code using Python models
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

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Managing Machine Learning Models
# MAGIC 
# MAGIC Once a model has been trained and bundled with the environment it was trained in, the next step is to package the model so that it can be used by a variety of serving tools.  The current deployment options include Docker-based REST servers, Spark using streaming or batch, and cloud platforms such as Azure ML and AWS SageMaker.  Packaging the final model in a platform-agnostic way offers the most flexibility in deployment options and allows for model reuse across a number of platforms.
# MAGIC 
# MAGIC **MLflow models is a tool for deploying models that's agnostic to both the framework the model was trained in and the environment it's being deployed to.  It's convention for packaging machine learning models that offers self-contained code, environments, and models.**  The main abstraction in this package is the concept of **flavors,** which are different ways the model can be used.  For instance, a TensorFlow model can be loaded as a TensorFlow DAG or as a Python function: using the MLflow model convention allows for the model to be used regardless of the library that was used to train it originally.
# MAGIC 
# MAGIC The primary difference between MLflow projects and models is that models are geared more towards inference and serving.  The `python_function` flavor of models gives a generic way of bundling models regardless of whether it was `sklearn`, `keras`, or any other machine learning library that trained the model.  We can thereby deploy a python function without worrying about the underlying format of the model.  **MLflow therefore maps any training framework to any deployment environment**, massively reducing the complexity of inference.
# MAGIC 
# MAGIC Finally, arbitrary pre and post-processing steps can be included in the pipeline such as data loading, cleansing, and featurization.  This means that the full pipeline, not just the model, can be preserved.
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/mlflow-models-enviornments.png" style="height: 400px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Model Flavors
# MAGIC 
# MAGIC Flavors offer a way of saving models in a way that's agnostic to the training development, making it significantly easier to be used in various deployment options.  Some of the most popular built-in flavors include the following:<br><br>
# MAGIC 
# MAGIC * <a href="https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#module-mlflow.pyfunc" target="_blank">mlflow.pyfunc</a>
# MAGIC * <a href="https://mlflow.org/docs/latest/python_api/mlflow.keras.html#module-mlflow.keras" target="_blank">mlflow.keras</a>
# MAGIC * <a href="https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html#module-mlflow.pytorch" target="_blank">mlflow.pytorch</a>
# MAGIC * <a href="https://mlflow.org/docs/latest/python_api/mlflow.sklearn.html#module-mlflow.sklearn" target="_blank">mlflow.sklearn</a>
# MAGIC * <a href="https://mlflow.org/docs/latest/python_api/mlflow.spark.html#module-mlflow.spark" target="_blank">mlflow.spark</a>
# MAGIC * <a href="https://mlflow.org/docs/latest/python_api/mlflow.tensorflow.html#module-mlflow.tensorflow" target="_blank">mlflow.tensorflow</a>
# MAGIC 
# MAGIC Models also offer reproducibility since the run ID and the timestamp of the run are preserved as well.  
# MAGIC 
# MAGIC <a href="https://mlflow.org/docs/latest/python_api/index.html" target="_blank">You can see all of the flavors and modules here.</a>
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/mlflow-models.png" style="height: 400px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md
# MAGIC To demonstrate the power of model flavors, let's first create two models using different frameworks.
# MAGIC 
# MAGIC Import the data.

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("/dbfs/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.csv")
X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df[["price"]].values.ravel(), random_state=42)

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC Train a random forest model.

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

rf = RandomForestRegressor(n_estimators=100, max_depth=5)
rf.fit(X_train, y_train)

rf_mse = mean_squared_error(y_test, rf.predict(X_test))

rf_mse

# COMMAND ----------

# MAGIC %md
# MAGIC Train a neural network.

# COMMAND ----------

import tensorflow as tf
tf.random.set_seed(42) # For reproducibility

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

nn = Sequential([
  Dense(40, input_dim=21, activation='relu'),
  Dense(20, activation='relu'),
  Dense(1, activation='linear')
])

nn.compile(optimizer="adam", loss="mse")
nn.fit(X_train, y_train, validation_split=.2, epochs=40, verbose=2)

# nn.evaluate(X_test, y_test)
nn_mse = mean_squared_error(y_test, nn.predict(X_test))

nn_mse

# COMMAND ----------

# MAGIC %md
# MAGIC Now log the two models.

# COMMAND ----------

import mlflow.sklearn

with mlflow.start_run(run_name="RF Model") as run:
  mlflow.sklearn.log_model(rf, "model")
  mlflow.log_metric("mse", rf_mse)

  sklearnRunID = run.info.run_uuid
  sklearnURI = run.info.artifact_uri
  
  experimentID = run.info.experiment_id

# COMMAND ----------

import mlflow.keras

with mlflow.start_run(run_name="NN Model") as run:
  mlflow.keras.log_model(nn, "model")
  mlflow.log_metric("mse", nn_mse)

  kerasRunID = run.info.run_uuid
  kerasURI = run.info.artifact_uri

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can use both of these models in the same way, even though they were trained by different packages.

# COMMAND ----------

import mlflow.pyfunc

rf_pyfunc_model = mlflow.pyfunc.load_model(model_uri="runs:/"+sklearnRunID+"/model")
type(rf_pyfunc_model)

# COMMAND ----------

import mlflow.pyfunc

nn_pyfunc_model = mlflow.pyfunc.mlflow.pyfunc.load_model(model_uri="runs:/"+kerasRunID+"/model")
type(nn_pyfunc_model)

# COMMAND ----------

# MAGIC %md
# MAGIC Both will implement a predict method.  The `sklearn` model is still of type `sklearn` because this package natively implements this method.

# COMMAND ----------

rf_pyfunc_model.predict(X_test)

# COMMAND ----------

nn_pyfunc_model.predict(X_test)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Pre and Post Processing Code using `pyfunc`
# MAGIC 
# MAGIC A `pyfunc` is a generic python model that can define any model, regardless of the libraries used to train it.  As such, it's defined as a directory structure with all of the dependencies.  It is then "just an object" with a predict method.  Since it makes very few assumptions, it can be deployed using MLflow, SageMaker, a Spark UDF or in any other environment.
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> Check out <a href="https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#pyfunc-create-custom" target="_blank">the `pyfunc` documentation for details</a><br>
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> Check out <a href="https://github.com/mlflow/mlflow/blob/master/docs/source/models.rst#example-saving-an-xgboost-model-in-mlflow-format" target="_blank">this README for generic example code and integration with `XGBoost`</a>

# COMMAND ----------

# MAGIC %md
# MAGIC To demonstrate how `pyfunc` works, create a basic class that adds `n` to the input values.
# MAGIC 
# MAGIC Define a model class.

# COMMAND ----------

import mlflow.pyfunc

class AddN(mlflow.pyfunc.PythonModel):

    def __init__(self, n):
        self.n = n

    def predict(self, context, model_input):
        return model_input.apply(lambda column: column + self.n)

# COMMAND ----------

# MAGIC %md
# MAGIC Construct and save the model.

# COMMAND ----------

from mlflow.exceptions import MlflowException

model_path = f"{workingDir}/add_n_model2"
add5_model = AddN(n=5)

dbutils.fs.rm(model_path, True) # Allows you to rerun the code multiple times

mlflow.pyfunc.save_model(path=model_path.replace("dbfs:", "/dbfs"), python_model=add5_model)

# COMMAND ----------

# MAGIC %md
# MAGIC Load the model in `python_function` format.

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model(model_path)

# COMMAND ----------

# MAGIC %md
# MAGIC Evaluate the model.

# COMMAND ----------

import pandas as pd

model_input = pd.DataFrame([range(10)])
model_output = loaded_model.predict(model_input)

assert model_output.equals(pd.DataFrame([range(5, 15)]))

model_output

# COMMAND ----------

# MAGIC %md
# MAGIC ## Review
# MAGIC **Question:** How do MLflow projects differ from models?  
# MAGIC **Answer:** The focus of MLflow projects is reproducibility of runs and packaging of code.  MLflow models focuses on various deployment environments.
# MAGIC 
# MAGIC **Question:** What is a ML model flavor?  
# MAGIC **Answer:** Flavors are a convention that deployment tools can use to understand the model, which makes it possible to write tools that work with models from any ML library without having to integrate each tool with each library.  Instead of having to map each training environment to a deployment environment, ML model flavors manages this mapping for you.
# MAGIC 
# MAGIC **Question:** How do I add pre and post processing logic to my models?  
# MAGIC **Answer:** A model class that extends `mlflow.pyfunc.PythonModel` allows you to have load, pre-processing, and post-processing logic.

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Classroom-Cleanup<br>
# MAGIC 
# MAGIC Run the **`Classroom-Cleanup`** cell below to remove any artifacts created by this lesson.

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Cleanup"

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Next Steps
# MAGIC 
# MAGIC Start the labs for this lesson, [Model Management Lab]($./Labs/05-Lab) 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Additional Topics & Resources
# MAGIC 
# MAGIC **Q:** Where can I find out more information on MLflow Models?  
# MAGIC **A:** Check out <a href="https://www.mlflow.org/docs/latest/models.html" target="_blank">the MLflow documentation</a>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
