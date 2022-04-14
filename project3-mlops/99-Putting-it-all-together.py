# Databricks notebook source
# MAGIC %md
# MAGIC # Putting it all together: Managing the Machine Learning Lifecycle
# MAGIC 
# MAGIC Create a workflow that includes pre-processing logic, the optimal ML algorithm and hyperparameters, and post-processing logic.
# MAGIC 
# MAGIC ## Instructions
# MAGIC 
# MAGIC In this course, we've primarily used Random Forest in `sklearn` to model the Airbnb dataset.  In this exercise, perform the following tasks:
# MAGIC <br><br>
# MAGIC 0. Create custom pre-processing logic to featurize the data
# MAGIC 0. Try a number of different algorithms and hyperparameters.  Choose the most performant solution
# MAGIC 0. Create related post-processing logic
# MAGIC 0. Package the results and execute it as its own run
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

# Adust our working directory from what DBFS sees to what python actually sees
working_path = workingDir.replace("dbfs:", "/dbfs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pre-processing
# MAGIC 
# MAGIC Take a look at the dataset and notice that there are plenty of strings and `NaN` values present. Our end goal is to train a sklearn regression model to predict the price of an airbnb listing.
# MAGIC 
# MAGIC 
# MAGIC Before we can start training, we need to pre-process our data to be compatible with sklearn models by making all features purely numerical. 

# COMMAND ----------

import pandas as pd

airbnbDF = spark.read.parquet("/mnt/training/airbnb/sf-listings/sf-listings-correct-types.parquet").toPandas()

display(airbnbDF)

# COMMAND ----------

# MAGIC %md
# MAGIC In the following cells we will walk you through the most basic pre-processing step necessary. Feel free to add additional steps afterwards to improve your model performance.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC First, convert the `price` from a string to a float since the regression model will be predicting numerical values.

# COMMAND ----------

# TODO
airbnbDF["int_price"] = airbnbDF["price"].apply(lambda s: float(s.replace("$", "").replace(",", "")))
airbnbDF_cleaned_price = airbnbDF.drop(["price"],axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC Take a look at our remaining columns with strings (or numbers) and decide if you would like to keep them as features or not.
# MAGIC 
# MAGIC Remove the features you decide not to keep.

# COMMAND ----------

# TODO
airbnbDF_cleaned_features = airbnbDF_cleaned_price.drop(["host_is_superhost", "instant_bookable","cancellation_policy"], axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC For the string columns that you've decided to keep, pick a numerical encoding for the string columns. Don't forget to deal with the `NaN` entries in those columns first.

# COMMAND ----------

# TODO
airbnbDF_cleaned_features = airbnbDF_cleaned_features[airbnbDF_cleaned_features["zipcode"] != "-- default zip code --"] # removed entry with unusual zipcode
airbnbDF_cleaned_features = airbnbDF_cleaned_features.dropna(subset=["zipcode"])

# encoded each string label into an integer
airbnbDF_cleaned_features['zipcode'] = pd.factorize(airbnbDF_cleaned_features['zipcode'])[0]
airbnbDF_cleaned_features['neighbourhood_cleansed'] = pd.factorize(airbnbDF_cleaned_features['neighbourhood_cleansed'])[0]
airbnbDF_cleaned_features['property_type'] = pd.factorize(airbnbDF_cleaned_features['property_type'])[0]
airbnbDF_cleaned_features['room_type'] = pd.factorize(airbnbDF_cleaned_features['room_type'])[0]
airbnbDF_cleaned_features['bed_type'] = pd.factorize(airbnbDF_cleaned_features['bed_type'])[0]

# COMMAND ----------

# MAGIC %md
# MAGIC Before we create a train test split, check that all your columns are numerical. Remember to drop the original string columns after creating numerical representations of them.
# MAGIC 
# MAGIC Make sure to drop the price column from the training data when doing the train test split.

# COMMAND ----------

# TODO
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(airbnbDF_cleaned_features.drop(["int_price"], axis=1), airbnbDF_cleaned_features[["int_price"]].values.ravel(), random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model
# MAGIC 
# MAGIC After cleaning our data, we can start creating our model!

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Firstly, if there are still `NaN`'s in your data, you may want to impute these values instead of dropping those entries entirely. Make sure that any further processing/imputing steps after the train test split is part of a model/pipeline that can be saved.
# MAGIC 
# MAGIC In the following cell, create and fit a single sklearn model.

# COMMAND ----------

# TODO
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# impute NaN values of data with median
columns_to_impute = ["review_scores_value", "review_scores_location", "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin", "review_scores_communication", "host_total_listings_count", "bathrooms", "beds"]
preprocessing_steps = []
for col in columns_to_impute:
  imp = SimpleImputer(missing_values=np.nan, strategy='median')
  preprocessing_steps.append((col+"Imputer", imp))

# define regression 
n_estimators=100
max_depth=5
model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth) 

# create and train pipeline
pipeline = Pipeline(preprocessing_steps+[("model", model)])
pipeline.fit(X_train, y_train) 

# COMMAND ----------

# MAGIC %md
# MAGIC Pick and calculate a regression metric for evaluating your model.

# COMMAND ----------

# TODO
import numpy as np
from sklearn.metrics import mean_squared_error

pipeline.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pipeline.predict(X_test)))
rmse

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Log your model on MLflow with the same metric you calculated above so we can compare all the different models you have tried! Make sure to also log any hyperparameters that you plan on tuning!

# COMMAND ----------

# TODO
import mlflow.sklearn

with mlflow.start_run() as run:
  mlflow.sklearn.log_model(pipeline, "model")
  mlflow.log_param("max_depth", max_depth)
  mlflow.log_param("n_estimators", n_estimators)
  mlflow.log_metric("rmse", rmse)

  experimentID = run.info.experiment_id

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Change and re-run the above 3 code cells to log different models and/or models with different hyperparameters until you are satisfied with the performance of at least 1 of them.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Look through the MLflow UI for the best model. Copy its `URI` so you can load it as a `pyfunc` model.

# COMMAND ----------

# TODO
import mlflow.pyfunc

runsDF = mlflow.search_runs(experimentID).loc[:, ["artifact_uri", "metrics.rmse"]]
best_URI = runsDF.sort_values("metrics.rmse").iloc[0,0]

best_model_path = best_URI + "/model"
best_model = mlflow.pyfunc.load_model(model_uri=best_model_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Post-processing
# MAGIC 
# MAGIC Our model currently gives us the predicted price per night for each Airbnb listing. Now we would like our model to tell us what the price per person would be for each listing, assuming the number of renters is equal to the `accommodates` value. 

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC Fill in the following model class to add in a post-processing step which will get us from total price per night to **price per person per night**.
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> Check out <a href="https://www.mlflow.org/docs/latest/models.html#id13" target="_blank">the MLFlow docs for help.</a>

# COMMAND ----------

# TODO

class Airbnb_Model(mlflow.pyfunc.PythonModel):

    def __init__(self, model):
        self.model = model

    def postprocess_result(self, model_input, results):
        return results/list(model_input["accommodates"])
    
    def predict(self, context, model_input):
        results = self.model.predict(model_input)
        return self.postprocess_result(model_input, results)

# COMMAND ----------

# MAGIC %md
# MAGIC Construct and save the model to the given `final_model_path`.

# COMMAND ----------

# TODO
final_model_path =  f"{working_path}/model"

import shutil # just in case
try: shutil.rmtree(final_model_path)
except: pass # ignore any errors

price_per_person = Airbnb_Model(best_model)
mlflow.pyfunc.save_model(path=final_model_path, python_model=price_per_person)

# COMMAND ----------

# MAGIC %md
# MAGIC Load the model in `python_function` format and apply it to our test data `X_test` to check that we are getting price per person predictions now.

# COMMAND ----------

# TODO
# Load the model
final_model = mlflow.pyfunc.load_model(final_model_path)

# Apply the model
final_model.predict(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Packaging your Model
# MAGIC 
# MAGIC Now we would like to package our completed model! 

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC First save your testing data at `test_data_path` so we can test the packaged model.
# MAGIC 
# MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** When using `.to_csv` make sure to set `index=False` so you don't end up with an extra index column in your saved dataframe.

# COMMAND ----------

# TODO
test_data_path = f"{working_path}/test_data.csv"
X_test.to_csv(test_data_path, index=False)

prediction_path = f"{working_path}/predictions.csv"

# COMMAND ----------

# MAGIC %md
# MAGIC First we will determine what the project script should do. Fill out the `model_predict` function to load out the trained model you just saved (at `final_model_path`) and make price per person predictions on the data at `test_data_path`. Then those predictions should be saved under `prediction_path` for the user to access later.
# MAGIC 
# MAGIC Run the cell to check that your function is behaving correctly and that you have predictions saved at `demo_prediction_path`.

# COMMAND ----------

# TODO
import click
import mlflow.pyfunc
import pandas as pd

@click.command()
@click.option("--final_model_path", default="", type=str)
@click.option("--test_data_path", default="", type=str)
@click.option("--prediction_path", default="", type=str)
def model_predict(final_model_path, test_data_path, prediction_path):
  final_model = mlflow.pyfunc.load_model(final_model_path)
  X_test = pd.read_csv(test_data_path)
  prediction = final_model.predict(X_test) 
  pd.DataFrame(prediction).to_csv(prediction_path, index=False)

# test model_predict function    
demo_prediction_path = f"{working_path}/predictions.csv"

from click.testing import CliRunner
runner = CliRunner()
result = runner.invoke(model_predict, ['--final_model_path', final_model_path, 
                                       '--test_data_path', test_data_path,
                                       '--prediction_path', demo_prediction_path], catch_exceptions=True)

assert result.exit_code == 0, "Code failed" # Check to see that it worked
print("Price per person predictions: ")
print(pd.read_csv(demo_prediction_path))

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we will create a MLproject file and put it under our `workingDir`. Complete the parameters and command of the file.

# COMMAND ----------

# TODO
dbutils.fs.put(f"{workingDir}/MLproject", 
'''
name: Capstone-Project
conda_env: conda.yaml
entry_points:
  main:
    parameters:
      final_model_path: {type: str, default: ""}
      test_data_path: {type: str, default: ""}
      prediction_path: {type: str, default: ""}
      stacktrace_path: {type: str, default: ""}
    command: "python predict.py --test_data_path {test_data_path} --final_model_path {final_model_path} --prediction_path {prediction_path} --stacktrace_path {stacktrace_path}"
'''.strip(), overwrite=True)

# COMMAND ----------

print(prediction_path)

# COMMAND ----------

# MAGIC %md
# MAGIC We then create a `conda.yaml` file to list the dependencies needed to run our script.
# MAGIC 
# MAGIC For simplicity, we will ensure we use the same version as we are running in this notebook.

# COMMAND ----------

import cloudpickle, numpy, pandas, sklearn, sys

version = sys.version_info # Handles possibly conflicting Python versions

file_contents = f"""
name: Capstone
channels:
  - defaults
dependencies:
  - python={version.major}.{version.minor}.{version.micro}
  - cloudpickle={cloudpickle.__version__}
  - numpy={numpy.__version__}
  - pandas={pandas.__version__}
  - scikit-learn={sklearn.__version__}
  - pip:
    - mlflow=={mlflow.__version__}
""".strip()

dbutils.fs.put(f"{workingDir}/conda.yaml", file_contents, overwrite=True)

print(file_contents)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we will put the **`predict.py`** script into our project package.
# MAGIC 
# MAGIC Complete the **`.py`** file by copying and placing the **`model_predict`** function you defined above.

# COMMAND ----------

# TODO
dbutils.fs.put(f"{workingDir}/predict.py", 
'''
import click
import mlflow.pyfunc
import pandas as pd
import traceback
@click.command()
@click.option("--final_model_path", default="", type=str)
@click.option("--test_data_path", default="", type=str)
@click.option("--prediction_path", default="", type=str)
@click.option("--stacktrace_path", default="", type=str)
def model_predict(final_model_path, test_data_path, prediction_path, stacktrace_path):
  try:
    final_model = mlflow.pyfunc.load_model(final_model_path)
    X_test = pd.read_csv(test_data_path)
    prediction = final_model.predict(X_test) 
    pd.DataFrame(prediction).to_csv(prediction_path, index = False)
    
  except:
    file = open(stacktrace_path, "w")
    traceback.print_exc(file=file)
    file.close()
    
if __name__ == "__main__":
  model_predict()
'''.strip(), overwrite=True)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's double check all the files we've created are in the `workingDir` folder. You should have at least the following 3 files:
# MAGIC * `MLproject`
# MAGIC * `conda.yaml`
# MAGIC * `predict.py`

# COMMAND ----------

display( dbutils.fs.ls(workingDir) )

# COMMAND ----------

# MAGIC %md
# MAGIC Under **`workingDir`** is your completely packaged project.
# MAGIC 
# MAGIC Run the project to use the model saved at **`final_model_path`** to predict the price per person of each Airbnb listing in **`test_data_path`** and save those predictions under **`second_prediction_path`** (defined below).

# COMMAND ----------

# TODO
second_prediction_path = f"{working_path}/predictions-2.csv"

mlflow.projects.run(working_path,
  parameters={
    "final_model_path": final_model_path,
    "test_data_path": test_data_path,
    "prediction_path": second_prediction_path,
    "stacktrace_path": f"{working_path}/stacktrace.txt"
})

try: print( dbutils.fs.head(f"{workingDir}/stacktrace.txt") )
except: print("No errors detected") # No file, then no error

# COMMAND ----------

# MAGIC %md
# MAGIC Run the following cell to check that your model's predictions are there!

# COMMAND ----------

print("Price per person predictions: ")
print(pd.read_csv(second_prediction_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Classroom-Cleanup<br>
# MAGIC 
# MAGIC Run the **`Classroom-Cleanup`** cell below to remove any artifacts created by this lesson.

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Cleanup"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC <h2><img src="https://files.training.databricks.com/images/105/logo_spark_tiny.png"> All done!</h2>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
