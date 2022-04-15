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
import numpy as np
airbnbDF["price"] = airbnbDF["price"].str.replace("$", "", regex=True)
airbnbDF["price"] = airbnbDF["price"].str.replace(",", "", regex=True)
airbnbDF["price"] = airbnbDF.price.astype('float32')
print(airbnbDF["price"])

# airbnbDF["price"] = airbnbDF["price"].str.replace('$', '')



# COMMAND ----------

# MAGIC %md
# MAGIC Take a look at our remaining columns with strings (or numbers) and decide if you would like to keep them as features or not.
# MAGIC 
# MAGIC Remove the features you decide not to keep.

# COMMAND ----------

# TODO

airbnbDF["trunc_lat"] = airbnbDF.latitude.round(decimals=2)
airbnbDF["trunc_long"] = airbnbDF.longitude.round(decimals=2)
airbnbDF["review_scores_sum"] = airbnbDF[['review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value']].mean(axis=1)
airbnbDF = airbnbDF.drop(["latitude", "longitude", 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', "neighbourhood_cleansed", "property_type", "zipcode"], axis=1)


# COMMAND ----------

airbnbDF.columns

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC For the string columns that you've decided to keep, pick a numerical encoding for the string columns. Don't forget to deal with the `NaN` entries in those columns first.

# COMMAND ----------

# TODO
from sklearn.impute import SimpleImputer

airbnbDF["host_is_superhost"] = airbnbDF["host_is_superhost"].str.replace("t", "0", regex=True)
airbnbDF["host_is_superhost"] = airbnbDF["host_is_superhost"].str.replace("f", "1", regex=True)
airbnbDF["instant_bookable"] = airbnbDF["instant_bookable"].str.replace("t", "0", regex=True)
airbnbDF["instant_bookable"] = airbnbDF["instant_bookable"].str.replace("f", "1", regex=True)
# airbnbDF["host_is_superhost"] = airbnbDF.host_is_superhost.astype(int)
# airbnbDF["instant_bookable"] = airbnbDF["instant_bookable"].astype(int)
airbnbDF["host_is_superhost"] = pd.to_numeric(airbnbDF["host_is_superhost"])
airbnbDF["instant_bookable"] = pd.to_numeric(airbnbDF["instant_bookable"])
airbnbDF["bed_type"] = np.where(airbnbDF["bed_type"] == "Real Bed", 0, 1)
airbnbDF["room_type"] = np.where(airbnbDF["room_type"] == "Entire home/apt", 0, 1)
airbnbDF["cancellation_policy"] = airbnbDF["cancellation_policy"].str.replace("flexible", "0", regex=True)
airbnbDF["cancellation_policy"] = airbnbDF["cancellation_policy"].str.replace("moderate", "1", regex=True)
airbnbDF["cancellation_policy"] = airbnbDF["cancellation_policy"].str.replace("super_strict_30", "3", regex=True)
airbnbDF["cancellation_policy"] = airbnbDF["cancellation_policy"].str.replace("super_strict_60", "3", regex=True)
airbnbDF["cancellation_policy"] = airbnbDF["cancellation_policy"].str.replace("strict", "2", regex=True)
# airbnbDF["cancellation_policy"] = airbnbDF["cancellation_policy"].astype(int)
airbnbDF["cancellation_policy"] = pd.to_numeric(airbnbDF["cancellation_policy"])
# airbnbDF["zipcode"] = airbnbDF["zipcode"].replace("-- default zip code --", np.nan, regex=True)
airbnbDF = airbnbDF.apply(pd.to_numeric)
impute = SimpleImputer(missing_values=np.nan, strategy='mean')
# airbnbDF = impute.fit(airbnbDF)
airbnbDF = impute.fit_transform(airbnbDF)
airbnbDF = pd.DataFrame(airbnbDF, columns=['host_is_superhost', 'cancellation_policy', 'instant_bookable',
       'host_total_listings_count', 'room_type', 'accommodates', 'bathrooms',
       'bedrooms', 'beds', 'bed_type', 'minimum_nights', 'number_of_reviews',
       'review_scores_rating', 'price', 'trunc_lat', 'trunc_long',
       'review_scores_sum'])
print(airbnbDF.head())
# airbnbDF.fillna(-1)

# COMMAND ----------

print(type(airbnbDF["price"][1]))
print(max(airbnbDF["price"]))
print(min(airbnbDF["price"]))

# COMMAND ----------

# MAGIC %md
# MAGIC Before we create a train test split, check that all your columns are numerical. Remember to drop the original string columns after creating numerical representations of them.
# MAGIC 
# MAGIC Make sure to drop the price column from the training data when doing the train test split.

# COMMAND ----------

# TODO
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(airbnbDF.drop(["price"], axis=1), airbnbDF[["price"]].values.ravel(), random_state=42)

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
from sklearn.ensemble import RandomForestRegressor

rfmodel = RandomForestRegressor(n_estimators=100, max_depth=25)
rfmodel.fit(X_train, y_train)

# class RF_with_preprocess(mlflow.pyfunc.PythonModel):

#     def __init__(self, trained_rf):
#         self.rf = trained_rf

#     def preprocess_X(model_input):
#         model_input = model_input.fillNA(value=-1)
#         model_input["trunc_lat"] = model_input.latitude.round(decimals=2)
#         model_input["trunc_long"] = model_input.longitude.round(decimals=2)
#         model_input["review_scores_sum"] = model_input[['review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value']].mean(axis=1)
#         model_input = model_input.drop(["latitude", "longitude", 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', "neighbourhood_cleansed", "property_type"], axis=1)
#         model_input["host_is_superhost"] = model_input["host_is_superhost"].str.replace("t", "0", regex=True)
#         model_input["host_is_superhost"] = model_input["host_is_superhost"].str.replace("f", "1", regex=True)
#         model_input["instant_bookable"] = model_input["instant_bookable"].str.replace("t", "0", regex=True)
#         model_input["instant_bookable"] = model_input["instant_bookable"].str.replace("f", "1", regex=True)
#         model_input["host_is_superhost"] = pd.to_numeric(model_input["host_is_superhost"])
#         model_input["instant_bookable"] = pd.to_numeric(model_input["instant_bookable"])
#         model_input["bed_type"] = np.where(model_input["bed_type"] == "Real Bed", 0, 1)
#         model_input["room_type"] = np.where(model_input["room_type"] == "Entire home/apt", 0, 1)
#         model_input["cancellation_policy"] = model_input["cancellation_policy"].str.replace("flexible", "0", regex=True)
#         model_input["cancellation_policy"] = model_input["cancellation_policy"].str.replace("moderate", "1", regex=True)
#         model_input["cancellation_policy"] = model_input["cancellation_policy"].str.replace("super_strict_30", "3", regex=True)
#         model_input["cancellation_policy"] = model_input["cancellation_policy"].str.replace("super_strict_60", "3", regex=True)
#         model_input["cancellation_policy"] = model_input["cancellation_policy"].str.replace("strict", "2", regex=True)
#         model_input["cancellation_policy"] = pd.to_numeric(model_input["cancellation_policy"])
#         airbnbDF["zipcode"] = airbnbDF["zipcode"].str.replace("-- default zip code --", "0", regex=True)
#         airbnbDF = airbnbDF.apply(pd.to_numeric)
#         return

#     def preprocess_y(model_input):
#         model_input["price"] = model_input["price"].str.replace("$", "", regex=True)
#         model_input["price"] = model_input["price"].str.replace(",", "", regex=True)
#         model_input["price"] = model_input.price.astype(float)
#         return

    

# preprocess_X(X_train)
# preprocess_X(X_test)
# preprocess_y(y_train)
# preprocess_y(y_test)



# COMMAND ----------

# MAGIC %md
# MAGIC Pick and calculate a regression metric for evaluating your model.

# COMMAND ----------

# TODO
# done below with model
from sklearn.metrics import mean_squared_error


rf_mse = mean_squared_error(y_test, rfmodel.predict(X_test))

rf_mse

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Log your model on MLflow with the same metric you calculated above so we can compare all the different models you have tried! Make sure to also log any hyperparameters that you plan on tuning!

# COMMAND ----------

# TODO
import mlflow.sklearn
params = {
  "n_estimators": 100,
  "max_depth": 30,
  "random_state": 42}

# parameters = {'n_estimators': [10, 100, 1000] , 
#               'max_depth': [5, 10, 25, 50] }

with mlflow.start_run(run_name="RF Model") as run:
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
#     grid_rf_model = GridSearchCV(rf, parameters, cv=3)
#     grid_rf_model.fit(X_train, y_train)
#     best_rf = grid_rf_model.best_estimator_
    
    mlflow.sklearn.log_model(rf, "random-forest-model")
    rf_mse = mean_squared_error(y_test, rfmodel.predict(X_test))
    mlflow.log_metric("mse", rf_mse)
    mlflow.log_params(params)

    experimentID = run.info.experiment_id
    artifactURI = mlflow.get_artifact_uri()

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
from  mlflow.tracking import MlflowClient

client = MlflowClient()

runs = client.search_runs(experimentID, order_by=["metrics.mse asc"], max_results=1)
# for i in range(len(runs.data.metrics)):
    
# print(runs[0].data.metrics)
artifactURI = 'runs:/'+runs[0].info.run_id+"/random-forest-model"

model = mlflow.sklearn.load_model(artifactURI)
model.feature_importances_

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
    
    def predict(self, context, model_input):
        prediction = self.model.predict(model_input.copy())
        perperson = [0]*len(prediction)
        acc = (model_input['accommodates'].iloc[:].copy()).tolist()
        for i in range(len(prediction)):
            perperson[i] = prediction[i]/(acc[i])
        return perperson


# COMMAND ----------

# MAGIC %md
# MAGIC Construct and save the model to the given `final_model_path`.

# COMMAND ----------

# TODO
final_model_path =  f"{working_path}/final-model"

# FILL_IN
dbutils.fs.rm(final_model_path, True) # remove folder if already exists

rf_model = Airbnb_Model(rf)
mlflow.pyfunc.save_model(path=final_model_path.replace("dbfs:", "/dbfs"), python_model=rf_model)

# COMMAND ----------

# MAGIC %md
# MAGIC Load the model in `python_function` format and apply it to our test data `X_test` to check that we are getting price per person predictions now.

# COMMAND ----------

# print(X_test.accommodates)
# print(X_test.head())

# COMMAND ----------

# TODO
loaded_model = mlflow.pyfunc.load_pyfunc(final_model_path.replace("dbfs:", "/dbfs"))
loaded_model.predict(X_test)

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
# save the testing data 
from pathlib import Path

test_data_path = f"{working_path}/test_data.csv"

def df2csv(a_dataframe, data_path):
    filepath = Path(data_path)  
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    a_dataframe.to_csv(data_path, index=False) 
df2csv(X_test, test_data_path)

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
    # FILL_IN
    with mlflow.start_run() as run:
    # Import the data
        df = pd.read_csv(test_data_path)
        # load model, train it, and create predictions
        loaded_model = mlflow.pyfunc.load_pyfunc(final_model_path.replace("dbfs:", "/dbfs"))
        predictions = loaded_model.predict(df)
        predictions = np.asarray(predictions)
        pd.DataFrame(predictions).to_csv(prediction_path, header=None, index=False)
#         np.savetxt(prediction_path, predictions, delimiter=",")

        # Log model
        mlflow.sklearn.log_model(loaded_model, "random-forest-model")



# test model_predict function    
demo_prediction_path = f"{working_path}/predictions.csv"


from click.testing import CliRunner
runner = CliRunner()
result = runner.invoke(model_predict, ['--final_model_path', final_model_path, 
                                       '--test_data_path', test_data_path,
                                       '--prediction_path', demo_prediction_path], catch_exceptions=True)

print(result.exception)
assert result.exit_code == 0, "Code failed" # Check to see that it worked
print("Price per person predictions: ")
print(pd.read_csv(demo_prediction_path, header=None))

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
      final_model_path: {type: str, default:""}
      test_data_path: {type: str, default:""}
      prediction_path: {type: str, default:""}
    command:  "python predict.py --final_model_path {final_model_path} --test_data_path {test_data_path} --prediction_path {prediction_path}"
    
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
import numpy as np


@click.command()
@click.option("--final_model_path", default="", type=str)
@click.option("--test_data_path", default="", type=str)
@click.option("--prediction_path", default="", type=str)
def model_predict(final_model_path, test_data_path, prediction_path):
    # FILL_IN
    with mlflow.start_run() as run:
    # Import the data
        df = pd.read_csv(test_data_path)
        # load model, and create predictions
        loaded_model = mlflow.pyfunc.load_pyfunc(final_model_path.replace("dbfs:", "/dbfs"))
        predictions = loaded_model.predict(df)
        predictions = np.asarray(predictions)
        pd.DataFrame(predictions).to_csv(prediction_path, header=None, index=False)
#         np.savetxt(prediction_path, predictions, delimiter=",")

        # Log model
        mlflow.sklearn.log_model(loaded_model, "random-forest-model")




    
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
    "prediction_path": second_prediction_path}
)

# COMMAND ----------

# MAGIC %md
# MAGIC Run the following cell to check that your model's predictions are there!

# COMMAND ----------

print("Price per person predictions: ")
print(pd.read_csv(second_prediction_path, header=None))

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
