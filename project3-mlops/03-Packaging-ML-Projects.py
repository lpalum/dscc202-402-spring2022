# Databricks notebook source
# MAGIC %md
# MAGIC # Packaging ML Projects
# MAGIC 
# MAGIC Machine learning projects need to produce both reusable code and reproducible results.  This lesson examines creating, organizing, and packaging machine learning projects with a focus on reproducibility and collaborating with a team.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Introduce organizing code into projects
# MAGIC  - Package a basic project with parameters and an environment
# MAGIC  - Run a basic project locally and remotely using Github
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
# MAGIC ### The Case for Packaging
# MAGIC 
# MAGIC There are a number of different reasons why teams need to package their machine learning projects:<br><br>
# MAGIC 
# MAGIC 1. Projects have various library dependencies so shipping a machine learning solution involves the environment in which it was built.  MLflow allows for this environment to be a conda environment or docker container.  This means that teams can easily share and publish their code for others to use.
# MAGIC 2. Machine learning projects become increasingly complex as time goes on.  This includes ETL and featurization steps, machine learning models used for pre-processing, and finally the model training itself.
# MAGIC 3. Each component of a machine learning pipeline needs to allow for tracing its lineage.  If there's a failure at some point, tracing the full end-to-end lineage of a model allows for easier debugging.
# MAGIC 
# MAGIC **ML Projects is a specification for how to organize code in a project.**  The heart of this is an **MLproject file,** a YAML specification for the components of an ML project.  This allows for more complex workflows since a project can execute another project, allowing for encapsulation of each stage of a more complex machine learning architecture.  This means that teams can collaborate more easily using this architecture.
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/mlflow-project.png" style="height: 400px; margin: 20px"/></div>
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> See <a href="https://github.com/mlflow/mlflow-example" target="_blank">this example project backed by GitHub</a> for an example of how to integrate MLflow with source control.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Packaging a Simple Project
# MAGIC 
# MAGIC First we're going to create a simple MLflow project consisting of the following elements:<br><br>
# MAGIC 
# MAGIC 1. MLProject file
# MAGIC 2. Conda environment
# MAGIC 3. Basic machine learning script
# MAGIC 
# MAGIC We're going to want to be able to pass parameters into this code so that we can try different hyperparameter options.

# COMMAND ----------

# MAGIC %md
# MAGIC Create a new experiment for this exercise.  Navigate to the UI in another tab.

# COMMAND ----------

experimentPath = "/Users/" + username + "/experiment-SPL3"

# COMMAND ----------

import mlflow

mlflow.set_experiment(experimentPath)
mlflow_client = mlflow.tracking.MlflowClient()
experimentID = mlflow_client.get_experiment_by_name(name=experimentPath).experiment_id

print(f"The experiment can be found at the path `{experimentPath}` and has an experiment_id of `{experimentID}`")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC First, examine the code we're going to run.  This looks similar to what we ran in the last lesson with the addition of decorators from the `click` library.  This allows us to parameterize our code.
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> We'll uncomment out the `__main__` block when we save this code as a Python file.
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> Check out the <a href="https://click.palletsprojects.com/en/7.x/" target="_blank">`click` docs here.</a>

# COMMAND ----------

import click
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

@click.command()
@click.option("--data_path", default="/dbfs/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.csv", type=str)
@click.option("--n_estimators", default=10, type=int)
@click.option("--max_depth", default=20, type=int)
@click.option("--max_features", default="auto", type=str)
def mlflow_rf(data_path, n_estimators, max_depth, max_features):

  with mlflow.start_run() as run:
    # Import the data
    df = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df[["price"]].values.ravel(), random_state=42)
    
    # Create model, train it, and create predictions
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)

    # Log model
    mlflow.sklearn.log_model(rf, "random-forest-model")
    
    # Log params
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("max_features", max_features)

    # Log metrics
    mlflow.log_metric("mse", mean_squared_error(y_test, predictions))
    mlflow.log_metric("mae", mean_absolute_error(y_test, predictions))  
    mlflow.log_metric("r2", r2_score(y_test, predictions))  

# if __name__ == "__main__":
#   mlflow_rf() # Note that this does not need arguments thanks to click

# COMMAND ----------

# MAGIC %md
# MAGIC Test that it works using the `click` `CliRunner`, which will execute the code in the same way we expect to.

# COMMAND ----------

from click.testing import CliRunner

runner = CliRunner()
result = runner.invoke(mlflow_rf, ['--n_estimators', 10, '--max_depth', 20], catch_exceptions=True)

assert result.exit_code == 0, "Code failed" # Check to see that it worked

print("Success!")

# COMMAND ----------

# MAGIC %md
# MAGIC Now create a directory to hold our project files.  This will be a unique directory for this lesson.

# COMMAND ----------

# Adust our working directory from what DBFS sees to what python actually sees
working_path = workingDir.replace("dbfs:", "/dbfs")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC Create the `MLproject` file.  This is the heart of an MLflow project.  It includes pointers to the conda environment and a `main` entry point, which is backed by the file `train.py`.
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> Any `.py` or `.sh` file can be an entry point.

# COMMAND ----------

dbutils.fs.put(f"{workingDir}/MLproject", 
'''
name: Lesson-3-Model-Training

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      data_path: {type: str, default: "/dbfs/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.csv"}
      n_estimators: {type: int, default: 10}
      max_depth: {type: int, default: 20}
      max_features: {type: str, default: "auto"}
    command: "python train.py --data_path {data_path} --n_estimators {n_estimators} --max_depth {max_depth} --max_features {max_features}"
'''.strip())

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC Create the conda environment.
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> You can also dynamically view and use a package version by calling `.__version__` on the package.

# COMMAND ----------

import cloudpickle, numpy, pandas, sklearn

file_contents = f"""
name: Lesson-03
channels:
  - defaults
dependencies:
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
# MAGIC Now create the code itself.  This is the same as above except for with the `__main__` is included.  Note how there are no arguments passed into `mlflow_rf()` on the final line.  `click` is handling the arguments for us.

# COMMAND ----------

dbutils.fs.put(f"{workingDir}/train.py", 
'''
import click
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

@click.command()
@click.option("--data_path", default="/dbfs/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.csv", type=str)
@click.option("--n_estimators", default=10, type=int)
@click.option("--max_depth", default=20, type=int)
@click.option("--max_features", default="auto", type=str)
def mlflow_rf(data_path, n_estimators, max_depth, max_features):

  with mlflow.start_run() as run:
    # Import the data
    df = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df[["price"]].values.ravel(), random_state=42)
    
    # Create model, train it, and create predictions
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)

    # Log model
    mlflow.sklearn.log_model(rf, "random-forest-model")
    
    # Log params
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("max_features", max_features)

    # Log metrics
    mlflow.log_metric("mse", mean_squared_error(y_test, predictions))
    mlflow.log_metric("mae", mean_absolute_error(y_test, predictions))  
    mlflow.log_metric("r2", r2_score(y_test, predictions))  

if __name__ == "__main__":
  mlflow_rf() # Note that this does not need arguments thanks to click
'''.strip(), True)

# COMMAND ----------

# MAGIC %md
# MAGIC To summarize, you now have three files: `MLproject`, `conda.yaml`, and `train.py`

# COMMAND ----------

display( dbutils.fs.ls(workingDir) )

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Running Projects
# MAGIC 
# MAGIC Now you have the three files we need to run the project, we can trigger the run.  We'll do this in a few different ways:<br><br>
# MAGIC 
# MAGIC 1. On the driver node of our Spark cluster
# MAGIC 2. On a new Spark cluster submitted as a job
# MAGIC 3. Using files backed by GitHub
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> This currently relies on environment variables.  [See the setup script for details.]($./Includes/MLflow)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC Now run the experiment.  This command will execute against the driver node of a Spark cluster, though it could be running locally or on a different remote VM.
# MAGIC 
# MAGIC First set the experiment using the `experimentPath` defined earlier.  Prepend `/dbfs` to the file path, which allows the cluster's file system to access DBFS.  Then, pass your parameters.
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> This will take a few minutes to build the environment for the first time.  Subsequent runs are faster since `mlflow` can reuse the same environment after it has been built.

# COMMAND ----------

import mlflow

mlflow.projects.run(uri=working_path,
  parameters={
    "data_path": "/dbfs/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.csv",
    "n_estimators": 10,
    "max_depth": 20,
    "max_features": "auto"
})

# COMMAND ----------

# MAGIC %md
# MAGIC Check the run in the UI.  Notice that you can see the run command.  **This is very helpful in debugging.**
# MAGIC 
# MAGIC Now that it's working, experiment with other parameters.  

# COMMAND ----------

mlflow.projects.run(working_path,
  parameters={
    "data_path": "/dbfs/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.csv",
    "n_estimators": 1000,
    "max_depth": 10,
    "max_features": "log2"
})

# COMMAND ----------

# MAGIC %md
# MAGIC How did the new model do?

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC Now try executing this code against a new Databricks cluster.  This needs cluster specifications in order for Databricks to know what kind of cluster to use.  Uncomment the following to run it.
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/>  <a href="https://docs.databricks.com/api/latest/clusters.html" target="_blank">See the clusters API docs</a> to see how to define cluster specifications.

# COMMAND ----------

# clusterspecs = {
#     "num_workers": 2,
#     "spark_version": "5.2.x-scala2.11",
#     "spark_conf": {},
#     "aws_attributes": {
#         "first_on_demand": 1,
#         "availability": "SPOT_WITH_FALLBACK",
#         "zone_id": "us-west-1c",
#         "spot_bid_price_percent": 100,
#         "ebs_volume_count": 0
#     },
#     "node_type_id": "i3.xlarge",
#     "driver_node_type_id": "i3.xlarge"
#   }

# mlflow.projects.run(
#   uri=working_path,
#   parameters={
#     "data_path": "/dbfs/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.csv",
#     "n_estimators": 1500,
#     "max_depth": 5,
#     "max_features": "sqrt"
# },
#   backend="databricks",
#   backend_config=clusterspecs
# )

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, run this example, which is <a href="https://github.com/mlflow/mlflow-example" target="_blank">a project backed by GitHub.</a>

# COMMAND ----------

mlflow.run(
  uri="https://github.com/mlflow/mlflow-example",
  parameters={'alpha':0.4}
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Review
# MAGIC 
# MAGIC **Question:** Why is packaging important?  
# MAGIC **Answer:** Packaging not only manages your code but the environment in which it was run.  This environment can be a Conda or Docker environment.  This ensures that you have reproducible code and models that can be used in a number of downstream environments.
# MAGIC 
# MAGIC **Question:** What are the core components of MLflow projects?  
# MAGIC **Answer:** An MLmodel specifies the project components using YAML.  The environment file contains specifics about the environment.  The code itself contains the steps to create a model or process data.
# MAGIC 
# MAGIC **Question:** What code can I run and where can I run it?  
# MAGIC **Answer:** Arbitrary code can be run in any number of different languages.  It can be run locally or remotely, whether on a remote VM, Spark cluster, or submitted as a Databricks job.

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
# MAGIC Start the labs for this lesson, [Packaging ML Projects Lab]($./Labs/03-Lab) 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Additional Topics & Resources
# MAGIC 
# MAGIC **Q:** Where can I find out more information on MLflow Projects?  
# MAGIC **A:** Check out the <a href="https://www.mlflow.org/docs/latest/projects.html" target="_blank">MLflow docs</a>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
