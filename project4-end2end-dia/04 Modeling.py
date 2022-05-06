# Databricks notebook source
# MAGIC %md
# MAGIC ## Rubric for this module
# MAGIC - Using the silver delta table(s) that were setup by your ETL module train and validate your token recommendation engine. Split, Fit, Score, Save
# MAGIC - Log all experiments using mlflow
# MAGIC - capture model parameters, signature, training/test metrics and artifacts
# MAGIC - Tune hyperparameters using an appropriate scaling mechanism for spark.  [Hyperopt/Spark Trials ](https://docs.databricks.com/_static/notebooks/hyperopt-spark-ml.html)
# MAGIC - Register your best model from the training run at **Staging**.

# COMMAND ----------

# MAGIC %run ./includes/utilities

# COMMAND ----------

# MAGIC %run ./includes/configuration

# COMMAND ----------

  # Grab the global variables
wallet_address,start_date, fractional = Utils.create_widgets()
print(wallet_address,start_date, fractional)


# COMMAND ----------

# MAGIC %md 
# MAGIC ## Your Code starts here...

# COMMAND ----------

display(dbutils.fs.ls(BASE_DELTA_PATH))

# COMMAND ----------

dataPath = '/mnt/dscc202-datasets/misc/G08/tokenrec/tables/silver/'

# COMMAND ----------

from pyspark.sql import DataFrame
from pyspark.sql.types import *
from pyspark.sql import functions as F
from delta.tables import *
import random

import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from pyspark.sql.functions import col
# configureation of the spark environment appropriate for my cluster
spark.conf.set("spark.sql.shuffle.partitions", "16")  # Configure the size of shuffles the same as core count
spark.conf.set("spark.sql.adaptive.enabled", "true")  # Spark 3.0 AQE - coalescing post-shuffle partitions, converting sort-merge join to broadcast join, and skew join optimization


class token_recommender():
    """
    Build out the required plumbing and electrical work:
    - Get the delta lake version of the training data to be stored with the model (bread crumbs)
    - Set (create) the experiment name
    - Load and create training test splits
    - instantiate an alternate least squares estimator and its associated evaluation and tuning grid
    - create a cross validation configuration for the model training
    - data_fraction is how many fraction of the ~20 million data we want to use for this run, for optimizing time
    """
    def __init__(self, dataPath: str, modelName: str, data_fraction: int)->None:
#     def __init__(self, modelName: str, data_fraction: int)->None:
        self.dataPath = dataPath
        self.modelName = modelName
        self.data_fraction = data_fraction
        

        # create an MLflow experiment for this model
        MY_EXPERIMENT = "/Users/" + dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get() + "/" + self.modelName + "-experiment/"
        mlflow.set_experiment(MY_EXPERIMENT)

        # split the data set into train, validation and test anc cache them
        # We'll hold out 60% for training, 20% of our data for validation, and leave 20% for testing
        #self.df_with_int_ids = spark.sql("select * from g08_db.notcoldstart").sample(False, data_fraction, 42)
        self.df_with_int_ids = spark.read.format('delta').load(self.dataPath).sample(False, data_fraction, 42)
        self.training_data_version = DeltaTable.forPath(spark, self.dataPath).history().head(1)[0]['version']

        seed = 42
        (split_60_df, split_a_20_df, split_b_20_df) = self.df_with_int_ids.randomSplit([0.6, 0.2, 0.2], seed = seed)
        # Let's cache these datasets for performance
        self.training_df = split_60_df.cache()
        self.validation_df = split_a_20_df.cache()
        self.test_df = split_b_20_df.cache()

        spark.sql("drop table if exists g08_db.test_data")
        self.test_df.write.format('delta').saveAsTable('g08_db.test_data')

        # Initialize our ALS learner
        als = ALS()

        # Now set the parameters for the method
        als.setMaxIter(5)\
           .setSeed(seed)\
           .setItemCol("new_token_id")\
           .setRatingCol("count")\
           .setUserCol("new_wallet_id")\
           .setColdStartStrategy("drop")

        # Now let's compute an evaluation metric for our test dataset
        # We Create an RMSE evaluator using the label and predicted columns
        self.reg_eval = RegressionEvaluator(predictionCol="prediction", labelCol="count", metricName="rmse")

        # Setup an ALS hyperparameter tuning grid search
        grid = ParamGridBuilder() \
          .addGrid(als.maxIter, [5, 10, 15]) \
          .addGrid(als.regParam, [0.15, 0.2, 0.25]) \
          .addGrid(als.rank, [12, 16, 24, 50]) \
          .build()

        # Create a cross validator, using the pipeline, evaluator, and parameter grid you created in previous steps.
        self.cv = CrossValidator(estimator=als, evaluator=self.reg_eval, estimatorParamMaps=grid, numFolds=3)

    def train(self):
        # setup the schema for the model
        input_schema = Schema([
          ColSpec("integer", "new_token_id"),
          ColSpec("integer", "new_wallet_id"),
        ])
        output_schema = Schema([ColSpec("double")])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        with mlflow.start_run(run_name=self.modelName+"-run") as run:
#           mlflow.log_params({"user_rating_training_data_version": self.training_data_version})

          # Run the cross validation on the training dataset. The cv.fit() call returns the best model it found.
          cvModel = self.cv.fit(self.training_df)

          # Evaluate the best model's performance on the validation dataset and log the result.
          validation_metric = self.reg_eval.evaluate(cvModel.transform(self.validation_df))

          mlflow.log_metric('test_' + self.reg_eval.getMetricName(), validation_metric) 

          # Log the best model.
          mlflow.spark.log_model(spark_model=cvModel.bestModel, signature = signature,
                                 artifact_path='als-model', registered_model_name=self.modelName)

        """
        - Capture the latest model version
        - archive any previous Staged version
        - Transition this version to Staging
        """
        client = MlflowClient()
        model_versions = []

        # Transition this model to staging and archive the current staging model if there is one
        for mv in client.search_model_versions(f"name='{self.modelName}'"):
          model_versions.append(dict(mv)['version'])
          if dict(mv)['current_stage'] == 'Staging':
            print("Archiving: {}".format(dict(mv)))
            # Archive the currently staged model
            client.transition_model_version_stage(
                name=self.modelName,
                version=dict(mv)['version'],
                stage="Archived"
            )
        client.transition_model_version_stage(
            name=self.modelName,
            version=model_versions[0],  # this model (current build)
            stage="Staging"
        )



# COMMAND ----------

tr = token_recommender(dataPath, "G08_model", fractional)

# COMMAND ----------

tr.train()

# COMMAND ----------

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
