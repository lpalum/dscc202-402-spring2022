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

display(dbutils.fs.ls(BASE_DELTA_PATH+"bronze/"))

# COMMAND ----------

  # Grab the global variables
wallet_address,start_date = Utils.create_widgets()
print(wallet_address,start_date)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Your Code starts here...

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



# COMMAND ----------

wallet_n_token_address = spark.sql("select * from g08_db.erc20_antijoin")

# COMMAND ----------

wallet_n_token_address.count()

# COMMAND ----------

#wallet_token_address_with_ids_2022 = wallet_token_address_with_ids_2022.withColumn("sum_ether", col("wai_value")/(10**13))

# COMMAND ----------

# wallet_token_address_with_ids_2022.select("*").where(col('wallet_address') == 
# '0x7a250d5630b4cf539739df2c5dacb4c659f2488d').where(col('token_address') == 
# '0x0ae055097c6d159879521c384f1d2123d1f195e6').count()

# COMMAND ----------

# wallet_n_token_address_not_null = wallet_n_token_address.filter(col("wallet_address").isNotNull())


# COMMAND ----------

# wallet_n_token_address_not_null.count()

# COMMAND ----------

wallet_n_token_address = wallet_n_token_address.withColumnRenamed("to_address","wallet_address")

# COMMAND ----------

token_transact_count_in_wallet = wallet_n_token_address.groupBy("wallet_address", "token_address").count()

# COMMAND ----------

display(token_transact_count_in_wallet)

# COMMAND ----------

token_transact_count_in_wallet.count()

# COMMAND ----------

#testing groupby
# sum(test.select("*").where(col('wallet_address') == 
# '0xdef171fe48cf0115b1d80b88dc8eab59176fee57').toPandas()["count"])

# COMMAND ----------

# MAGIC %md ## Encoding IDs

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id
unique_wallet = token_transact_count_in_wallet.select("wallet_address").distinct().withColumn('new_wallet_id',monotonically_increasing_id().cast(IntegerType())).withColumnRenamed("wallet_address","w_address")
display(unique_wallet)
unique_token = token_transact_count_in_wallet.select("token_address").distinct().withColumn('new_token_id',monotonically_increasing_id().cast(IntegerType())).withColumnRenamed("token_address","t_address")
display(unique_token)

# COMMAND ----------

unique_token.count()

# COMMAND ----------

inter = token_transact_count_in_wallet.join(unique_wallet, token_transact_count_in_wallet.wallet_address==unique_wallet.w_address,"inner").drop("w_address")
token_transact_count_in_wallet_with_IDs = inter.join(unique_token, inter.token_address == unique_token.t_address, "inner").drop("t_address")
display(token_transact_count_in_wallet_with_IDs)

# COMMAND ----------

# cast ids to int

token_transact_count_in_wallet_with_IDs = token_transact_count_in_wallet_with_IDs.withColumn("new_wallet_id", token_transact_count_in_wallet_with_IDs["new_wallet_id"].cast(IntegerType()))

token_transact_count_in_wallet_with_IDs = token_transact_count_in_wallet_with_IDs.withColumn("new_token_id", token_transact_count_in_wallet_with_IDs["new_token_id"].cast(IntegerType()))

token_transact_count_in_wallet_with_IDs = token_transact_count_in_wallet_with_IDs.withColumn("count", token_transact_count_in_wallet_with_IDs["count"].cast(IntegerType()))

# COMMAND ----------

token_transact_count_in_wallet_with_IDs.select("*").where(col('count') < 5).count()

# COMMAND ----------

# MAGIC %md ## 71% transactions only transact 1 time, 90% transaction only transacted less than 5 times 

# COMMAND ----------

token_transact_count_in_wallet_with_IDs.select("*").where(col('count') < 2).count()

# COMMAND ----------

display(token_transact_count_in_wallet_with_IDs.orderBy(col("count").desc()))

# COMMAND ----------

notColdStart = token_transact_count_in_wallet_with_IDs.select("*").where(col('count') > 1)

# COMMAND ----------

# token_transact_count_in_wallet_with_IDs = token_transact_count_in_wallet_with_IDs.filter(col("wallet_address").isNotNull())

# COMMAND ----------

sorted_df = notColdStart.sort(col('count').desc()).withColumn("index",monotonically_increasing_id())
display(sorted_df)

# COMMAND ----------

highest_counts = sorted_df.limit(180)
display(highest_counts)

# COMMAND ----------

reg_counts = sorted_df.filter(col('index') > 180)

# COMMAND ----------

highest_counts = highest_counts.withColumn('count', lit(1000000))

# COMMAND ----------

reg_notColdStart = reg_counts.unionByName(highest_counts)

# COMMAND ----------

display(reg_notColdStart)

# COMMAND ----------

#notColdStart.write.format('delta').saveAsTable('g08_db.notColdStart')

# COMMAND ----------

notColdStart = spark.sql("select * from g08_db.notColdStart ORDER BY RAND() LIMIT 100000")

# COMMAND ----------

# We'll hold out 60% for training, 20% of our data for validation, and leave 20% for testing
seed = 42
(split_60_df, split_a_20_df, split_b_20_df) = notColdStart.randomSplit([0.6, 0.2, 0.2], seed = seed)

# Let's cache these datasets for performance
training_df = split_60_df.cache()
validation_df = split_a_20_df.cache()
test_df = split_b_20_df.cache()

print('Training: {0}, validation: {1}, test: {2}\n'.format(
  training_df.count(), validation_df.count(), test_df.count())
)
training_df.show(3)
validation_df.show(3)
test_df.show(3)

# COMMAND ----------

# Filter out inconsistent new_wallet_id
# wallet_ids = notColdStart.toPandas().new_wallet_id
# for wid in wallet_ids:
#     if training_df['new_wallet_id'].contains(wid) or wid not in validation_df['new_wallet_id'].contains(wid) or wid not in test_df['new_wallet_id'].contains(wid):
#         training_df = training_df.filter('new_wallet_id' != wid)
#         validation_df = validation_df.filter('new_wallet_id' != wid)
#         test_df = test_df.filter('new_wallet_id' != wid)
#         # remove row in training_df where new_wallet_id == wid
#         # remove row in validatoin_df where new_wallet_id ==w
        
# token_ids = notColdStart.toPandas().new_token_id
# for wid in token_ids:
#     if wid not in training_df['new_token_id'] or wid not in validation_df['new_token_id'] or wid not in test_df['new_token_id']:
#         training_df = training_df.filter('new_token_id' != wid)
#         validation_df = validation_df.filter('new_token_id' != wid)
#         test_df = test_df.filter('new_token_id' != wid)
    

#Filter out inconsistent new_token_id

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists g08_db.test_data

# COMMAND ----------

test_df.write.format('delta').saveAsTable('g08_db.test_data')

# COMMAND ----------

from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Let's initialize our ALS learner
als = ALS()

# Now set the parameters for the method
als.setMaxIter(5)\
   .setSeed(seed)\
   .setItemCol("new_token_id")\
   .setRatingCol("count")\
   .setUserCol("new_wallet_id")\
   .setColdStartStrategy("drop")
# ^^Spark allows users to set the coldStartStrategy parameter to “drop” in order to drop any rows in the DataFrame of predictions that contain NaN values. The evaluation metric will then be computed over the non-NaN data and will be valid. 


# Now let's compute an evaluation metric for our test dataset
# We Create an RMSE evaluator using the label and predicted columns
reg_eval = RegressionEvaluator(predictionCol="prediction", labelCol="count", metricName="rmse")

grid = ParamGridBuilder() \
  .addGrid(als.maxIter, [10,15]) \
  .addGrid(als.regParam, [0.15, 0.2, 0.25, 0.5, 0.75, 1]) \
  .addGrid(als.rank, [8, 12, 16, 20, 24, 50]) \
  .build()

# Create a cross validator, using the pipeline, evaluator, and parameter grid you created in previous steps.
cv = CrossValidator(estimator=als, evaluator=reg_eval, estimatorParamMaps=grid, numFolds=3)

tolerance = 0.03
ranks = [8, 12, 16, 20, 24]
regParams = [0.25, 0.5, 0.75, 1]
errors = [[0]*len(ranks)]*len(regParams)
models = [[0]*len(ranks)]*len(regParams)
err = 0
min_error = float('inf')
best_rank = -1
i = 0

for regParam in regParams:
  j = 0
  for rank in ranks:
    # Set the rank here:
    als.setParams(rank = rank, regParam = regParam)
    # Create the model with these parameters.
    model = als.fit(training_df)
    # Run the model to create a prediction. Predict against the validation_df.
    predict_df = model.transform(validation_df)

    # Remove NaN values from prediction (due to SPARK-14489)
    predicted_count_df = predict_df.filter(predict_df.prediction != float('nan'))
    predicted_count_df = predicted_count_df.withColumn("prediction", F.abs(F.round(predicted_count_df["prediction"],0)))
    # Run the previously created RMSE evaluator, reg_eval, on the predicted_ratings_df DataFrame
    error = reg_eval.evaluate(predicted_count_df)
    errors[i][j] = error
    models[i][j] = model
    print( 'For rank %s, regularization parameter %s the RMSE is %s' % (rank, regParam, error))
    if error < min_error:
      min_error = error
      best_params = [i,j]
    j += 1
  i += 1

als.setRegParam(regParams[best_params[0]])
als.setRank(ranks[best_params[1]])
print( 'The best model was trained with regularization parameter %s' % regParams[best_params[0]])
print( 'The best model was trained with rank %s' % ranks[best_params[1]])
my_model = models[best_params[0]][best_params[1]]

# COMMAND ----------

predicted_count_df.show(10)

# COMMAND ----------

# create an MLflow experiment for this model
MY_EXPERIMENT = "/Users/" + dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get() + "/" + 'test2' + "-experiment/"
mlflow.set_experiment(MY_EXPERIMENT)

# COMMAND ----------



# COMMAND ----------

class token_recommender()

# COMMAND ----------

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
reg_eval = RegressionEvaluator(predictionCol="prediction", labelCol="count", metricName="rmse")

# Setup an ALS hyperparameter tuning grid search
grid = ParamGridBuilder() \
  .addGrid(als.maxIter, [5, 10, 15]) \
  .addGrid(als.regParam, [0.15, 0.25, 0.5]) \
  .addGrid(als.rank, [12, 16, 24, 50]) \
  .build()

"""
grid = ParamGridBuilder() \
  .addGrid(als.maxIter, [5]) \
  .addGrid(als.regParam, [0.25]) \
  .addGrid(als.rank, [16]) \
  .build()
"""

# Create a cross validator, using the pipeline, evaluator, and parameter grid you created in previous steps.
cv = CrossValidator(estimator=als, evaluator=reg_eval, estimatorParamMaps=grid, numFolds=3)

# COMMAND ----------

# dataPath = ""

# setup the schema for the model
input_schema = Schema([
  ColSpec("integer", "new_token_id"),
  ColSpec("integer", "new_wallet_id"),
])
output_schema = Schema([ColSpec("double")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

#training_data_version = DeltaTable.forPath(spark, dataPath+"triplets").history().head(1)[0]['version']

with mlflow.start_run(run_name='test'+"-run") as run:
  mlflow.set_tags({"group": 'g08', "class": "test"})
  #mlflow.log_params({"user_rating_training_data_version": training_data_version})
#   mlflow.log_params({"user_rating_training_data_version": training_data_version,"minimum_play_count":self.minPlayCount})


  # Run the cross validation on the training dataset. The cv.fit() call returns the best model it found.
  cvModel = cv.fit(training_df)

  # Evaluate the best model's performance on the validation dataset and log the result.
  validation_metric = reg_eval.evaluate(cvModel.transform(validation_df))

  mlflow.log_metric('test_' + reg_eval.getMetricName(), validation_metric) 

  # Log the best model.
  mlflow.spark.log_model(spark_model=cvModel.bestModel, signature = signature,
                         artifact_path='als-model', registered_model_name='test')



# COMMAND ----------

mlflow.spark.log_model(spark_model=cvModel.bestModel, signature = signature,
                         artifact_path='als-model', registered_model_name='test2')

# COMMAND ----------

"""
- Capture the latest model version
- archive any previous Staged version
- Transition this version to Staging
"""
client = MlflowClient()
model_versions = []

# Transition this model to staging and archive the current staging model if there is one
for mv in client.search_model_versions(f"name='test'"):
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

"""
Class that encapsulates the definition, training and testing of an ALS Recommender
using the facilites of MLflow for tracking and registration/packaging
"""

class Recommender():
    
"""
Build out the required plumbing and electrical work:
- Get the delta lake version of the training data to be stored with the model (bread crumbs)
- Set (create) the experiment name
- Load and create training test splits
- instantiate an alternate least squares estimator and its associated evaluation and tuning grid
- create a cross validation configuration for the model training
"""
  def __init__(self, dataPath: str, modelName: str, minPlayCount: int = 1)->None:
    self.dataPath = dataPath
    self.modelName = modelName
    self.minPlayCount = minPlayCount

    # create an MLflow experiment for this model
    MY_EXPERIMENT = "/Users/" + dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get() + "/" + self.modelName + "-experiment/"
    mlflow.set_experiment(MY_EXPERIMENT)

    # split the data set into train, validation and test anc cache them
    # We'll hold out 60% for training, 20% of our data for validation, and leave 20% for testing
    self.raw_plays_df_with_int_ids = spark.read.format('delta').load(self.dataPath+"triplets")
    self.raw_plays_df_with_int_ids = self.raw_plays_df_with_int_ids.filter(self.raw_plays_df_with_int_ids.Plays >= self.minPlayCount).cache()
    self.metadata_df = spark.read.format('delta').load(self.dataPath+"metadata").cache()
    self.training_data_version = DeltaTable.forPath(spark, self.dataPath+"triplets").history().head(1)[0]['version']

    seed = 42
    (split_60_df, split_a_20_df, split_b_20_df) = self.raw_plays_df_with_int_ids.randomSplit([0.6, 0.2, 0.2], seed = seed)
    # Let's cache these datasets for performance
    self.training_df = split_60_df.cache()
    self.validation_df = split_a_20_df.cache()
    self.test_df = split_b_20_df.cache()

    # Initialize our ALS learner
    als = ALS()

    # Now set the parameters for the method
    als.setMaxIter(5)\
       .setSeed(seed)\
       .setItemCol("new_songId")\
       .setRatingCol("Plays")\
       .setUserCol("new_userId")\
       .setColdStartStrategy("drop")

    # Now let's compute an evaluation metric for our test dataset
    # We Create an RMSE evaluator using the label and predicted columns
    self.reg_eval = RegressionEvaluator(predictionCol="prediction", labelCol="Plays", metricName="rmse")

    # Setup an ALS hyperparameter tuning grid search
    grid = ParamGridBuilder() \
      .addGrid(als.maxIter, [5, 10, 15]) \
      .addGrid(als.regParam, [0.15, 0.2, 0.25]) \
      .addGrid(als.rank, [4, 8, 12, 16, 20]) \
      .build()

    """
    grid = ParamGridBuilder() \
      .addGrid(als.maxIter, [5]) \
      .addGrid(als.regParam, [0.25]) \
      .addGrid(als.rank, [16]) \
      .build()
    """

    # Create a cross validator, using the pipeline, evaluator, and parameter grid you created in previous steps.
    self.cv = CrossValidator(estimator=als, evaluator=self.reg_eval, estimatorParamMaps=grid, numFolds=3)

      """
      Train the ALS music recommendation using the training and validation set and the cross validation created
      at the time of instantiation.  Use MLflow to log the training results and push the best model from this
      training session to the MLflow registry at "Staging"
      """
      def train(self):
        # setup the schema for the model
        input_schema = Schema([
          ColSpec("integer", "new_songId"),
          ColSpec("integer", "new_userId"),
        ])
        output_schema = Schema([ColSpec("double")])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        with mlflow.start_run(run_name=self.modelName+"-run") as run:
          mlflow.set_tags({"group": 'admin', "class": "DSCC202-402"})
          mlflow.log_params({"user_rating_training_data_version": self.training_data_version,"minimum_play_count":self.minPlayCount})

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

  """
  Test the model in staging with the test dataset generated when this object (music recommender)
  was instantiated.
  """
  def test(self):
    # THIS SHOULD BE THE VERSION JUST TRANINED
    model = mlflow.spark.load_model('models:/'+self.modelName+'/Staging')
    # View the predictions
    test_predictions = model.transform(self.test_df)
    RMSE = self.reg_eval.evaluate(test_predictions)
    print("Staging Model Root-mean-square error on the test dataset = " + str(RMSE))  
  
 

# COMMAND ----------

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
