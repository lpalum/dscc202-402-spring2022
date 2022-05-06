# Databricks notebook source
# MAGIC %md
# MAGIC ## Rubric for this module
# MAGIC - Implement a routine to "promote" your model at **Staging** in the registry to **Production** based on a boolean flag that you set in the code.
# MAGIC - Using wallet addresses from your **Staging** and **Production** model test data, compare the recommendations of the two models.

# COMMAND ----------

# MAGIC %run ./includes/utilities

# COMMAND ----------

# MAGIC %run ./includes/configuration

# COMMAND ----------

# Grab the global variables
# wallet_address,start_date = Utils.create_widgets()
# print(wallet_address,start_date)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Your Code Starts Here...

# COMMAND ----------

def shouldWePromote(staging_model, production_model, test_data_df)->bool:
    from pyspark.ml.evaluation import RegressionEvaluator
    test_predictions = staging_model.transform(test_data_df)
    reg_eval = RegressionEvaluator(predictionCol="prediction", labelCol="count", metricName="rmse")
    stg_RMSE = reg_eval.evaluate(test_predictions)
    
    test_predictions = production_model.transform(test_data_df)
    reg_eval = RegressionEvaluator(predictionCol="prediction", labelCol="count", metricName="rmse")
    prod_RMSE = reg_eval.evaluate(test_predictions)
    
    print(f"staging error: {stg_RMSE}, production error: {prod_RMSE}")
    return stg_RMSE < prod_RMSE



# COMMAND ----------

from mlflow.tracking.client import MlflowClient
import time

client = MlflowClient()

model_name = "G08_db"

latest_staging_version = None
latest_production_version = None
for mv in client.search_model_versions(f"name='{model_name}'"):
    if dict(mv)['current_stage'] == 'Staging':
        latest_staging_version = dict(mv)['version']
    elif dict(mv)['current_stage'] == 'Production':
        latest_production_version = dict(mv)['version']


if latest_staging_version is None:
    print("No model in staging, exiting....")
else:    
    
    model_version_details = client.get_model_version(name=model_name, version=latest_staging_version)

    while model_version_details.status != "READY":
        time.sleep(20)

    if latest_production_version is not None:
        test_df = spark.sql("select * from g08_db.test_data")
        import mlflow.pyfunc

        stg_uri = "models:/{model_name}/{ver}".format(model_name=model_name, ver=latest_staging_version)
        prod_uri = "models:/{model_name}/{ver}".format(model_name=model_name, ver=latest_production_version)
        staging = mlflow.spark.load_model(stg_uri)
        production = mlflow.spark.load_model(prod_uri)
        if shouldWePromote(staging, production, test_df):
            print("Promoting Staging to Production")
            client.transition_model_version_stage(
              name=model_name,
              version=latest_staging_version,
              stage='Production',
            )

        else:
            print("Current model in production performs better, no change")
#             return

    else:
        print("No model in production, promoting...")
        model_version_details = client.get_model_version(name=model_name, version=latest_staging_version)
        client.transition_model_version_stage(
          name=model_version_details.name,
          version=model_version_details.version,
          stage='Production',
        )

    timeout_counter = 10
    model_version_details = client.get_model_version(name=model_name, version=latest_staging_version)

    while model_version_details.current_stage != "Production" and timeout_counter > 0:
        timeout_counter-=1
        time.sleep(20)

    if latest_production_version is not None:
        client.transition_model_version_stage(
          name=model_name,
          version=latest_production_version,
          stage="Archived",
        )

# COMMAND ----------

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
