# Databricks notebook source
# MAGIC %md
# MAGIC # Silver to Gold - Building Aggregate Data Marts for End Users
# MAGIC 
# MAGIC We will now perform some aggregations on the data, as requested by one of our end users who wants to be able to quickly see summary statistics, aggregated by device id, in a dashboard in their chosen BI tool.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notebook Objective
# MAGIC 
# MAGIC In this notebook we:
# MAGIC 1. Create aggregations on the Silver table data
# MAGIC 1. Load the aggregate data into a Gold table

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step Configuration

# COMMAND ----------

# MAGIC %run ./includes/configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Operation Functions

# COMMAND ----------

# MAGIC %run ./includes/main/python/operations

# COMMAND ----------

# MAGIC %md
# MAGIC ### Display the Files in the Raw Paths

# COMMAND ----------

display(dbutils.fs.ls(rawPath))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make Notebook Idempotent

# COMMAND ----------

dbutils.fs.rm(goldPath, recurse=True)
dbutils.fs.rm(goldCheckpoint, recurse=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Start Streams
# MAGIC 
# MAGIC Before we add new streams, let's start the streams we have previously engineered.
# MAGIC 
# MAGIC We will start two named streams:
# MAGIC 
# MAGIC - `write_raw_to_bronze`
# MAGIC - `write_bronze_to_silver`

# COMMAND ----------

# MAGIC %md
# MAGIC ### Current Delta Architecture
# MAGIC 
# MAGIC Next, we demonstrate everything we have built up to this point in our
# MAGIC Delta architecture.
# MAGIC 
# MAGIC Again, we do so with composable functions included in the
# MAGIC file `includes/main/python/operations`.

# COMMAND ----------

rawDF = read_stream_raw(spark, rawPath)
transformedRawDF = transform_raw(rawDF)
rawToBronzeWriter = create_stream_writer(
    dataframe=transformedRawDF,
    checkpoint=bronzeCheckpoint,
    name="write_raw_to_bronze",
    partition_column="p_ingestdate",
)
rawToBronzeWriter.start(bronzePath)

bronzeDF = read_stream_delta(spark, bronzePath)
transformedBronzeDF = transform_bronze(bronzeDF)
bronzeToSilverWriter = create_stream_writer(
    dataframe=transformedBronzeDF,
    checkpoint=silverCheckpoint,
    name="write_bronze_to_silver",
    partition_column="p_eventdate",
)
bronzeToSilverWriter.start(silverPath)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Update the Silver Table
# MAGIC 
# MAGIC We periodically run the `update_silver_table` function to update the table and address the known issue of negative readings being ingested.

# COMMAND ----------

update_silver_table(spark, silverPath)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Show Running Streams

# COMMAND ----------

for stream in spark.streams.active:
    print(stream.name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Aggregation per User

# COMMAND ----------

# MAGIC %md
# MAGIC **Exercise:** Create a read stream DataFrame and aggregate over the Silver table
# MAGIC 
# MAGIC Use the following aggregates:
# MAGIC - mean of heartrate, aliased as `mean_heartrate`
# MAGIC - standard deviation of heartrate, aliased as `std_heartrate`
# MAGIC - maximum of heartrate, aliased as `max_heartrate`

# COMMAND ----------

# TODO

from pyspark.sql.functions import col, mean, stddev, max

silverTableReadStream = read_stream_delta(FILL_THIS_IN)

gold_health_tracker_data_df =(
  silverTableReadStream.groupBy("device_id")
  .agg(
    FILL_THIS_IN
  )
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## WRITE Stream Gold Table Aggregation
# MAGIC 
# MAGIC Note that we cannot use outputMode "append" for aggregations - we have to use "complete".

# COMMAND ----------

# MAGIC %md
# MAGIC **Exercise:** Write the aggregate DataFrame to a Gold table

# COMMAND ----------

# TODO

tableName = "aggregate_heartrate"
tableCheckpoint = goldCheckpoint + tableName
tablePath = goldPath + tableName

(
  gold_health_tracker_data_df.writeStream
  FILL_THIS_IN
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register Gold Table in the Metastore

# COMMAND ----------

spark.sql(
    """
DROP TABLE IF EXISTS health_tracker_gold_aggregate_heartrate
"""
)

spark.sql(
    f"""
CREATE TABLE health_tracker_gold_aggregate_heartrate
USING DELTA
LOCATION "{tablePath}"
"""
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Troubleshooting
# MAGIC 
# MAGIC 😫 If you try to run this before the `writeStream` above has been created, you may see the following error:
# MAGIC 
# MAGIC `
# MAGIC AnalysisException: Table schema is not set.  Write data into it or use CREATE TABLE to set the schema.;`
# MAGIC 
# MAGIC If this happens, wait a moment for the `writeStream` to instantiate and run the command again.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC We could now use this `health_tracker_gold` Delta table to define a dashboard. The query used to create the table could be issued nightly to prepare the dashboard for the following business day, or as often as needed according to SLA requirements.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stop All Streams
# MAGIC 
# MAGIC In the next notebook, you will harden the Silver to Gold Step.
# MAGIC 
# MAGIC Before we do so, let's shut down all streams in this notebook.

# COMMAND ----------

stop_all_streams()


# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
