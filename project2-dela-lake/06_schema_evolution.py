# Databricks notebook source
# MAGIC %md
# MAGIC # Schema Evolution
# MAGIC 
# MAGIC 😲 The health tracker changed how it records data, which means that the
# MAGIC raw data schema has changed. In this notebook, we show how to build our
# MAGIC streams to merge the changes to the schema.
# MAGIC 
# MAGIC **TODO** *Discussion on what kinds of changes will work with the merge option.*

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notebook Objective
# MAGIC 
# MAGIC In this notebook we:
# MAGIC 1. Use schema evolution to deal with schema changes

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step Configuration

# COMMAND ----------

# MAGIC %run ./includes/configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Operation Functions

# COMMAND ----------

# MAGIC %run ./includes/main/python/operations_v2

# COMMAND ----------

# MAGIC %md
# MAGIC ### Display the Files in the Raw Paths

# COMMAND ----------

display(dbutils.fs.ls(rawPath))

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
# MAGIC 
# MAGIC ❗️Note that we have loaded our operation functions from the file `includes/main/python/operations_v2`. This updated operations file has been modified to transform the bronze table using the new schema.
# MAGIC 
# MAGIC The new schema has been loaded as `json_schema_v2`.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Current Delta Architecture
# MAGIC **TODO**
# MAGIC Next, we demonstrate everything we have built up to this point in our
# MAGIC Delta Architecture.
# MAGIC 
# MAGIC Again, we do so with composable functions included in the
# MAGIC file `includes/main/python/operations`.
# MAGIC 
# MAGIC Add the `mergeSchema=True` argument to the Silver table stream writer.

# COMMAND ----------

# TODO
rawDF             = read_stream_raw(spark, rawPath)
transformedRawDF  = transform_raw(rawDF)
rawToBronzeWriter = create_stream_writer(
  dataframe=transformedRawDF,
  checkpoint=bronzeCheckpoint,
  name="write_raw_to_bronze",
  partition_column="p_ingestdate"
)
rawToBronzeWriter.start(bronzePath)

bronzeDF             = read_stream_delta(spark, bronzePath)
transformedBronzeDF  = transform_bronze(bronzeDF)
bronzeToSilverWriter = create_stream_writer(
  dataframe=transformedBronzeDF,
  checkpoint=silverCheckpoint,
  name="write_bronze_to_silver",
  partition_column="p_eventdate"
)
bronzeToSilverWriter.start(silverPath)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Show Running Streams

# COMMAND ----------

for stream in spark.streams.active:
    print(stream.name)

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT COUNT(*) FROM health_tracker_plus_bronze

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT COUNT(*) FROM health_tracker_plus_silver

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DESCRIBE health_tracker_plus_silver

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stop All Streams
# MAGIC 
# MAGIC In the next notebook in this course, we will take a look at schema enforcement and evolution with Delta Lake.
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
