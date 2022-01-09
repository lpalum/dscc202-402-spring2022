# Databricks notebook source
# MAGIC %md
# MAGIC # Schema Enforcement
# MAGIC 
# MAGIC 😲 The health tracker changed how it records data, which means that the raw data schema has changed.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notebook Objective
# MAGIC 
# MAGIC In this notebook we:
# MAGIC 1. Observe how schema enforcement deals with schema changes

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
# MAGIC ❗️Note that we have loaded our operation functions from the file `includes/main/python/operations_v2`. This updated operations file has been modified to transform the bronze table using the new schema.
# MAGIC 
# MAGIC The new schema has been loaded as `json_schema_v2`.

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

# COMMAND ----------

# MAGIC %md
# MAGIC ### Current Delta Architecture
# MAGIC 
# MAGIC Next, we demonstrate everything we have built up to this point in our
# MAGIC Delta Architecture.
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
# MAGIC We periodically run the `update_silver_table` function to update the Silver table based on the known issue of negative readings being ingested.

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
# MAGIC ## Retrieve Third Month of Data
# MAGIC 
# MAGIC Next, we use the utility function, `retrieve_data` to retrieve another file.
# MAGIC 
# MAGIC After you ingest the file, view the streams above.

# COMMAND ----------

retrieve_data(2020, 3, rawPath)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise:Write an Assertion Statement to Verify File Ingestion

# COMMAND ----------

# MAGIC %md
# MAGIC ### Expected File
# MAGIC 
# MAGIC The expected file has the following name:

# COMMAND ----------

file_2020_3 = "health_tracker_data_2020_3.json"

# COMMAND ----------

# TODO
assert FILL_THIS_IN in [item.name for item in dbutils.fs.ls(FILL_THIS_IN)]

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
# MAGIC ## What Is Schema Enforcement?
# MAGIC Schema enforcement, also known as schema validation, is a safeguard in Delta Lake that ensures data quality by rejecting writes to a table that do not match the table’s schema. Like the front desk manager at a busy restaurant that only accepts reservations, it checks to see whether each column in data inserted into the table is on its list of expected columns (in other words, whether each one has a “reservation”), and rejects any writes with columns that aren’t on the list.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Show Running Streams

# COMMAND ----------

for stream in spark.streams.active:
    print(stream.name)

# COMMAND ----------

# MAGIC %md
# MAGIC Note that the `write_bronze_to_silver` stream has died. If you navigate back up to the cell in which we started the streams, you should see the following error:
# MAGIC 
# MAGIC `org.apache.spark.sql.AnalysisException: A schema mismatch detected when writing to the Delta table`.
# MAGIC 
# MAGIC The stream has died because the schema of the incoming data did not match the schema of the table being written to.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stop All Streams
# MAGIC 
# MAGIC In the next notebook, we will take a look at schema evolution with Delta Lake.
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
