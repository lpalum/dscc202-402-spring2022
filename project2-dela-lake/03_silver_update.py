# Databricks notebook source
# MAGIC %md
# MAGIC # Silver Table Updates
# MAGIC 
# MAGIC We have processed data from the Bronze table to the Silver table.
# MAGIC We need to do some updates to ensure high quality in the Silver
# MAGIC table.
# MAGIC 
# MAGIC 😎 We're reading _from_ the Delta table now because a Delta table can be both a source AND a sink.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notebook Objective
# MAGIC 
# MAGIC In this notebook we:
# MAGIC 1. Harden the Raw to Bronze and Bronze to Silver Steps we wrote in a
# MAGIC    previous notebook.
# MAGIC 1. Diagnose data quality issues.
# MAGIC 1. Update the broken readings in the Silver table.
# MAGIC 1. Handle late-arriving data.

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
# MAGIC ### Display the Files in the Raw and Bronze Paths

# COMMAND ----------

display(dbutils.fs.ls(rawPath))

# COMMAND ----------

display(dbutils.fs.ls(bronzePath))

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
# MAGIC Next, we demonstrate everything we have built up to this point in our
# MAGIC Delta Architecture.
# MAGIC 
# MAGIC Again, we do so with composable functions included in the
# MAGIC file `includes/main/python/operations`.

# COMMAND ----------

# MAGIC %md
# MAGIC ### The Hardened Raw to Bronze Pipeline

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

# COMMAND ----------

# MAGIC %md
# MAGIC ### The Hardened Bronze to Silver Pipeline

# COMMAND ----------

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
# MAGIC ## Show Running Streams

# COMMAND ----------

for stream in spark.streams.active:
    print(stream.name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Diagnose Data Quality Issues
# MAGIC 
# MAGIC It is a good idea to perform quality checking on the data - such as looking for and reconciling anomalies - as well as further transformations such as cleaning and/or enriching the data.
# MAGIC 
# MAGIC In a visualization in the previous notebook, we noticed:
# MAGIC 
# MAGIC 1. the table is missing records.
# MAGIC 1. the presence of some negative recordings even though negative heart rates are impossible.
# MAGIC 
# MAGIC Let's assess the extent of the negative reading anomalies.

# COMMAND ----------

# MAGIC %md
# MAGIC **Exercise:** Create a Temporary View of the Broken Readings in the Silver Table
# MAGIC 
# MAGIC Display a count of the number of records for each day in the Silver
# MAGIC table where the measured heartrate is negative.

# COMMAND ----------

# TODO

broken_readings = (
  spark.read
  .format("delta")
  .load(silverPath)
  FILL_THIS_IN
)

broken_readings.createOrReplaceTempView("broken_readings")


# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT * FROM broken_readings

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT SUM(`count(heartrate)`) FROM broken_readings

# COMMAND ----------

# MAGIC %md
# MAGIC We have identified two issues with the Silver table:
# MAGIC 
# MAGIC 1. There are missing records
# MAGIC 1. There are records with broken readings
# MAGIC 
# MAGIC Let's update the table.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Update the Broken Readings
# MAGIC To update the broken sensor readings (heartrate less than zero), we'll interpolate using the value recorded before and after for each device. The `pyspark.sql` functions `lag()` and `lead()` will make this a trivial calculation.
# MAGIC In order to use these functions, we need to import the pyspark.sql.window function `Window`. This will allow us to create a date window consisting of the dates immediately before and after our missing value.
# MAGIC 
# MAGIC 🚎Window functions operate on a group of rows, referred to as a window, and calculate a return value for each row based on the group of rows. Window functions are useful for processing tasks such as calculating a moving average, computing a cumulative statistic, or accessing the value of rows given the relative position of the current row.
# MAGIC 
# MAGIC We'll write these values to a temporary view called `updates`. This view will be used later to upsert values into our Silver Delta table.
# MAGIC 
# MAGIC [pyspark.sql window functions documentation](https://spark.apache.org/docs/3.0.0/sql-ref-functions-builtin.html#window-functions)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create a DataFrame that Interpolates the Broken Values

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import col, lag, lead

dateWindow = Window.orderBy("p_eventdate")

interpolatedDF = spark.read.table("health_tracker_plus_silver").select(
    "*",
    lag(col("heartrate")).over(dateWindow).alias("prev_amt"),
    lead(col("heartrate")).over(dateWindow).alias("next_amt"),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create a DataFrame of Updates

# COMMAND ----------

updatesDF = interpolatedDF.where(col("heartrate") < 0).select(
    "device_id",
    ((col("prev_amt") + col("next_amt")) / 2).alias("heartrate"),
    "eventtime",
    "name",
    "p_eventdate",
)

display(updatesDF)

# COMMAND ----------

# MAGIC %md
# MAGIC **Exercise:** Write an assertion to verify that the Silver table and the UpdatesDF have the same schema

# COMMAND ----------

# TODO
assert FILL_THIS_IN, "Schemas do not match"
print("Assertion passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Update the Silver Table

# COMMAND ----------

from delta.tables import DeltaTable

silverTable = DeltaTable.forPath(spark, silverPath)

update_match = """
  health_tracker.eventtime = updates.eventtime
  AND
  health_tracker.device_id = updates.device_id
"""

update = {"heartrate": "updates.heartrate"}

(
    silverTable.alias("health_tracker")
    .merge(updatesDF.alias("updates"), update_match)
    .whenMatchedUpdate(set=update)
    .execute()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Handle Late-Arriving Data
# MAGIC 
# MAGIC 🤦🏼‍It turns out that our expectation of receiving the missing records late was correct. The complete month of February has subsequently been made available to us.

# COMMAND ----------

retrieve_data(2020, 2, rawPath, is_late=True)

# COMMAND ----------

display(dbutils.fs.ls(rawPath + "/late"))

# COMMAND ----------

# MAGIC %md
# MAGIC **Exercise:** Count the records in the late file
# MAGIC 
# MAGIC The late file is a json file in the `rawPath + "late"` directory.

# COMMAND ----------

# TODO
FILL_THIS_IN

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 🧐 You should note that the late file has all the records from the month of February, a count of 3480.
# MAGIC 
# MAGIC ❗️ If we simply append this file to the Bronze Delta table, it will create many duplicate entries.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read the Late File
# MAGIC 
# MAGIC Next we read in the late file. Note that we make use of the `transform_raw` function loaded from the `includes/main/python/operations` notebook.

# COMMAND ----------

kafka_schema = "value STRING"

lateRawDF = spark.read.format("text").schema(kafka_schema).load(rawPath + "/late")

transformedLateRawDF = transform_raw(lateRawDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Merge the Late-Arriving Data with the Bronze Table
# MAGIC 
# MAGIC We use the special method `.whenNotMatchedInsertAll` to insert only the records that are not present in the Bronze table. This is a best practice for preventing duplicate entries in a Delta table.

# COMMAND ----------

bronzeTable = DeltaTable.forPath(spark, bronzePath)

existing_record_match = "bronze.value = latearrivals.value"

(
    bronzeTable.alias("bronze")
    .merge(transformedLateRawDF.alias("latearrivals"), existing_record_match)
    .whenNotMatchedInsertAll()
    .execute()
)

# COMMAND ----------

# MAGIC %md
# MAGIC **Exercise:** Write An Aggregation on the Silver table
# MAGIC 
# MAGIC ### Count the number of records in the Silver table for each device id
# MAGIC 
# MAGIC 💪🏼 The Silver table is registered in the Metastore as `health_tracker_plus_silver`.
# MAGIC 
# MAGIC 👀 **Hint**: We did this exact query in the previous notebook.

# COMMAND ----------

# TODO
from pyspark.sql.functions import count

display(
FILL_THIS_IN
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Troubleshooting
# MAGIC 
# MAGIC 😫 If you run this query before the stream from the Bronze to the Silver tables has been picked up you will still see missing records for `device_id`: 4.
# MAGIC 
# MAGIC Wait a moment and run the query again.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check Yourself
# MAGIC 
# MAGIC You should see that there are an equal number of entries, 1440, for each device id.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table Histories

# COMMAND ----------

display(bronzeTable.history())

# COMMAND ----------

display(silverTable.history())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Time Travel
# MAGIC We can query an earlier version of the Delta table using the time travel feature. By running the following two cells, we can see that the current table count is larger than it was before we ingested the new data file into the stream.

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT COUNT(*) FROM health_tracker_plus_silver VERSION AS OF 0

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT COUNT(*) FROM health_tracker_plus_silver VERSION AS OF 1

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT COUNT(*) FROM health_tracker_plus_silver VERSION AS OF 2

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stop All Streams
# MAGIC 
# MAGIC In the next notebook, we will build the Silver to Gold Step.
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
