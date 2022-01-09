# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze to Silver - ETL into a Silver table
# MAGIC 
# MAGIC We need to perform some transformations on the data to move it from bronze to silver tables.
# MAGIC 
# MAGIC 😎 We're reading _from_ the Delta table now because a Delta table can be both a source AND a sink.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notebook Objective
# MAGIC 
# MAGIC In this notebook we:
# MAGIC 1. Harden the Raw to Bronze Step we wrote in a previous notebook
# MAGIC 2. Develop the Bronze to Silver Step
# MAGIC    - Extract and Transform the Raw string to columns
# MAGIC    - Load this Data into the Silver Table

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
# MAGIC - `display_bronze`
# MAGIC 
# MAGIC 🤠 In a typical production setting, you would not interact with your streams as we are doing here. We stop and restart our streams in each new notebook for demonstration purposes. *It is easier to track everything that is happening if our streams are only running in our current notebook.*

# COMMAND ----------

# MAGIC %md
# MAGIC ### Current Delta Architecture
# MAGIC Next, we demonstrate everything we have built up to this point in our
# MAGIC Delta Architecture.
# MAGIC 
# MAGIC #### Harden the Raw to Bronze Step
# MAGIC 
# MAGIC We do so not with the ad hoc queries as written before, but now with
# MAGIC composable functions included in the file `includes/main/python/operations`.
# MAGIC This is a process known as **hardening** the step. If the data engineering
# MAGIC code is written in composable functions, it can be unit tested to ensure
# MAGIC stability.
# MAGIC 
# MAGIC 🛠 In our composable functions we will be making use of
# MAGIC [Python Type Hints](https://docs.python.org/3/library/typing.html).
# MAGIC 
# MAGIC #### Python Type Hints
# MAGIC 
# MAGIC For example, the function below takes and returns a string and is annotated as follows:
# MAGIC 
# MAGIC ```
# MAGIC def greeting(name: str) -> str:
# MAGIC     return 'Hello ' + name
# MAGIC ```
# MAGIC In the function `greeting`, the argument `name` is expected to be of type `str`
# MAGIC and the return type `str`.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Create the `rawDF` Streaming DataFrame
# MAGIC 
# MAGIC In the previous notebook, we wrote:
# MAGIC 
# MAGIC ```
# MAGIC rawDF = (
# MAGIC   spark.readStream
# MAGIC   .format("text")
# MAGIC   .schema(kafka_schema)
# MAGIC   .load(rawPath)
# MAGIC )
# MAGIC ```
# MAGIC 
# MAGIC Now, we use the following function in `includes/main/python/operations`
# MAGIC 
# MAGIC ```
# MAGIC def read_stream_raw(spark: SparkSession, rawPath: str) -> DataFrame:
# MAGIC   kafka_schema = "value STRING"
# MAGIC   return (
# MAGIC     spark.readStream
# MAGIC     .format("text")
# MAGIC     .schema(kafka_schema)
# MAGIC     .load(rawPath)
# MAGIC   )
# MAGIC ```
# MAGIC 
# MAGIC 🤩 Note that we have injected the current Spark Session into the function as the variable `spark`.

# COMMAND ----------

rawDF = read_stream_raw(spark, rawPath)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Transform the Raw Data
# MAGIC 
# MAGIC Next, we transform the raw data, `rawDF`. Previously, we wrote:
# MAGIC 
# MAGIC ```
# MAGIC rawDF = (
# MAGIC   rawDF.select(
# MAGIC     lit("files.training.databricks.com").alias("datasource"),
# MAGIC     current_timestamp().alias("ingesttime"),
# MAGIC     "value",
# MAGIC     current_timestamp().cast("date").alias("ingestdate")
# MAGIC   )
# MAGIC )
# MAGIC ```
# MAGIC 
# MAGIC Now, we use the following function in `includes/main/python/operations`
# MAGIC 
# MAGIC ```
# MAGIC def transform_raw(df: DataFrame) -> DataFrame:
# MAGIC   return (
# MAGIC     df.select(
# MAGIC       lit("files.training.databricks.com").alias("datasource"),
# MAGIC       current_timestamp().alias("ingesttime"),
# MAGIC       "value",
# MAGIC       current_timestamp().cast("date").alias("p_ingestdate")
# MAGIC     )
# MAGIC   )
# MAGIC ```

# COMMAND ----------

transformedRawDF = transform_raw(rawDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3: Write Stream to a Bronze Table
# MAGIC 
# MAGIC Finally, we write to the Bronze Table using Structured Streaming.
# MAGIC Previously, we wrote:
# MAGIC 
# MAGIC ```
# MAGIC (
# MAGIC   raw_health_tracker_data_df
# MAGIC   .select("datasource", "ingesttime", "value", col("ingestdate").alias("p_ingestdate"))
# MAGIC   .writeStream
# MAGIC   .format("delta")
# MAGIC   .outputMode("append")
# MAGIC   .option("checkpointLocation", bronzeCheckpoint)
# MAGIC   .partitionBy("p_ingestdate")
# MAGIC   .queryName("write_raw_to_bronze")
# MAGIC   .start(bronzePath)
# MAGIC )
# MAGIC ```
# MAGIC Now, we use the following function in `includes/main/python/operations`
# MAGIC 
# MAGIC ```
# MAGIC def create_stream_writer(dataframe: DataFrame, checkpoint: str,
# MAGIC                          name: str, partition_column: str=None,
# MAGIC                          mode: str="append") -> DataStreamWriter:
# MAGIC 
# MAGIC     stream_writer = (
# MAGIC         dataframe.writeStream
# MAGIC         .format("delta")
# MAGIC         .outputMode(mode)
# MAGIC         .option("checkpointLocation", checkpoint)
# MAGIC         .queryName(name)
# MAGIC     )
# MAGIC     if partition_column is not None:
# MAGIC       return stream_writer.partitionBy(partition_column)
# MAGIC     return stream_writer
# MAGIC ```
# MAGIC 
# MAGIC 🤯 **Note**: This function will be used repeatedly, every time we create
# MAGIC a `DataStreamWriter`.
# MAGIC 
# MAGIC ☝🏿 This function returns a `DataStreamWriter`, not a `DataFrame`. This means
# MAGIC that we will have to call `.start()` as a function method to start the stream.

# COMMAND ----------

rawToBronzeWriter = create_stream_writer(
    dataframe=transformedRawDF,
    checkpoint=bronzeCheckpoint,
    name="write_raw_to_bronze",
    partition_column="p_ingestdate",
)

rawToBronzeWriter.start(bronzePath)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Display the Bronze Table

# COMMAND ----------

bronzeDF = read_stream_delta(spark, bronzePath)
display(bronzeDF, streamName="display_bronze")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Show Running Streams

# COMMAND ----------

for stream in spark.streams.active:
    print(stream.name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make Notebook Idempotent

# COMMAND ----------

dbutils.fs.rm(silverPath, recurse=True)
dbutils.fs.rm(silverCheckpoint, recurse=True)

# COMMAND ----------

# MAGIC %md
# MAGIC **Exercise:** Count Records in the Bronze Table
# MAGIC 
# MAGIC Display how many records are in our table so we can watch it grow as the data streams in. As we ingest more files, you will be able to return to this streaming display and watch the count increase.
# MAGIC 
# MAGIC - Use the DataFrame, `bronzeDF`, which is a reference to the Bronze Delta table
# MAGIC - Write spark code to count the number of records in the Bronze Delta table
# MAGIC 
# MAGIC 💡 **Hint:** While a standard DataFrame has a simple `.count()` method, when performing operations such as `count` on a stream, you must use `.groupby()` before the aggregate operation.

# COMMAND ----------

# TODO
display(
  bronzeDF
  FILL_THIS_IN,
  streamName="display_bronze_count"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrieve Second Month of Data
# MAGIC 
# MAGIC Next, we use the utility function, `retrieve_data` to retrieve another file.
# MAGIC 
# MAGIC After you ingest the file by running the following cell, view the streams above; you should be able to watch the data being ingested.

# COMMAND ----------

retrieve_data(2020, 2, rawPath)

# COMMAND ----------

# MAGIC %md
# MAGIC **Exercise:** Write an Assertion Statement to Verify File Ingestion
# MAGIC 
# MAGIC The expected file has the following name:

# COMMAND ----------

file_2020_2 = "health_tracker_data_2020_2.json"

# COMMAND ----------

# TODO
assert FILL_THIS_IN in [item.name for item in dbutils.fs.ls(FILL_THIS_IN)], "File not present in Raw Path"
print("Assertion passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extracting Nested JSON
# MAGIC 
# MAGIC We now begin the work of creating the Silver Table. First, we extract the JSON data from the `value` column in the Bronze Delta table. That this is being done after first landing our ingested data in a Bronze table means that we do not need to worry about the ingestion process breaking because the data did not parse.
# MAGIC 
# MAGIC This extraction consists of two steps:
# MAGIC 
# MAGIC 1. We extract the nested JSON from `bronzeDF` using the `pyspark.sql` function `from_json`.
# MAGIC 
# MAGIC    📒 The `from_json` function requires that a schema be passed as argument. Here we pass the schema `json_schema = "device_id INTEGER, heartrate DOUBLE, name STRING, time FLOAT"`.
# MAGIC 
# MAGIC 1. We flatten the nested JSON into a new DataFrame by selecting all nested values of the `nested_json` column.

# COMMAND ----------

from pyspark.sql.functions import from_json

json_schema = "device_id INTEGER, heartrate DOUBLE, name STRING, time FLOAT"

silver_health_tracker = bronzeDF.select(
    from_json(col("value"), json_schema).alias("nested_json")
).select("nested_json.*")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transform the Data
# MAGIC 
# MAGIC The "time" column isn't currently human-readable in Unix time format.
# MAGIC We need to transform it to make it useful. We also extract just the date
# MAGIC from the timestamp. Next, we transform `silver_health_tracker` with the
# MAGIC following transformations:
# MAGIC 
# MAGIC - convert the `time` column to a timestamp with the name `eventtime`
# MAGIC - convert the `time` column to a date with the name `p_eventdate`
# MAGIC 
# MAGIC Note that we name the new column `p_eventdate` to indicate that we are
# MAGIC partitioning on this column.

# COMMAND ----------

from pyspark.sql.functions import col, from_unixtime

silver_health_tracker = silver_health_tracker.select(
    "device_id",
    "heartrate",
    from_unixtime("time").cast("timestamp").alias("eventtime"),
    "name",
    from_unixtime("time").cast("date").alias("p_eventdate"),
)

# COMMAND ----------

# MAGIC %md
# MAGIC **Exercise:** Write an Assertion To Verify the Schema
# MAGIC 
# MAGIC The DataFrame `silver_health_tracker` should now have the following schema:
# MAGIC 
# MAGIC ```
# MAGIC device_id: integer
# MAGIC heartrate: double
# MAGIC eventtime: timestamp
# MAGIC name: string
# MAGIC p_eventdate: date```
# MAGIC 
# MAGIC Write a schema using DDL format to complete the below assertion statement.
# MAGIC 
# MAGIC 💪🏼 Remember, the function `_parse_datatype_string` converts a DDL format schema string into a Spark schema.

# COMMAND ----------

# TODO
from pyspark.sql.types import _parse_datatype_string

assert FILL_THIS_IN == _parse_datatype_string(FILL_THIS_IN), "File not present in Silver Path"
print("Assertion passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## WRITE Stream to a Silver Table
# MAGIC 
# MAGIC Next, we stream write to the Silver table.
# MAGIC 
# MAGIC We partion this table on event data (`p_eventdate`).

# COMMAND ----------

(
    silver_health_tracker.writeStream.format("delta")
    .outputMode("append")
    .option("checkpointLocation", silverCheckpoint)
    .partitionBy("p_eventdate")
    .queryName("write_bronze_to_silver")
    .start(silverPath)
)

# COMMAND ----------

spark.sql(
    """
DROP TABLE IF EXISTS health_tracker_plus_silver
"""
)

spark.sql(
    f"""
CREATE TABLE health_tracker_plus_silver
USING DELTA
LOCATION "{silverPath}"
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
# MAGIC ## Explore and Visualize the Data
# MAGIC 
# MAGIC After running the following cell, click on "Plot Options..." and set the plot options as shown below:
# MAGIC 
# MAGIC ![Plot Options](https://files.training.databricks.com/images/pipelines_plot_options.png)

# COMMAND ----------

display(
    spark.readStream.table("health_tracker_plus_silver"), streamName="display_silver"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### What patterns do you notice in the data? Anomalies?

# COMMAND ----------

# MAGIC %md
# MAGIC ### Missing Records
# MAGIC 
# MAGIC When we look at the Silver table, we expect to see two months of data, five device measurements, 24 hours a day for (31 + 29) days, or 7200 records. (The data was recorded during the month of February in a leap year, which is why there are 29 days in the month.)
# MAGIC 
# MAGIC ❗️We do not have a correct count. It looks like `device_id`: 4 is missing 72 records.

# COMMAND ----------

from pyspark.sql.functions import count

display(
    spark.read.table("health_tracker_plus_silver").groupby("device_id").agg(count("*"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table Histories
# MAGIC 
# MAGIC Recall that the Delta transaction log allows us to view all of the commits that have taken place in a Delta table's history.

# COMMAND ----------

from delta.tables import DeltaTable

bronzeTable = DeltaTable.forPath(spark, bronzePath)
silverTable = DeltaTable.forPath(spark, silverPath)

# COMMAND ----------

display(bronzeTable.history())

# COMMAND ----------

display(silverTable.history())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Time Travel
# MAGIC 
# MAGIC We can query an earlier version of the Delta table using the time travel feature. By running the following two cells, we can see that the current table count is larger than it was before we ingested the new data file into the stream.

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT COUNT(*) FROM health_tracker_plus_bronze VERSION AS OF 0

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT COUNT(*) FROM health_tracker_plus_bronze VERSION AS OF 1

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT COUNT(*) FROM health_tracker_plus_bronze

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stop All Streams
# MAGIC 
# MAGIC In the next notebook, we will analyze data in the Silver Delta table, and perform some update operations on the data.
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
