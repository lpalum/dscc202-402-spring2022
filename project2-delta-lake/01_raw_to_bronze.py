# Databricks notebook source
# MAGIC %md
# MAGIC # Raw to Bronze Pattern

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notebook Objective
# MAGIC 
# MAGIC In this notebook we:
# MAGIC 1. Ingest Raw Data
# MAGIC 2. Augment the data with Ingestion Metadata
# MAGIC 3. Stream write the augmented data to a Bronze Table

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step Configuration

# COMMAND ----------

# MAGIC %run ./includes/configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ### Display the Files in the Raw Path

# COMMAND ----------

display(dbutils.fs.ls(rawPath))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make Notebook Idempotent

# COMMAND ----------

dbutils.fs.rm(bronzePath, recurse=True)
dbutils.fs.rm(bronzeCheckpoint, recurse=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingest raw data
# MAGIC 
# MAGIC Next, we will stream files from the source directory and write each line as a string to the Bronze table.

# COMMAND ----------

kafka_schema = "value STRING"

raw_health_tracker_data_df = (
    spark.readStream.format("text").schema(kafka_schema).load(rawPath)
)

# COMMAND ----------

# MAGIC %md
# MAGIC **Exercise:** Write an Assertion Statement to Verify the Schema of the Raw Data
# MAGIC 
# MAGIC At this point, we write an assertion statement to verify that our streaming DataFrame has the schema we expect.
# MAGIC 
# MAGIC Your assertion should make sure that the `raw_health_tracker_data_df` DataFrame has the correct schema.
# MAGIC 
# MAGIC 🤠 The function `_parse_datatype_string` (read more [here](http://spark.apache.org/docs/2.1.2/api/python/_modules/pyspark/sql/types.html)) converts a DDL format schema string into a Spark schema.

# COMMAND ----------

# TODO
from pyspark.sql.types import _parse_datatype_string
assert FILL_THIS_IN == _parse_datatype_string(FILL_THIS_IN), "File not present in Raw Path"
print("Assertion passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Display the Raw Data
# MAGIC 
# MAGIC 🤓 Each row here is a raw string in JSON format, as would be passed by a stream server like Kafka.

# COMMAND ----------

display(raw_health_tracker_data_df, streamName="display_raw")

# COMMAND ----------

# MAGIC %md
# MAGIC ❗️ To prevent the `display` function from continuously streaming, run the following utility function.

# COMMAND ----------

stop_named_stream(spark, "display_raw")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingestion Metadata
# MAGIC 
# MAGIC As part of the ingestion process, we record metadata for the ingestion. In this case, we track the data sources, the ingestion time (`ingesttime`), and the ingest date (`ingestdate`) using the `pyspark.sql` functions `current_timestamp` and `lit`.

# COMMAND ----------

from pyspark.sql.functions import current_timestamp, lit

raw_health_tracker_data_df = raw_health_tracker_data_df.select(
    lit("files.training.databricks.com").alias("datasource"),
    current_timestamp().alias("ingesttime"),
    "value",
    current_timestamp().cast("date").alias("ingestdate"),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## WRITE Stream to a Bronze Table
# MAGIC 
# MAGIC Finally, we write to the Bronze Table using Structured Streaming.
# MAGIC 
# MAGIC 🙅🏽‍♀️ While we _can_ write directly to tables using the `.table()` notation, this will create fully managed tables by writing output to a default location on DBFS. This is not best practice and should be avoided in nearly all cases.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Partitioning
# MAGIC This course uses a dataset that is extremely small relative to an actual production system. Still we demonstrate the best practice of partitioning by date and partition on the ingestion date, column `p_ingestdate`.
# MAGIC 
# MAGIC 😲 Note that we have aliased the `ingestdate` column to be `p_ingestdate`. We have done this in order to inform anyone who looks at the schema for this table that it has been partitioned by the ingestion date.

# COMMAND ----------

from pyspark.sql.functions import col

(
    raw_health_tracker_data_df.select(
        "datasource", "ingesttime", "value", col("ingestdate").alias("p_ingestdate")
    )
    .writeStream.format("delta")
    .outputMode("append")
    .option("checkpointLocation", bronzeCheckpoint)
    .partitionBy("p_ingestdate")
    .queryName("write_raw_to_bronze")
    .start(bronzePath)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Checkpointing
# MAGIC 
# MAGIC When defining a Delta Lake streaming query, one of the options that you need to specify is the location of a checkpoint directory.
# MAGIC 
# MAGIC `.writeStream.format("delta").option("checkpointLocation", <path-to-checkpoint-directory>) ...`
# MAGIC 
# MAGIC This is actually a structured streaming feature. It stores the current state of your streaming job.
# MAGIC 
# MAGIC Should your streaming job stop for some reason and you restart it, it will continue from where it left off.
# MAGIC 
# MAGIC 💀 If you do not have a checkpoint directory, when the streaming job stops, you lose all state around your streaming job and upon restart, you start from scratch.
# MAGIC 
# MAGIC ✋🏽 Also note that every streaming job should have its own checkpoint directory: no sharing.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a Reference to the Delta table files
# MAGIC 
# MAGIC In this command we create a Spark DataFrame via a reference to the Delta file in DBFS.

# COMMAND ----------

bronze_health_tracker = spark.readStream.format("delta").load(bronzePath)

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
# MAGIC ## Display the files in the Delta table
# MAGIC 
# MAGIC These files can be viewed using the `dbutils.fs.ls` function.

# COMMAND ----------

display(dbutils.fs.ls(bronzePath))

# COMMAND ----------

# MAGIC %md
# MAGIC **Exercise:** Write an Assertion Statement to Verify the Schema of the Bronze Delta Table
# MAGIC 
# MAGIC At this point, we write an assertion statement to verify that our Bronze Delta table has the schema we expect.
# MAGIC 
# MAGIC Your assertion should make sure that the `bronze_health_tracker` DataFrame has the correct schema.
# MAGIC 
# MAGIC 💪🏼 Remember, the function `_parse_datatype_string` converts a DDL format schema string into a Spark schema.

# COMMAND ----------

# TODO
assert FILL_THIS_IN == _parse_datatype_string(FILL_THIS_IN), "File not present in Bronze Path"
print("Assertion passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Display Running Streams
# MAGIC 
# MAGIC You can use the following code to display all streams that are currently running.

# COMMAND ----------

for stream in spark.streams.active:
    print(stream.name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the Bronze Table in the Metastore
# MAGIC 
# MAGIC Recall that a Delta table registered in the Metastore is a reference to a physical table created in object storage.
# MAGIC 
# MAGIC We just created a Bronze Delta table in object storage by writing data to a specific location. If we register that location with the Metastore as a table, we can query the tables using SQL.
# MAGIC 
# MAGIC (Because we will never directly query the Bronze table, it is not strictly necessary to register this table in the Metastore, but we will do so for demonstration purposes.)
# MAGIC 
# MAGIC At Delta table creation, the Delta files in Object Storage define the schema, partitioning, and table properties. For this reason, it is not necessary to specify any of these when registering the table with the Metastore. Furthermore, no table repair is required. The transaction log stored with the Delta files contains all metadata needed for an immediate query.

# COMMAND ----------

spark.sql(
    """
DROP TABLE IF EXISTS health_tracker_plus_bronze
"""
)

spark.sql(
    f"""
CREATE TABLE health_tracker_plus_bronze
USING DELTA
LOCATION "{bronzePath}"
"""
)

# COMMAND ----------

display(
    spark.sql(
        """
  DESCRIBE DETAIL health_tracker_plus_bronze
  """
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delta Lake Python API
# MAGIC Delta Lake provides programmatic APIs to examine and manipulate Delta tables.
# MAGIC 
# MAGIC Here, we create a reference to the Bronze table using the Delta Lake Python API.

# COMMAND ----------

from delta.tables import DeltaTable

bronzeTable = DeltaTable.forPath(spark, bronzePath)

# COMMAND ----------

display(bronzeTable.history())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stop All Streams
# MAGIC 
# MAGIC In the next notebook, we will stream data from the Bronze table to a Silver Delta table.
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
