# Databricks notebook source
# MAGIC %md
# MAGIC # Silver to Gold Step

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notebook Objective
# MAGIC 
# MAGIC In this notebook you:
# MAGIC 1. Harden the Silver to Gold step we wrote in the previous notebook using the composable functions in the operations file.

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
# MAGIC **Exercise:** Harden the Silver to Gold step that we created in the previous notebook.
# MAGIC 
# MAGIC Now that you have seen the pattern, fill out the following code block to complete this step.
# MAGIC 
# MAGIC 💁🏻‍♀️Remember to use `mode="complete"` with streaming aggregate tables.

# COMMAND ----------

# TODO

tableName = "/aggregate_heartrate"
tableCheckpoint = goldCheckpoint + tableName
tablePath = goldPath + tableName

silverDF             = FILL_THIS_IN
transformedSilverDF  = FILL_THIS_IN
silverToGoldAggWriter = FILL_THIS_IN

silverToGoldAggWriter.FILL_THIS_IN

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Exercise:** Show all running streams

# COMMAND ----------

# TODO

FILL_THIS_IN

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stop All Streams
# MAGIC 
# MAGIC In the next notebook, we will take a look at schema enforcement and evolution with Delta Lake.
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
