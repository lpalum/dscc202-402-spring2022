# Databricks notebook source
# MAGIC %md
# MAGIC # Raw Data Retrieval

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notebook Objective
# MAGIC 
# MAGIC In this notebook we:
# MAGIC 
# MAGIC 1. Ingest data from a remote source into our source directory, `rawPath`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step Configuration
# MAGIC 
# MAGIC Before you run this cell, make sure to add a unique user name to the file
# MAGIC `includes/configuration`, e.g.
# MAGIC 
# MAGIC ```
# MAGIC username = "yourfirstname_yourlastname"
# MAGIC ```

# COMMAND ----------

# MAGIC %run ./includes/configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make Notebook Idempotent

# COMMAND ----------

dbutils.fs.rm(rawPath, recurse=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrieve First Month of Data
# MAGIC 
# MAGIC Next, we use the utility function, `retrieve_data` to retrieve the first file we will ingest. The function takes three arguments:
# MAGIC 
# MAGIC - `year: int`
# MAGIC - `month: int`
# MAGIC - `rawPath: str`
# MAGIC - `is_late: bool` (optional)

# COMMAND ----------

retrieve_data(2020, 1, rawPath)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Expected File
# MAGIC 
# MAGIC The expected file has the following name:

# COMMAND ----------

file_2020_1 = "health_tracker_data_2020_1.json"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Display the Files in the Raw Path

# COMMAND ----------

display(dbutils.fs.ls(rawPath))

# COMMAND ----------

# MAGIC %md
# MAGIC **Exercise:** Write an Assertion Statement to Verify File Ingestion
# MAGIC 
# MAGIC Note: the `print` statement would typically not be included in production code, nor in code used to test this notebook.

# COMMAND ----------

# TODO
assert FILL_THIS_IN in [item.name for item in dbutils.fs.ls(FILL_THIS_IN)], "File not present in Raw Path"
print("Assertion passed.")


# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
