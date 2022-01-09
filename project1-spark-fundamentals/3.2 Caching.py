# Databricks notebook source
# MAGIC %md
# MAGIC # Caching
# MAGIC 1. Clear cache
# MAGIC 1. Cache DataFrame
# MAGIC 1. Remove Cache
# MAGIC 1. Cache table for RDD name
# MAGIC 1. Spark UI - Storage
# MAGIC 
# MAGIC ##### Methods
# MAGIC - DataFrame (<a href="https://spark.apache.org/docs/latest/api/python/pyspark.sql.html?highlight=dataframe#pyspark.sql.DataFrame" target="_blank">Python</a>/<a href="http://spark.apache.org/docs/latest/api/scala/org/apache/spark/sql/Dataset.html" target="_blank">Scala</a>): `union` `cache`, `unpersist`
# MAGIC - Catalog (<a href="https://spark.apache.org/docs/latest/api/python/pyspark.sql.html?highlight=catalog#pyspark.sql.Catalog" target="_blank">Python</a>/<a href="http://spark.apache.org/docs/latest/api/scala/org/apache/spark/sql/catalog/Catalog.html" target="_blank">Scala</a>): `cacheTable`

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md
# MAGIC ### ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Clear all cache
# MAGIC You can clear cache on your cluster by restarting your cluster or using the method below.

# COMMAND ----------

# DO NOT RUN ON SHARED CLUSTER - CLEARS YOUR CACHE AND YOUR COWORKER'S
# spark.catalog.clearCache()

# COMMAND ----------

# MAGIC %md
# MAGIC Let's use the BedBricks events dataset

# COMMAND ----------

eventsJsonPath = "/mnt/training/ecommerce/events/events-1m.json"

df = (spark.read
  .option("inferSchema", True)
  .json(eventsJsonPath))

# COMMAND ----------

df.orderBy("event_timestamp").count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Cache DataFrame

# COMMAND ----------

# MAGIC %md
# MAGIC #### `cache()`
# MAGIC Persist this Dataset with the default storage level
# MAGIC 
# MAGIC ##### Alias for `persist`

# COMMAND ----------

df.cache()

# COMMAND ----------

# MAGIC %md
# MAGIC :NOTE: A call to `cache()` does not immediately materialize the data in cache.
# MAGIC 
# MAGIC An action using the DataFrame must be executed for Spark to actually cache the data.
# MAGIC 
# MAGIC Check Spark UI Storage tab before and after materializing the cache below.

# COMMAND ----------

df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC Observe change in execution time below.

# COMMAND ----------

df.orderBy("event_timestamp").count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Remove cache
# MAGIC :NOTE: As a best practice, you should always evict your DataFrames from cache when you no longer need them.

# COMMAND ----------

# MAGIC %md
# MAGIC #### `unpersist()`
# MAGIC Removes cache for a DataFrame

# COMMAND ----------

df.unpersist()

# COMMAND ----------

# MAGIC %md
# MAGIC ###![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Cache table for RDD name
# MAGIC Cache a table to assign a nicer name to the cached RDD for the Storage UI.

# COMMAND ----------

df.createOrReplaceTempView("Pageviews_DF_Python")
spark.catalog.cacheTable("Pageviews_DF_Python")

df.count()

# COMMAND ----------

df.unpersist()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Clean up classroom

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Cleanup
