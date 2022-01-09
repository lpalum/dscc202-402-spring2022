# Databricks notebook source
# MAGIC %md
# MAGIC # Partitioning
# MAGIC 1. Get partitions and cores
# MAGIC 1. Repartition DataFrames
# MAGIC 1. Configure default shuffle partitions
# MAGIC 
# MAGIC ##### Methods
# MAGIC - DataFrame (<a href="https://spark.apache.org/docs/latest/api/python/pyspark.sql.html?highlight=dataframe#pyspark.sql.DataFrame" target="_blank">Python</a>/<a href="http://spark.apache.org/docs/latest/api/scala/org/apache/spark/sql/Dataset.html" target="_blank">Scala</a>): `repartition`, `coalesce`, `rdd.getNumPartitions`
# MAGIC - SparkConf (<a href="https://spark.apache.org/docs/latest/api/python/pyspark.html?highlight=sparkconf#pyspark.SparkConf" target="_blank">Python</a>/<a href="http://spark.apache.org/docs/latest/api/scala/org/apache/spark/SparkConf.html" target="_blank">Scala</a>): `get`, `set`
# MAGIC - SparkSession (<a href="https://spark.apache.org/docs/latest/api/python/pyspark.sql.html?highlight=sparksession#pyspark.sql.SparkSession" target="_blank">Python</a>/<a href="http://spark.apache.org/docs/latest/api/scala/org/apache/spark/sql/SparkSession.html" target="_blank">Scala</a>): `sparkContext.defaultParallelism`
# MAGIC 
# MAGIC ##### SparkConf parameters
# MAGIC - `spark.sql.shuffle.partitions`, `spark.sql.adaptive.enabled`

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md
# MAGIC ### ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Get partitions and cores
# MAGIC 
# MAGIC Use an `rdd` method to get the number of DataFrame partitions

# COMMAND ----------

df = spark.read.parquet(eventsPath)
df.rdd.getNumPartitions()

# COMMAND ----------

# MAGIC %md
# MAGIC Access SparkContext through SparkSession to get the number of cores or slots
# MAGIC 
# MAGIC SparkContext is also provided in Databricks notebooks as the variable `sc`

# COMMAND ----------

print(spark.sparkContext.defaultParallelism)
# print(sc.defaultParallelism)

# COMMAND ----------

# MAGIC %md
# MAGIC ### ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Repartition DataFrame

# COMMAND ----------

# MAGIC %md
# MAGIC #### `repartition`
# MAGIC Returns a new DataFrame that has exactly `n` partitions.

# COMMAND ----------

repartitionedDF = df.repartition(8)

# COMMAND ----------

repartitionedDF.rdd.getNumPartitions()

# COMMAND ----------

# MAGIC %md
# MAGIC #### `coalesce`
# MAGIC Returns a new DataFrame that has exactly `n` partitions, when the fewer partitions are requested
# MAGIC 
# MAGIC If a larger number of partitions is requested, it will stay at the current number of partitions

# COMMAND ----------

coalesceDF = df.coalesce(8)

# COMMAND ----------

coalesceDF.rdd.getNumPartitions()

# COMMAND ----------

# MAGIC %md
# MAGIC ### ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Configure default shuffle partitions
# MAGIC 
# MAGIC Use `SparkConf` to access the spark configuration parameter for default shuffle partitions

# COMMAND ----------

spark.conf.get("spark.sql.shuffle.partitions")

# COMMAND ----------

# MAGIC %md
# MAGIC Configure default shuffle partitions to match the number of cores

# COMMAND ----------

spark.conf.set("spark.sql.shuffle.partitions", "8")

# COMMAND ----------

# MAGIC %md
# MAGIC ###![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Adaptive Query Execution
# MAGIC 
# MAGIC In Spark 3, <a href="https://spark.apache.org/docs/latest/sql-performance-tuning.html#adaptive-query-execution" target="_blank">AQE</a> is now able to <a href="https://databricks.com/blog/2020/05/29/adaptive-query-execution-speeding-up-spark-sql-at-runtime.html" target="_blank"> dynamically coalesce shuffle partitions</a> at runtime
# MAGIC 
# MAGIC Spark SQL can use `spark.sql.adaptive.enabled` to control whether AQE is turned on/off (disabled by default)

# COMMAND ----------

spark.conf.get("spark.sql.adaptive.enabled")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Clean up classroom

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Cleanup
