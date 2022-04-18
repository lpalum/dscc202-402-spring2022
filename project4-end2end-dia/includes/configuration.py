# Databricks notebook source
## Enter your group specific information here...

GROUP='GXX'   # CHANGE TO YOUR GROUP NAME format Gxx

if GROUP not in ["G01","G02","G03","G04","G05","G06","G07","G08","G09","G10","G11","G12","G13","G14"]:
  print("DONT FORGET TO SET YOUR GROUP NAME IN includes/configuration NOTEBOOK")
  dbutils.notebook.exit(json.dumps({"exit_code": "BAD GROUP NAME"}))

# COMMAND ----------

"""
Enter any project wide configuration here...
- paths
- table names
- constants
- etc
"""

# Some configuration of the cluster
spark.conf.set("spark.sql.shuffle.partitions", "32")  # Configure the size of shuffles the same as core count on your cluster
spark.conf.set("spark.sql.adaptive.enabled", "true")  # Spark 3.0 AQE - coalescing post-shuffle partitions, converting sort-merge join to broadcast join, and skew join optimization
spark.conf.set("spark.databricks.io.cache.enabled", "true") # set the delta file cache to true

# Mount the S3 class and group specific buckets
CLASS_DATA_PATH, GROUP_DATA_PATH = Utils.mount_datasets(GROUP)

# create the delta tables base dir in the s3 bucket for your group
BASE_DELTA_PATH = Utils.create_delta_dir(GROUP)

# Create the metastore for your group and set as the default db
GROUP_DBNAME = Utils.create_metastore(GROUP)

# ETHEREUM DB Name (bronze data resides here)
ETHEREUM_DBNAME = "ethereumetl"

# Setup this spark configuration to use these variables in the SQL code (if needed)
spark.conf.set('ethereum.db',ETHEREUM_DBNAME)
spark.conf.set('group.db',GROUP_DBNAME)

# COMMAND ----------

displayHTML(f"""
<table border=1>
<tr><td><b>Variable Name</b></td><td><b>Value</b></td></tr>
<tr><td>CLASS_DATA_PATH</td><td>{CLASS_DATA_PATH}</td></tr>
<tr><td>GROUP_DATA_PATH</td><td>{GROUP_DATA_PATH}</td></tr>
<tr><td>BASE_DELTA_PATH</td><td>{BASE_DELTA_PATH}</td></tr>
<tr><td>GROUP_DBNAME</td><td>{GROUP_DBNAME}</td></tr>
<tr><td>ETHEREUM_DBNAME</td><td>{ETHEREUM_DBNAME}</td></tr>
</table>
""")
