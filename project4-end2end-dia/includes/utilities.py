# Databricks notebook source
"""
## Helper routines...
- mounting buckets in s3
- use of database
- creation of tables in the metastore
- ploting
- data reading
"""

import mlflow
import pandas as pd
import tempfile
from datetime import datetime as dt
from datetime import timedelta
import warnings
import json

from pyspark.sql.functions import *
from pyspark.sql.types import *

warnings.filterwarnings("ignore")

# COMMAND ----------

class Utils:
    
    @staticmethod
    def mount_datasets(group_name):
      class_mount_name = "dscc202-datasets"
      group_mount_name = "dscc202-datasets/misc/{}".format(group_name)
      
      class_s3_bucket = "s3a://dscc202-datasets/"
      try:
        dbutils.fs.mount(class_s3_bucket, "/mnt/%s" % class_mount_name)
      except:
        dbutils.fs.unmount("/mnt/%s" % class_mount_name)
        dbutils.fs.mount(class_s3_bucket, "/mnt/%s" % class_mount_name)
        
      return "/mnt/%s" % class_mount_name, "/mnt/%s" % group_mount_name

    @staticmethod
    def create_metastore(group_name):
      # Setup the hive meta store if it does not exist and select database as the focus of future sql commands in this notebook
      spark.sql(f"CREATE DATABASE IF NOT EXISTS {group_name}_db")
      spark.sql(f"USE {group_name}_db")
      return f"{group_name}_db"
      
    @staticmethod
    def create_delta_dir(group_name):
      delta_dir = f"/mnt/dscc202-datasets/misc/{group_name}/tokenrec/tables/"
      dbutils.fs.mkdirs(delta_dir)
      return delta_dir
    
    @staticmethod
    def create_widgets():
      dbutils.widgets.removeAll()

      dbutils.widgets.text('00.Wallet_Address', "0xf02d7ee27ff9b2279e76a60978bf8cca9b18a3ff")
      dbutils.widgets.text('01.Start_Date', "2022-01-01")

      wallet_address = str(dbutils.widgets.get('00.Wallet_Address'))
      start_date = str(dbutils.widgets.get('01.Start_Date'))

      return wallet_address,start_date
    
