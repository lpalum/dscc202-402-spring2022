# Databricks notebook source
# MAGIC %md
# MAGIC ## Token Recommendation
# MAGIC <table border=0>
# MAGIC   <tr><td><img src='https://data-science-at-scale.s3.amazonaws.com/images/rec-application.png'></td>
# MAGIC     <td>Your application should allow a specific wallet address to be entered via a widget in your application notebook.  Each time a new wallet address is entered, a new recommendation of the top tokens for consideration should be made. <br> **Bonus** (3 points): include links to the Token contract on the blockchain or etherscan.io for further investigation.</td></tr>
# MAGIC   </table>

# COMMAND ----------

# MAGIC %run ./includes/utilities

# COMMAND ----------

# MAGIC %run ./includes/configuration

# COMMAND ----------

# Grab the global variables
wallet_address,start_date = Utils.create_widgets()
print(wallet_address,start_date)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Your code starts here...

# COMMAND ----------

# MAGIC %md ## Basic recommendation (cold start)

# COMMAND ----------

from pyspark.sql import DataFrame
from pyspark.sql.types import *
from pyspark.sql import functions as F
from delta.tables import *
import random

import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
song_ids_with_total_listens = raw_plays_df_with_int_ids.groupBy('songId') \
                                                       .agg(F.count(raw_plays_df_with_int_ids.Plays).alias('User_Count'),
                                                            F.sum(raw_plays_df_with_int_ids.Plays).alias('Total_Plays')) \
                                                       .orderBy('Total_Plays', ascending = False)

print('song_ids_with_total_listens:',
song_ids_with_total_listens.show(3, truncate=False))

# Join with metadata to get artist and song title
song_names_with_plays_df = song_ids_with_total_listens.join(metadata_df, 'songId' ) \
                                                      .filter('User_Count >= 2') \
                                                      .select('artist_name', 'title', 'songId', 'User_Count','Total_Plays') \
                                                      .orderBy('Total_Plays', ascending = False)

print('song_names_with_plays_df:',
song_names_with_plays_df.show(20, truncate = False))

# COMMAND ----------

# MAGIC %md ##Recommendation based on model

# COMMAND ----------

"""
Method takes a specific userId and returns the songs that they have listened to
and a set of recommendations in rank order that they may like based on their
listening history.
"""
def recommend(self, userId: int)->(DataFrame,DataFrame):
# generate a dataframe of songs that the user has previously listened to
listened_songs = self.raw_plays_df_with_int_ids.filter(self.raw_plays_df_with_int_ids.new_userId == userId) \
                                          .join(self.metadata_df, 'songId') \
                                          .select('new_songId', 'artist_name', 'title','Plays') \

# generate dataframe of unlistened songs
unlistened_songs = self.raw_plays_df_with_int_ids.filter(~ self.raw_plays_df_with_int_ids['new_songId'].isin([song['new_songId'] for song in listened_songs.collect()])) \
                                            .select('new_songId').withColumn('new_userId', F.lit(userId)).distinct()

# feed unlistened songs into model for a predicted Play count
model = mlflow.spark.load_model('models:/'+self.modelName+'/Staging')
predicted_listens = model.transform(unlistened_songs)

return (listened_songs.select('artist_name','title','Plays').orderBy('Plays', ascending = False), predicted_listens.join(self.raw_plays_df_with_int_ids, 'new_songId') \
                 .join(self.metadata_df, 'songId') \
                 .select('artist_name', 'title', 'prediction') \
                 .distinct() \
                 .orderBy('prediction', ascending = False)) 

 

"""
Generate a data frame that recommends a number of songs for each of the users in the dataset (model)
"""
def recommendForUsers(self, numOfSongs: int) -> DataFrame:
    model = mlflow.spark.load_model('models:/'+self.modelName+'/Staging')
    return model.stages[0].recommendForAllUsers(numOfSongs)

"""
Generate a data frame that recommends a number of users for each of the songs in the dataset (model)
"""
  
def recommendForSongs(self, numOfUsers: int) -> DataFrame:
    model = mlflow.spark.load_model('models:/'+self.modelName+'/Staging')
    return model.stages[0].recommendForAllItems(numOfUsers)

# COMMAND ----------

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
