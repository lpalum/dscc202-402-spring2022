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

# MAGIC %python
# MAGIC #Mock Data --> Replace with real data
# MAGIC data = [
# MAGIC     ("https://assets.coingecko.com/coins/images/18581/thumb/soul.jpg?1632491969", "AppCoins","(APPC)", "0x1a7a8bd9106f2b8d977e08582dc7d24c723ab0db"),
# MAGIC     ("https://assets.coingecko.com/coins/images/18581/thumb/soul.jpg?1632491969", "AppCoins","(APPC)", "0x1a7a8bd9106f2b8d977e08582dc7d24c723ab0db"),
# MAGIC     ("https://assets.coingecko.com/coins/images/18581/thumb/soul.jpg?1632491969", "AppCoins","(APPC)", "0x1a7a8bd9106f2b8d977e08582dc7d24c723ab0db"),
# MAGIC 
# MAGIC ]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Your code starts here...

# COMMAND ----------

# MAGIC %python
# MAGIC from typing import List, Tuple
# MAGIC def generateAppOutput(recommendations: List[Tuple[str, str, str, str]], wallet_address: str):
# MAGIC     header =f"""
# MAGIC     <h3>Recommend Tokens for user address:</h3>
# MAGIC     {wallet_address}
# MAGIC     """
# MAGIC     rows = ""
# MAGIC     for d in recommendations:
# MAGIC         icon, name, symbol, address = d
# MAGIC #         print(f"https://etherscan.io/token/{address}")
# MAGIC         rows += f"""
# MAGIC         <tr>
# MAGIC         <td style="text-align:center"><img src="{icon}" alt="{name}"></td>
# MAGIC         <td style="text-align:center">{name} {symbol}</td>
# MAGIC         <td style="text-align:center"><a href="https://etherscan.io/token/{address}">Etherscan Link</a></td>
# MAGIC         </tr>
# MAGIC         """
# MAGIC 
# MAGIC     table = f"""
# MAGIC     <table padding="15px">
# MAGIC     {rows}
# MAGIC     </table>
# MAGIC     """
# MAGIC 
# MAGIC     result = f"""
# MAGIC 
# MAGIC     <h2>Recommend Tokens for user address:</h2>
# MAGIC     {wallet_address}
# MAGIC     <br>
# MAGIC     <br>
# MAGIC     """
# MAGIC     
# MAGIC     displayHTML(result + table)

# COMMAND ----------

generateAppOutput(data, wallet_address)

# COMMAND ----------

  def recommend(self, wallet_address: int)->(DataFrame,DataFrame):
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

# COMMAND ----------

def runTokenRecommender(wallet_address: str):
    #TODO: check if cold start or not
    ownedTokens = spark.sql
    recommendations = getRecommendations(wallet_address)
    generateAppOutput(recommendations, wallet_address)

# COMMAND ----------

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
