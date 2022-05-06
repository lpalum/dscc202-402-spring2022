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
# MAGIC         <td style="text-align:center">{name} ({symbol})</td>
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

# MAGIC %sql
# MAGIC select * from ethereumetl.token_prices_usd

# COMMAND ----------

not_cold_start_df = spark.sql("select * from g08_db.notcoldstart")
display(not_cold_start_df)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from g08_db.test_data

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from g08_db.popular_tokens

# COMMAND ----------

def recommend(wallet_address: str)->DataFrame:
    # generate a dataframe of songs that the user has previously listened to
    from pyspark.sql import functions as F

    not_cold_start_df = spark.sql("select * from g08_db.notcoldstart")
    metadata_df = spark.sql("select name, symbol, image, contract_address  from ethereumetl.token_prices_usd")
    wallet_df = not_cold_start_df.filter(not_cold_start_df.wallet_address == wallet_address)
#     display(wallet_df)
    not_cold_start = wallet_df.count() > 1
    if not_cold_start:
        tokens_transacted = wallet_df.join(metadata_df, wallet_df.token_address == metadata_df.contract_address, "inner").select('image', 'name', 'symbol', 'contract_address')
#         print(tokens_transacted.count())
        tokens_unused = not_cold_start_df.filter(~not_cold_start_df["token_address"].isin([row["contract_address"] for row in tokens_transacted.collect()])).select('new_token_id').withColumn('new_wallet_id', F.lit(wallet_df.first().new_wallet_id)).distinct()
        display(tokens_unused)
#         display(tokens_transacted)
#         print(tokens_unused.count())
        # feed unlistened songs into model for a predicted Play count
        model = mlflow.spark.load_model('models:/test2/Production')
        predicted_tokens = model.transform(tokens_unused)
        display(predicted_tokens)
        
        return predicted_tokens
    else: # Handle for cold start
        print("cold start")
        pass

    #     return (tokens_transacted.select('image', 'name', 'symbol', 'contract_address'), predicted_listens.join(not_cold_start_df, 'new_songId') \
    #                      .join(self.metadata_df, 'songId') \
    #                      .select('artist_name', 'title', 'prediction') \
    #                      .distinct() \
    #                      .orderBy('prediction', ascending = False)) 
        return None

# COMMAND ----------

results = recommend("0x05b1984c74c531eb52ea749ca2d5c51d9f058ff6")

# COMMAND ----------

display(results[0])

# COMMAND ----------

def runTokenRecommender(wallet_address: str):
    #TODO: check if cold start or not
    ownedTokens = spark.sql
    recommendations = getRecommendations(wallet_address)
    generateAppOutput(recommendations, wallet_address)

# COMMAND ----------

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
