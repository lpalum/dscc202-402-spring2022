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
wallet_address,start_date, fractional = Utils.create_widgets()
print(wallet_address,start_date, fractional)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Your code starts here...

# COMMAND ----------

# MAGIC %python
# MAGIC model_name = "G08_model"
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
        tokens_unused = not_cold_start_df.filter(~not_cold_start_df["token_address"].isin([row["contract_address"] for row in tokens_transacted.collect()])).select('wallet_address','token_address','count','new_wallet_id','new_token_id')
    #.withColumn('new_wallet_id', F.lit(wallet_df.first().new_wallet_id)).distinct()
        model = mlflow.spark.load_model(f'models:/{model_name}/Production')
        predicted_tokens = model.transform(tokens_unused)
        results = predicted_tokens.join(metadata_df, predicted_tokens.token_address == metadata_df.contract_address, 'inner').select('image', 'name', 'symbol', 'contract_address', 'prediction').distinct().dropDuplicates(["symbol"]).orderBy('prediction', ascending = False)
        top_5_results = results.take(5)
        return top_5_results
    else: # Handle for cold start
        pop_tokens = spark.sql("select * from g08_db.popular_token")
        pop_tokens = pop_tokens.select('image', 'name','symbol','token_address')
        
        return pop_tokens.take(5)

# COMMAND ----------

def runTokenRecommender(wallet_address: str):
    #TODO: check if cold start or not
    results = recommend(wallet_address)
    if len(results) < 5:
        print("No recommendations generated!")
        return
    recommendations = []
    for result in results:
        image, name, symbol, address = result[0], result[1], result[2].upper(), result[3]
#         print(result[4])
        recommendations.append((image, name, symbol, address))
    generateAppOutput(recommendations, wallet_address)
    
    return results
    

# COMMAND ----------

results = runTokenRecommender(wallet_address)

# COMMAND ----------

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
