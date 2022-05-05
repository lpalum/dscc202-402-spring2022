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
# MAGIC     ("https://assets.coingecko.com/coins/images/18581/thumb/soul.jpg?1632491969", "AppCoins","(APPC)", "https://appcoins.io/"),
# MAGIC     ("https://assets.coingecko.com/coins/images/18581/thumb/soul.jpg?1632491969", "AppCoins","(APPC)", "https://appcoins.io/"),
# MAGIC     ("https://assets.coingecko.com/coins/images/18581/thumb/soul.jpg?1632491969", "AppCoins","(APPC)", "https://appcoins.io/"),
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
# MAGIC         icon, name, symbol, link = d
# MAGIC         rows += f"""
# MAGIC         <tr>
# MAGIC         <td style="text-align:center"><img src="{icon}" alt="{name}"></td>
# MAGIC         <td style="text-align:center">{name} {symbol}</td>
# MAGIC         <td style="text-align:center"><a href="{link}">Website</a></td>
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

def runTokenRecommender(wallet_address: str):
    recommendations = getRecommendations(wallet_address)
    generateAppOutput(recommendations, wallet_address)

# COMMAND ----------

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
