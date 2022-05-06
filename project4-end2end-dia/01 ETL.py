# Databricks notebook source
# MAGIC %md
# MAGIC ## Ethereum Blockchain Data Analysis - <a href=https://github.com/blockchain-etl/ethereum-etl-airflow/tree/master/dags/resources/stages/raw/schemas>Table Schemas</a>
# MAGIC - **Transactions** - Each block in the blockchain is composed of zero or more transactions. Each transaction has a source address, a target address, an amount of Ether transferred, and an array of input bytes. This table contains a set of all transactions from all blocks, and contains a block identifier to get associated block-specific information associated with each transaction.
# MAGIC - **Blocks** - The Ethereum blockchain is composed of a series of blocks. This table contains a set of all blocks in the blockchain and their attributes.
# MAGIC - **Receipts** - the cost of gas for specific transactions.
# MAGIC - **Traces** - The trace module is for getting a deeper insight into transaction processing. Traces exported using <a href=https://openethereum.github.io/JSONRPC-trace-module.html>Parity trace module</a>
# MAGIC - **Tokens** - Token data including contract address and symbol information.
# MAGIC - **Token Transfers** - The most popular type of transaction on the Ethereum blockchain invokes a contract of type ERC20 to perform a transfer operation, moving some number of tokens from one 20-byte address to another 20-byte address. This table contains the subset of those transactions and has further processed and denormalized the data to make it easier to consume for analysis of token transfer events.
# MAGIC - **Contracts** - Some transactions create smart contracts from their input bytes, and this smart contract is stored at a particular 20-byte address. This table contains a subset of Ethereum addresses that contain contract byte-code, as well as some basic analysis of that byte-code. 
# MAGIC - **Logs** - Similar to the token_transfers table, the logs table contains data for smart contract events. However, it contains all log data, not only ERC20 token transfers. This table is generally useful for reporting on any logged event type on the Ethereum blockchain.
# MAGIC 
# MAGIC In Addition, there is a price feed that changes daily (noon) that is in the **token_prices_usd** table
# MAGIC 
# MAGIC ### Rubric for this module
# MAGIC - Transform the needed information in ethereumetl database into the silver delta table needed by your modeling module
# MAGIC - Clearly document using the notation from [lecture](https://learn-us-east-1-prod-fleet02-xythos.content.blackboardcdn.com/5fdd9eaf5f408/8720758?X-Blackboard-Expiration=1650142800000&X-Blackboard-Signature=h%2FZwerNOQMWwPxvtdvr%2FmnTtTlgRvYSRhrDqlEhPS1w%3D&X-Blackboard-Client-Id=152571&response-cache-control=private%2C%20max-age%3D21600&response-content-disposition=inline%3B%20filename%2A%3DUTF-8%27%27Delta%2520Lake%2520Hands%2520On%2520-%2520Introduction%2520Lecture%25204.pdf&response-content-type=application%2Fpdf&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEAAaCXVzLWVhc3QtMSJHMEUCIQDEC48E90xPbpKjvru3nmnTlrRjfSYLpm0weWYSe6yIwwIgJb5RG3yM29XgiM%2BP1fKh%2Bi88nvYD9kJNoBNtbPHvNfAqgwQIqP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARACGgw2MzU1Njc5MjQxODMiDM%2BMXZJ%2BnzG25TzIYCrXAznC%2BAwJP2ee6jaZYITTq07VKW61Y%2Fn10a6V%2FntRiWEXW7LLNftH37h8L5XBsIueV4F4AhT%2Fv6FVmxJwf0KUAJ6Z1LTVpQ0HSIbvwsLHm6Ld8kB6nCm4Ea9hveD9FuaZrgqWEyJgSX7O0bKIof%2FPihEy5xp3D329FR3tpue8vfPfHId2WFTiESp0z0XCu6alefOcw5rxYtI%2Bz9s%2FaJ9OI%2BCVVQ6oBS8tqW7bA7hVbe2ILu0HLJXA2rUSJ0lCF%2B052ScNT7zSV%2FB3s%2FViRS2l1OuThecnoaAJzATzBsm7SpEKmbuJkTLNRN0JD4Y8YrzrB8Ezn%2F5elllu5OsAlg4JasuAh5pPq42BY7VSKL9VK6PxHZ5%2BPQjcoW0ccBdR%2Bvhva13cdFDzW193jAaE1fzn61KW7dKdjva%2BtFYUy6vGlvY4XwTlrUbtSoGE3Gr9cdyCnWM5RMoU0NSqwkucNS%2F6RHZzGlItKf0iPIZXT3zWdUZxhcGuX%2FIIA3DR72srAJznDKj%2FINdUZ2s8p2N2u8UMGW7PiwamRKHtE1q7KDKj0RZfHsIwRCr4ZCIGASw3iQ%2FDuGrHapdJizHvrFMvjbT4ilCquhz4FnS5oSVqpr0TZvDvlGgUGdUI4DCdvOuSBjqlAVCEvFuQakCILbJ6w8WStnBx1BDSsbowIYaGgH0RGc%2B1ukFS4op7aqVyLdK5m6ywLfoFGwtYa5G1P6f3wvVEJO3vyUV16m0QjrFSdaD3Pd49H2yB4SFVu9fgHpdarvXm06kgvX10IfwxTfmYn%2FhTMus0bpXRAswklk2fxJeWNlQF%2FqxEmgQ6j4X6Q8blSAnUD1E8h%2FBMeSz%2F5ycm7aZnkN6h0xkkqQ%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20220416T150000Z&X-Amz-SignedHeaders=host&X-Amz-Expires=21600&X-Amz-Credential=ASIAZH6WM4PLXLBTPKO4%2F20220416%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=321103582bd509ccadb1ed33d679da5ca312f19bcf887b7d63fbbb03babae64c) how your pipeline is structured.
# MAGIC - Your pipeline should be immutable
# MAGIC - Use the starting date widget to limit how much of the historic data in ethereumetl database that your pipeline processes.

# COMMAND ----------

# MAGIC %run ./includes/utilities

# COMMAND ----------

# MAGIC %run ./includes/configuration

# COMMAND ----------

# Grab the global variables
wallet_address,start_date = Utils.create_widgets()
print(wallet_address,start_date)
spark.conf.set('wallet.address',wallet_address)
spark.conf.set('start.date',start_date)

# COMMAND ----------

# MAGIC %md
# MAGIC ## YOUR SOLUTION STARTS HERE...

# COMMAND ----------

#utility functions
def clear_prev_folders():
    dbutils.fs.rm(BASE_DELTA_PATH+"silver/",True) 
    
    dbutils.fs.mkdirs(BASE_DELTA_PATH+"silver/")
    print("successfully deleted the folders")
    

# COMMAND ----------

#DO NOT run this if you don't want to things start over
#make sure we delete all the previous folders before each run
clear_prev_folders()

# COMMAND ----------

#giving each path a unique name

silver_path = BASE_DELTA_PATH+"silver/"
from pyspark.sql.functions import *
from pyspark.sql import functions as f
from pyspark.sql import types as t

#asserting the path 

print(silver_path)



# COMMAND ----------

# MAGIC %md ## Table: ERC20_token_table
# MAGIC Filtered from ethereumetl.token_prices_usd, all tokens in here are unique and ERC20

# COMMAND ----------

#start first by filtering out the unrelated token only take ERC20 tokens from token price usd 
ERC20_token_table = spark.sql("select contract_address, name from ethereumetl.token_prices_usd where asset_platform_id== 'ethereum' ").distinct().cache()

# COMMAND ----------

#we also want the token_transfer table because it has token information
token_transfer_df = spark.sql("select token_address,transaction_hash,to_address from ethereumetl.token_transfers").cache()

# COMMAND ----------

# MAGIC %md ##Table: ERC20_contract
# MAGIC Unique ERC20 contracts, but include non-recommend-worthy tokens

# COMMAND ----------

## this table is used to filter out the token contract address from to address
ERC20_contract = spark.sql("select address from ethereumetl.silver_contracts").cache()

# COMMAND ----------

# MAGIC %md ## Table: bronze_init
# MAGIC bronze_init: token_transfer inner join with ERC20_token_table
# MAGIC 
# MAGIC All tokens and transactions in this table are ERC20, and involves the 1149 tokens that we can recommend 
# MAGIC 
# MAGIC In total we have 900 million token transactions on chain, among them 537() millions transaction involves the 1149 tokens that we can recommend

# COMMAND ----------

bronze_init =token_transfer_df.join(ERC20_token_table, token_transfer_df.token_address== ERC20_token_table.contract_address, "inner").drop("contract_address").distinct().cache()


# COMMAND ----------

# MAGIC %md ## Table: bronze_df_inter
# MAGIC 
# MAGIC bronze_df: left antijoin bronze_init_1 with ERC20_contract on to_address to filter out the contract address.
# MAGIC 
# MAGIC because we only care about user wallet address. 

# COMMAND ----------

# perform left antijoin to filter out the contract address. 
bronze_df = bronze_init.join(ERC20_contract, bronze_init.to_address == ERC20_contract.address, "leftanti")

# COMMAND ----------

bronze_df_plus = bronze_df.groupBy("to_address","token_address").count().withColumnRenamed("to_address","wallet_address")

# COMMAND ----------

unique_wallet = bronze_df_plus.select("wallet_address").distinct().withColumn('new_wallet_id',monotonically_increasing_id().cast(IntegerType())).withColumnRenamed("wallet_address","w_address")

unique_token = bronze_df_plus.select("token_address").distinct().withColumn('new_token_id',monotonically_increasing_id().cast(IntegerType())).withColumnRenamed("token_address","t_address")



# COMMAND ----------

# MAGIC %md ## silver_df is the triplet, with 90 million data

# COMMAND ----------

silver_df_inter = bronze_df_plus.join(unique_wallet, bronze_df_plus.wallet_address == unique_wallet.w_address,"inner").drop("w_address")
silver_df = silver_df_inter.join(unique_token, bronze_df_plus.token_address == unique_token.t_address,"inner").drop("t_address")
silver_df = silver_df.withColumn("count", silver_df["count"].cast(IntegerType()))

# COMMAND ----------

# MAGIC %md ## Saving table where transaction count = 1
# MAGIC 
# MAGIC About 71% of the total data, 52 million

# COMMAND ----------

# from pyspark.sql.functions import col
# silver_count_one = silver_df.filter(col('count') < 2)

# COMMAND ----------

# spark.sql("drop table if exists g08_db.silver_count_one")
# silver_count_one.write.format("delta").saveAsTable("g08_db.silver_count_one")

# COMMAND ----------

# silver_count_one_path = BASE_DELTA_PATH+"silver_count_one/"

# COMMAND ----------

# silver_count_one.write.format('delat').option("mergeSchema", "true").save(silver_count_one_path)

# COMMAND ----------

# MAGIC %md ## Saving top 10 most transact tokens for cold start

# COMMAND ----------

# silver_top_10 = silver_df.sort(col('count').desc())

# COMMAND ----------

# spark.sql("drop table if exists g08_db.silver_top_10")
# silver_top_10.write.format("delta").saveAsTable("g08_db.silver_top_10")

# COMMAND ----------

# silver_top_10_path = BASE_DELTA_PATH+"silver_top_10/"

# COMMAND ----------

# silver_top_10.write.format('delat').option("mergeSchema", "true").save(silver_top_10_path)

# COMMAND ----------

# MAGIC %md ##Saving table where transaction count > 1

# COMMAND ----------

# silver_modeling = silver_df.filter(col('count') > 1)

# COMMAND ----------

# spark.sql("drop table if exists g08_db.silver_modeling")
# silver_modeling.write.format("delta").saveAsTable("g08_db.silver_modeling")

# COMMAND ----------

# silver_modeling_path = BASE_DELTA_PATH+"silver_modeling/"

# COMMAND ----------

# silver_modeling.write.format('delat').option("mergeSchema", "true").save(silver_modeling_path)

# COMMAND ----------

# MAGIC %md ## Table: bronze plus
# MAGIC bronze plus: rename the bronze_table.contract_address to bronze_table.wallet_address

# COMMAND ----------

display(spark.sql("DROP TABLE  IF EXISTS delta_silver"))
 
display(spark.sql("CREATE TABLE flights USING DELTA LOCATION '/mnt/dscc202-datasets/misc/G08/tokenrec/tables/silver/'"))
                  
display(spark.sql("OPTIMIZE delta_silver ZORDER BY (count)"))

# COMMAND ----------

silver_df = spark.read.format('delta').load(silver_path)
#display(bronze_df)

# COMMAND ----------


