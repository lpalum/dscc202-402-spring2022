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
    dbutils.fs.rm(BASE_DELTA_PATH+"/bronze/", True)
    dbutils.fs.rm(BASE_DELTA_PATH+"/bronze_plus/",True)
    dbutils.fs.rm(BASE_DELTA_PATH+"/silver/",True)
    dbutils.fs.rm(BASE_DELTA_PATH+"/silver_plus/",True)
    
    
    dbutils.fs.mkdirs(BASE_DELTA_PATH+"/bronze/")
    dbutils.fs.mkdirs(BASE_DELTA_PATH+"/bronze_plus/")
    dbutils.fs.mkdirs(BASE_DELTA_PATH+"/silver/")
    dbutils.fs.mkdirs(BASE_DELTA_PATH+"/silver_plus/")
    print("successfully deleted the folders")
        

# COMMAND ----------

#DO NOT run this if you don't want to things start over
#make sure we delete all the previous folders before each run


clear_prev_folders()

# COMMAND ----------

#giving each path a unique name
bronze_path = BASE_DELTA_PATH+"/bronze/"

bronze_plus_path = BASE_DELTA_PATH+"/bronze_plus/"

silver_path = BASE_DELTA_PATH+"/silver/"

silver_plus_path = BASE_DELTA_PATH+"/silver_plus/"

#asserting the path 
print(bronze_path)

print(bronze_plus_path)

print(silver_path)

print(silver_plus_path)


# COMMAND ----------

# MAGIC %md ## Table: ERC20_token_table
# MAGIC Filtered from ethereumetl.token_prices_usd, all tokens in here are unique and ERC20

# COMMAND ----------

#start first by filtering out the unrelated token only take ERC20 tokens from token price usd 
ERC20_token_table = spark.sql("select contract_address, name from ethereumetl.token_prices_usd where asset_platform_id== 'ethereum' ").distinct()

# COMMAND ----------

ERC20_token_table.count()

# COMMAND ----------




#we want to get transaction information
transaction_df = spark.sql("select to_address, value, block_hash, hash as transact_hash from ethereumetl.transactions")
#we also want the token_transfer table because it has token information
token_transfer_df = spark.sql("select token_address,transaction_hash from ethereumetl.token_transfers")




# COMMAND ----------

token_transfer_df.count()

# COMMAND ----------

# MAGIC %md ## Table: block_short
# MAGIC unique block hash with timestamp

# COMMAND ----------

#block_short gives you timestamp of each transaction ???
block_short = spark.sql("select hash, timestamp from g08_db.blocks_ts_clean")

# COMMAND ----------

# MAGIC %md ##Table: ERC20_contract
# MAGIC Unique ERC20 contracts, but include non-recommend-worthy tokens

# COMMAND ----------


## this table is used to filter out the token contract address from to address
ERC20_contract = spark.sql("select address from ethereumetl.silver_contracts where is_erc20==True")

# COMMAND ----------

ERC20_contract.count()

# COMMAND ----------

# MAGIC %md ## CREATING BRONZE TABLE...
# MAGIC bronze table: we only take to_address from the token transfer that are personal wallet but not smart contracts and filter out the non ERC20 tokens

# COMMAND ----------

# MAGIC %md ## Table: bronze_init
# MAGIC bronze_init: token_transfer inner join with ERC20_token_table
# MAGIC 
# MAGIC All tokens and transactions in this table are ERC20, and involves the 1149 tokens that we can recommend 
# MAGIC 
# MAGIC In total we have 900 million token transactions on chain, among them 537() millions transaction involves the 1149 tokens that we can recommend

# COMMAND ----------

bronze_init =token_transfer_df.join(ERC20_token_table, token_transfer_df.token_address== ERC20_token_table.contract_address, "inner").drop("contract_address").distinct()

# COMMAND ----------

bronze_init.count()
# which means we have 537 million ERC20 transactions

# COMMAND ----------

# MAGIC %md ##Table: bronze_init_1
# MAGIC bronze_init_1: bronze_init inner join transaction_df on transaction hash to get block hash.
# MAGIC 
# MAGIC Assigned block hash to each transaction_hash
# MAGIC 
# MAGIC Compare with 537 million ERC20 transactions, now we only have 56 million transactions, which 
# MAGIC means 90% ERC20 token transactions are not recorded in transaction table 
# MAGIC 
# MAGIC Hypothesis: 
# MAGIC token_transfer recorded many transaction during 2018-2021, however, 
# MAGIC transaction table did not record that time period

# COMMAND ----------

bronze_init_1 = bronze_init.join(transaction_df, bronze_init.transaction_hash == transaction_df.transact_hash).drop("transact_hash")


# COMMAND ----------

bronze_init_1.count()


# COMMAND ----------

# MAGIC %md ## Table: bronze_df
# MAGIC 
# MAGIC bronze_df: left antijoin bronze_init_1 with ERC20_contract on to_address to filter out the contract address.
# MAGIC 
# MAGIC because we only care about user wallet address. 

# COMMAND ----------

# perform left antijoin to filter out the contract address. 

bronze_df = bronze_init_1.join(ERC20_contract, bronze_init_1.to_address == ERC20_contract.address, "leftanti").select("*").where(col("value")>0)


# COMMAND ----------

#store the dataframe into the bronze_path
print(bronze_df.count())

# COMMAND ----------

display(bronze_df)

# COMMAND ----------

#write bronze_df into the bronze_path

bronze_df.write.format('delta').save(bronze_path)

# COMMAND ----------

# bronze_df.write.format('delta').saveAsTable("g08_db.bronze_table")

# COMMAND ----------

# MAGIC %md ## Table: bronze plus
# MAGIC bronze plus: rename the bronze_table.contract_address to bronze_table.wallet_address

# COMMAND ----------

bronze_df = spark.read.format('delta').load(bronze_path)
display(bronze_df)

# COMMAND ----------

bronze_df.count()

# COMMAND ----------

bronze_plus_df = bronze_df.withColumnRenamed("to_address","wallet_address").withColumnRenamed("value","wai_value")

# COMMAND ----------


bronze_plus_df.write.format('delta').save(bronze_plus_path)

# COMMAND ----------

# bronze_plus_df.write.format('delta').saveAsTable("g08_db.bronze_plus_table")

# COMMAND ----------

# MAGIC %md ## Table: token_transaction_rank
# MAGIC rank most popular token based on # of transaction on the chain

# COMMAND ----------

## to deal with the cold start problem, I ranked the most used token for all ERC20 token transactions
from pyspark.sql.functions import col
token_transaction_rank = bronze_plus_df.groupBy("name","token_address").count().sort(col("count").desc())

# COMMAND ----------

#token_transaction_rank.write.format('delta').saveAsTable("g08_db.token_transaction_rank")

# COMMAND ----------

# MAGIC %md ## Table: silver_df
# MAGIC we need to add dates into our bronze table. 

# COMMAND ----------

bronze_plus_df = spark.read.format('delta').load(bronze_plus_path)
display(bronze_plus_df)

# COMMAND ----------

bronze_plus_df.join(block_short,bronze_plus_df.block_hash==block_short.hash,"inner").drop('hash').write.format('delta').save(silver_path)

# COMMAND ----------

silver_df = spark.read.format("delta").load(silver_path)


# COMMAND ----------

#silver_df.write.format('delta').saveAsTable("g08_db.silver_table")


# COMMAND ----------

display(silver_df)

# COMMAND ----------

# MAGIC %md ## Table: silver_plus_table
# MAGIC 
# MAGIC encode wallet_address and token_address to long 
# MAGIC 
# MAGIC added col new_wallet_id and new_token_id
# MAGIC 
# MAGIC data from 2016 to 2022 (with 2017-2021 missing)

# COMMAND ----------

unique_wallet = silver_df.select("wallet_address").distinct().withColumn('new_wallet_id',monotonically_increasing_id()).withColumnRenamed("wallet_address","w_address")
display(unique_wallet)
unique_token = silver_df.select("token_address").distinct().withColumn('new_token_id',monotonically_increasing_id()).withColumnRenamed("token_address","t_address")
display(unique_token)

# COMMAND ----------

silver_plus_table_inter = silver_table.join(unique_wallet, silver_table.wallet_address==unique_wallet.w_address,"inner").drop("w_address")
silver_plus_table = silver_plus_table_inter.join(unique_token, silver_plus_table_inter.token_address == unique_token.t_address, "inner").drop("t_address")
display(silver_plus_table)

# COMMAND ----------

silver_plus_table.write.format("delta").save(silver_plus_path)


# COMMAND ----------

silver_plus_table.write.format("delta").saveAsTable("g08_db.silver_plus_table")

# COMMAND ----------

# MAGIC %md ## Table: silver_2022
# MAGIC silver data after 2022-01-01

# COMMAND ----------

silver_2022 = silver_plus_table.select("*").where(col("timestamp")>start_date)
print(new_silver.count())

# COMMAND ----------

silver_2022.write.format("delta").saveAsTable("g08_db.silver_2022_01_01")
