-- Databricks notebook source
-- MAGIC %md
-- MAGIC ## Ethereum Blockchain Data Analysis - <a href=https://github.com/blockchain-etl/ethereum-etl-airflow/tree/master/dags/resources/stages/raw/schemas>Table Schemas</a>
-- MAGIC - **Transactions** - Each block in the blockchain is composed of zero or more transactions. Each transaction has a source address, a target address, an amount of Ether transferred, and an array of input bytes. This table contains a set of all transactions from all blocks, and contains a block identifier to get associated block-specific information associated with each transaction.
-- MAGIC - **Blocks** - The Ethereum blockchain is composed of a series of blocks. This table contains a set of all blocks in the blockchain and their attributes.
-- MAGIC - **Receipts** - the cost of gas for specific transactions.
-- MAGIC - **Traces** - The trace module is for getting a deeper insight into transaction processing. Traces exported using <a href=https://openethereum.github.io/JSONRPC-trace-module.html>Parity trace module</a>
-- MAGIC - **Tokens** - Token data including contract address and symbol information.
-- MAGIC - **Token Transfers** - The most popular type of transaction on the Ethereum blockchain invokes a contract of type ERC20 to perform a transfer operation, moving some number of tokens from one 20-byte address to another 20-byte address. This table contains the subset of those transactions and has further processed and denormalized the data to make it easier to consume for analysis of token transfer events.
-- MAGIC - **Contracts** - Some transactions create smart contracts from their input bytes, and this smart contract is stored at a particular 20-byte address. This table contains a subset of Ethereum addresses that contain contract byte-code, as well as some basic analysis of that byte-code. 
-- MAGIC - **Logs** - Similar to the token_transfers table, the logs table contains data for smart contract events. However, it contains all log data, not only ERC20 token transfers. This table is generally useful for reporting on any logged event type on the Ethereum blockchain.
-- MAGIC 
-- MAGIC ### Rubric for this module
-- MAGIC Answer the quetions listed below.

-- COMMAND ----------

-- MAGIC %run ./includes/utilities

-- COMMAND ----------

-- MAGIC %run ./includes/configuration

-- COMMAND ----------

use g08_db;
show tables;

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from pyspark.sql.functions import *
-- MAGIC from pyspark.sql import functions as f
-- MAGIC from pyspark.sql import types as t

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q1: What is the maximum block number and date of block in the database

-- COMMAND ----------

-- #EDA 1.1
-- %python
-- blocks = spark.sql("select * from ethereumetl.blocks")
-- display(blocks.sort(col('number').desc()))

-- #EDA 1.2
-- %python
-- from pyspark.sql.functions import *
-- from pyspark.sql import functions as f
-- from pyspark.sql import types as t
-- from pyspark.sql.functions import *
-- from pyspark.sql.functions import to_date
-- # token_prices_usd = spark.sql("select * from ethereumetl.token_prices_usd")
-- cols_drop = ['nonce', 'parent_hash', 'sha3_uncles', 'logs_bloom', 'transactions_root', 'state_root', 'receipts_root', 'miner', 'difficulty', 'total_difficulty', 'extra_data', 'size', 'gas_limit', 'base_fee_per_gas']
-- blocks_clean = blocks.drop(*cols_drop)
-- # blocks_ts_clean = blocks_clean.select(
-- #     from_unixtime(col("timestamp")).alias("timestamp1")
-- # )
-- blocks_clean = blocks_clean.withColumn('timestamp', f.date_format(blocks_clean.timestamp.cast(dataType=t.TimestampType()), "yyyy-MM-dd"))
-- blocks_ts_clean = blocks_clean.withColumn('timestamp', f.to_date(blocks_clean.timestamp.cast(dataType=t.TimestampType())))

-- %python
-- display(blocks_ts_clean.sort(col('timestamp').desc()))

-- COMMAND ----------

-- MAGIC %python
-- MAGIC blocks_ts_clean = spark.sql("select * from g08_db.blocks_ts_clean")
-- MAGIC display(blocks_ts_clean.sort(col('timestamp').desc()))
-- MAGIC 
-- MAGIC print(blocks_ts_clean.agg(max("number")).collect()[0][0])

-- COMMAND ----------

-- MAGIC %md
-- MAGIC The Maximum block number is 14044000, date of block is 2022-01-20

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q2: At what block did the first ERC20 transfer happen?

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # token_transfer = spark.sql("select transaction_hash from ethereumetl.token_transfers")
-- MAGIC # token_transaction = spark.sql("select hash,block_hash from ethereumetl.transactions")
-- MAGIC # blocks_date  = blocks_ts_clean.select("hash","number","timestamp")
-- MAGIC 
-- MAGIC # q2_table_inter = token_transfer.join(token_transaction,token_transfer.transaction_hash==token_transaction.hash,"inner").drop("hash")
-- MAGIC # q2_table = q2_table_inter.join(blocks_date,q2_table_inter.block_hash==blocks_date.hash,"inner").drop("hash")
-- MAGIC # q2_table.write.format('delta').saveAsTable("g08_db.q2_table")

-- COMMAND ----------

-- MAGIC %python
-- MAGIC q2_table = spark.sql("select * from g08_db.q2_table")
-- MAGIC display(q2_table.sort(col("timestamp").asc()))

-- COMMAND ----------

-- MAGIC %md
-- MAGIC first ERC transfer happens on October 27th 2015

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q3: How many ERC20 compatible contracts are there on the blockchain?

-- COMMAND ----------

-- MAGIC %python
-- MAGIC 
-- MAGIC silver_contracts = spark.sql("select * from g08_db.silver_erc20_contracts where is_erc20 = True" )
-- MAGIC silver_contracts.dropDuplicates()
-- MAGIC print(silver_contracts.count())

-- COMMAND ----------

-- MAGIC %md
-- MAGIC there are 181937 ERC20 compatible contracts on blockchain

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC ## Q4: What percentage of transactions are calls to contracts

-- COMMAND ----------

-- MAGIC %python
-- MAGIC contracts = spark.sql("select address from ethereumetl.silver_contracts").distinct()
-- MAGIC transactions = spark.sql('select hash, to_address from ethereumetl.transactions')
-- MAGIC trns_contrs_inner = transactions.join(contracts, transactions.to_address == contracts.address, "inner")
-- MAGIC pct_call_to_contracts = 100*trns_contrs_inner.distinct().count()/transactions.distinct().count()
-- MAGIC print(f"{pct_call_to_contracts}% transactions are calls to contracts")

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 45.74135958458722% transactions are calls to contracts

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q5: What are the top 100 tokens based on transfer count?

-- COMMAND ----------

-- MAGIC %python
-- MAGIC token_transfers = spark.sql('select * from ethereumetl.token_transfers')
-- MAGIC transfer_count = token_transfers.groupBy("token_address").count()
-- MAGIC tokens = spark.sql('select * from ethereumetl.tokens')
-- MAGIC sorted_transfer_count = transfer_count.sort(col("count").desc())
-- MAGIC sorted_token_count = sorted_transfer_count.join(tokens, sorted_transfer_count.token_address == tokens.address, "inner").select("token_address", "symbol", "count").distinct()
-- MAGIC sorted_token_count = sorted_token_count.sort(col("count").desc()).na.drop("all")
-- MAGIC display(sorted_token_count)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q6: What fraction of ERC-20 transfers are sent to new addresses
-- MAGIC (i.e. addresses that have a transfer count of 1 meaning there are no other transfers to this address for this token this is the first)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # token_transfers = spark.sql('select * from ethereumetl.token_transfers')
-- MAGIC # unique_transfers_count = token_transfers.distinct().count()
-- MAGIC # token_count = token_transfers.groupBy("token_address", "to_address").count().filter(col('count')==1).count()
-- MAGIC # percentage = token_count/unique_transfers
-- MAGIC # print(percentage)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # take all observations from token transfer
-- MAGIC token_transfers = spark.sql('select * from ethereumetl.token_transfers')
-- MAGIC #for demoninator we only interested in unique token transfer
-- MAGIC unique_transfers = token_transfers.distinct().count()
-- MAGIC #number of address only have 1 transaction. 
-- MAGIC token_count = token_transfers.groupBy("to_address").count().filter(col('count')==1).count()
-- MAGIC #total transaction 
-- MAGIC percentage = token_count/unique_transfers
-- MAGIC print(percentage)
-- MAGIC # print(f"{100*transfer_is_1_count/token_transfers.distinct().count()}%")

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC 6.52785% of the ERC20 transfers are sent to new addresses

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q7: In what order are transactions included in a block in relation to their gas price?
-- MAGIC - hint: find a block with multiple transactions 

-- COMMAND ----------

-- MAGIC %python
-- MAGIC transaction = spark.sql("select * from ethereumetl.transactions")
-- MAGIC display(transaction.filter("block_number == 13856856"))

-- COMMAND ----------

-- MAGIC %md
-- MAGIC It looks like the transaction is ordered by the gas_price in a descending order 

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q8: What was the highest transaction throughput in transactions per second?
-- MAGIC hint: assume 15 second block time

-- COMMAND ----------

-- MAGIC %python
-- MAGIC transaction = spark.sql("select * from ethereumetl.blocks")
-- MAGIC rate = transaction.groupBy("number").sum("transaction_count")

-- COMMAND ----------

-- MAGIC %python
-- MAGIC rate = rate.sort(desc("sum(transaction_count)"))
-- MAGIC display(rate)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC since the highest transaction block has transaction 1431, then the highest transaction per second is 95.4  

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q9: What is the total Ether volume?
-- MAGIC Note: 1x10^18 wei to 1 eth and value in the transaction table is in wei

-- COMMAND ----------

-- MAGIC %python
-- MAGIC transaction = spark.sql("select * from ethereumetl.transactions")
-- MAGIC display(transaction)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from pyspark.sql.functions import  sum
-- MAGIC total_value = transaction.select(sum('value')/(10**18)).collect()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC total_value[0]

-- COMMAND ----------

-- MAGIC %md
-- MAGIC the total Ether volume is 4819303728.848482

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q10: What is the total gas used in all transactions?

-- COMMAND ----------

-- MAGIC %python
-- MAGIC transaction = spark.sql("select * from ethereumetl.transactions")
-- MAGIC 
-- MAGIC #display(transaction.filter("block_number == 3300360"))
-- MAGIC 
-- MAGIC import pyspark.sql.functions as F     
-- MAGIC 
-- MAGIC transaction.agg(F.sum("gas")).collect()[0][0]

-- COMMAND ----------

-- MAGIC %md
-- MAGIC total_gas is 26962124687146

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q11: Maximum ERC-20 transfers in a single transaction

-- COMMAND ----------

-- MAGIC %python
-- MAGIC 
-- MAGIC q11_df = spark.sql("select * from g08_db.silver_transaction_token_transfer_inner")
-- MAGIC q11_df = q11_df.sort(col("transaction_value").desc())
-- MAGIC display(q11_df)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Maximum ERC_20 transfers in a single transaction is: 112000000000000000000000 or 112000 ether

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q12: Token balance for any address on any date?

-- COMMAND ----------

-- MAGIC %python
-- MAGIC #first I select hash feature from block_ts_clean table and rename it as hash_block, then I selected timestamp from block_ts_clean
-- MAGIC #then I created a dataframe out of those two selected features.
-- MAGIC 
-- MAGIC # short_block = spark.sql("select hash as hash_block,timestamp from g08_db.blocks_ts_clean ")
-- MAGIC # display(short_block)
-- MAGIC 
-- MAGIC #for transaction table from ethereumetl, i selected block_hash,to_address,from address, and value and created a dataframe out of it 
-- MAGIC #and name it short transaction.
-- MAGIC 
-- MAGIC # short_transaction = spark.sql("select block_hash,to_address,from_address,value from ethereumetl.transactions")
-- MAGIC # display(short_transaction)
-- MAGIC 
-- MAGIC #those two dataframe is what we need for this question
-- MAGIC 
-- MAGIC # i used a inner join on block hash to connect the two dataframe together then name the new dataframe q12_full 
-- MAGIC 
-- MAGIC # q12_full = short_block.join(short_transaction,short_transaction.block_hash ==  short_block.hash_block,"inner")
-- MAGIC 
-- MAGIC #i then created a dataframe called sent balance which keep track of how much token value an address sent to other address in a given day. 
-- MAGIC 
-- MAGIC # since the senting address lost tokens, the value should be negative 
-- MAGIC 
-- MAGIC # sent_balance = q12_full.groupBy("timestamp","from_address").sum("value")
-- MAGIC # sent_balance = sent_balance.withColumnRenamed("sum(value)","balance")
-- MAGIC # sent_balance = sent_balance.withColumnRenamed("from_address","address")
-- MAGIC # sent_balance = sent_balance.withColumn("balance", sent_balance.balance*-1)
-- MAGIC 
-- MAGIC #i then created a dataframe called recieve balance which keep track of how much token value an address recieve from other address in a given day. 
-- MAGIC # since the recieving address get tokens, the value should be positive 
-- MAGIC 
-- MAGIC # recive_balance = q12_full.groupBy("timestamp","to_address").sum("value")
-- MAGIC # recive_balance = recive_balance.withColumnRenamed("sum(value)","balance")
-- MAGIC # recive_balance = recive_balance.withColumnRenamed("to_address","address")
-- MAGIC # recive_balance.withColumn("balance", recive_balance.balance)
-- MAGIC 
-- MAGIC 
-- MAGIC #then added up the positive value(how much value an address recieve) and the negative value(how much value an address sent) on a given day to get the balance for that specific address. 
-- MAGIC 
-- MAGIC # balance_df =recive_balance.union(sent_balance).groupBy("timestamp","address").sum("balance").withColumnRenamed("sum(balance)","balance_wei")
-- MAGIC # display(balance_df)
-- MAGIC 
-- MAGIC #change to ether
-- MAGIC 
-- MAGIC # balance_df = balance_df.withColumn("balance_ether", balance_df.balance_wei/(10**18))
-- MAGIC # display(balance_df)
-- MAGIC 
-- MAGIC # balance_df.write.saveAsTable("g08_db.q12_table")

-- COMMAND ----------

-- MAGIC %python
-- MAGIC balance_df = spark.sql("select * from g08_db.q12_table")
-- MAGIC display(balance_df)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q13 Viz the transaction count over time (network use)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC #In order to find the transaction count on a given day. we need the following tables: blocks_ts_clean from our group database. 
-- MAGIC # and transaction table from ethereumetl. I only picked the most relavant features from both tables to avoid distraction.
-- MAGIC # i made two new table short_block for and short_transaction 
-- MAGIC # short_block = spark.sql("select hash as hash_block,timestamp from g08_db.blocks_ts_clean ")
-- MAGIC # display(short_block)
-- MAGIC # short_transaction = spark.sql("select block_hash,to_address,from_address,value from ethereumetl.transactions")
-- MAGIC # display(short_transaction)
-- MAGIC # i then inner join the two tables on block hash 
-- MAGIC # q13_full = short_block.join(short_transaction,short_transaction.block_hash ==  short_block.hash_block,"inner")

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # I group by the timestamp and count the transactions. this will give the total number of 
-- MAGIC # transaction on a given day, store the new dataframe as q13_table 
-- MAGIC # q13_table = q13_full.groupBy("timestamp").count()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from pyspark.sql.functions import col
-- MAGIC # I then sorted the table in ascending order which put earlier dates in the front. 
-- MAGIC # then I displayed it as barplot. 
-- MAGIC blocks = spark.sql("select * from ethereumetl.blocks where timestamp>1525656044")
-- MAGIC blocks_clean = blocks.withColumn('timestamp_convert', f.date_format(blocks.timestamp.cast(dataType=t.TimestampType()), "yyyy-MM-dd"))
-- MAGIC blocks_clean.drop('timestamp')
-- MAGIC blocks_clean = blocks_clean.groupBy('timestamp_convert').sum('transaction_count')
-- MAGIC q13_table = blocks_clean.sort(col("timestamp_convert").asc())
-- MAGIC 
-- MAGIC 
-- MAGIC # blocks_clean = blocks_clean.withColumn('timestamp', f.date_format(blocks_clean.timestamp.cast(dataType=t.TimestampType()), "yyyy-MM-dd"))
-- MAGIC # blocks_ts_clean = blocks_clean.withColumn('timestamp', f.to_date(blocks_clean.timestamp.cast(dataType=t.TimestampType())))

-- COMMAND ----------

-- MAGIC %python
-- MAGIC display(q13_table)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q14 Viz ERC-20 transfer count over time
-- MAGIC interesting note: https://blog.ins.world/insp-ins-promo-token-mixup-clarified-d67ef20876a3

-- COMMAND ----------

-- MAGIC %python
-- MAGIC #for question 14 we need one extra table: token transfer because token transfer has the ERC-20 token transfer
-- MAGIC # Again I selected the most relevant features for each table and store them into dataframes
-- MAGIC # token_transfer table => transfer_df, transaction table =>transaction_df, block table => short_block
-- MAGIC 
-- MAGIC 
-- MAGIC transfer_df = spark.sql("select transaction_hash from ethereumetl.token_transfers")
-- MAGIC transaction_df = spark.sql("select hash,block_hash from ethereumetl.transactions")
-- MAGIC short_block = spark.sql("select hash as hash_block,timestamp,transaction_count from ethereumetl.blocks where timestamp>1525656044")
-- MAGIC 
-- MAGIC transaction_date = transaction_df.join(short_block, transaction_df.block_hash==short_block.hash_block, "inner").drop("hash_block")
-- MAGIC transaction_date_transfer = transaction_date.join(transfer_df, transaction_date.hash==transfer_df.transaction_hash, "inner").drop("hash")
-- MAGIC transaction_date_transfer = transaction_date_transfer.withColumn('timestamp_convert', f.date_format(transaction_date_transfer.timestamp.cast(dataType=t.TimestampType()), "yyyy-MM-dd"))
-- MAGIC transaction_date_transfer = transaction_date_transfer.groupBy('timestamp_convert').count()
-- MAGIC q14_table = transaction_date_transfer.sort(col("timestamp_convert").asc())

-- COMMAND ----------

-- MAGIC %python
-- MAGIC #you then display it
-- MAGIC display(q14_table)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # Return Success
-- MAGIC dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
