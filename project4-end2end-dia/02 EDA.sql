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

-- MAGIC %python
-- MAGIC # Grab the global variables
-- MAGIC wallet_address,start_date = Utils.create_widgets()
-- MAGIC print(wallet_address,start_date)
-- MAGIC spark.conf.set('wallet.address',wallet_address)
-- MAGIC spark.conf.set('start.date',start_date)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q1: What is the maximum block number and date of block in the database

-- COMMAND ----------

-- MAGIC %sql
-- MAGIC SELECT number,to_date(cast(timestamp as TIMESTAMP)) as DATE_ from ethereumetl.blocks
-- MAGIC order by number desc
-- MAGIC limit 1;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC maximum block number is 14044000 and date of block in the database is 2022-01-20

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q2: At what block did the first ERC20 transfer happen?

-- COMMAND ----------

select min(block_number) from g07_db.EDA_erc20;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC The first ERC20 transfer happened at block 913198

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q3: How many ERC20 compatible contracts are there on the blockchain?

-- COMMAND ----------

SELECT COUNT(*) from ethereumetl.silver_contracts
where is_erc20 = True;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC There are 181937 ERC20 compatible contracts.

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC ## Q4: What percentage of transactions are calls to contracts

-- COMMAND ----------

USE ETHEREUMETL;

select (sum(cast((silver_contracts.address is not null) as integer))/count(1))*100 as calls_to_contract_percent 
from transactions 
left join silver_contracts on transactions.to_address = silver_contracts.address;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 45.742 % of transactions are calls to contracts

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q5: What are the top 100 tokens based on transfer count?

-- COMMAND ----------

USE ETHEREUMETL;
select
token_address, count(distinct (transaction_hash)) as transfer_count
from token_transfers
group by token_address
order by transfer_count desc
limit 100;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q6: What fraction of ERC-20 transfers are sent to new addresses
-- MAGIC (i.e. addresses that have a transfer count of 1 meaning there are no other transfers to this address for this token this is the first)

-- COMMAND ----------

select count(distinct concat(token_address, to_address))/count(token_address) as fraction_value
from g07_db.EDA_erc20;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Approximately 3/10 ERC-20 transfers are sent to new addresses

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q7: In what order are transactions included in a block in relation to their gas price?
-- MAGIC - hint: find a block with multiple transactions 

-- COMMAND ----------

select number,transaction_count from ethereumetl.blocks
where transaction_count > 1
and start_block in (select distinct start_block from transactions)
order by transaction_count desc
limit 5;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC From the above cell's output, we do have the block numbers where there were more than one transactions. Now we will randomly choose a block and look for relation between transaction and gas price.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC spark.sql("show partitions transactions").show(100000,truncate=False)

-- COMMAND ----------

SELECT  block_number, transaction_index, gas_price 
FROM transactions 
WHERE start_block >=13710100 and end_block <= 13873035 and block_number in (13782464,13644381,13701572)
;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC From the output cell, we can see that transactions are ordered in decreasing gas price for the same block number.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q8: What was the highest transaction throughput in transactions per second?
-- MAGIC hint: assume 15 second block time

-- COMMAND ----------

select max(transaction_count)/15 as highest_transaction_throughput from ethereumetl.blocks

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Highest transaction throughput in transactions per second was 95.4

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q9: What is the total Ether volume?
-- MAGIC Note: 1x10^18 wei to 1 eth and value in the transaction table is in wei

-- COMMAND ----------

select sum(value)/power(10,18) as TOTAL_ETHER_VOL from ethereumetl.transactions

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Total Ether Volume is : 4819303728.848482

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q10: What is the total gas used in all transactions?

-- COMMAND ----------

select sum(gas) as Total_gas_used
from ethereumetl.transactions;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC The Total Gas Used in all transactions is : 26962124687146 

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q11: Maximum ERC-20 transfers in a single transaction

-- COMMAND ----------

select transaction_hash, count(*) from g07_db.eda_erc20 group by transaction_hash
order by count(*) desc;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Max number of ERC-20 tokens in a transaction is 3001 

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q12: Token balance for any address on any date?

-- COMMAND ----------

select token_address, sum(case when from_address = '${wallet.address}' then -1*value else value end) as token_balance
from ethereumetl.token_transfers as table_1
inner join ethereumetl.blocks table_2 on
table_2.start_block = table_1.start_block and 
table_2.end_block = table_1.end_block and
table_2.number = table_1.block_number and
to_date(cast(table_2.timestamp as TIMESTAMP)) <= '${start.date}' and
(from_address = '${wallet.address}' or to_address = '${wallet.address}')
group by token_address
order by token_balance ;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 13.Viz the transaction count over time (network use)

-- COMMAND ----------

select to_date(CAST(timestamp as timestamp)) as Date,sum(transaction_count) as trans_count_per_day from blocks
group by date
order by date desc

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 14.Viz ERC-20 transfer count over time
-- MAGIC interesting note: https://blog.ins.world/insp-ins-promo-token-mixup-clarified-d67ef20876a3

-- COMMAND ----------

-- TBD
use ethereumetl;

select date, sum(transfer_count) as trans_count_per_day
 from(select start_block, end_block, number, to_date(CAST(timestamp AS timestamp)) as Date from blocks) as table_1 
 left join
 (select start_block, end_block, block_number, count(distinct transaction_hash) as transfer_count from g07_db.eda_erc20
   group by start_block, end_block, block_number) as table_2
 on table_1.start_block = table_2.start_block 
 and table_1.end_block = table_2.end_block 
 and table_1.number = table_2.block_number
 group by date
 order by date desc;

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # Return Success
-- MAGIC dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
