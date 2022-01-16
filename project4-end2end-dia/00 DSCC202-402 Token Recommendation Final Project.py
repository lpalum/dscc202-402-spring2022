# Databricks notebook source
# MAGIC %md
# MAGIC ## Description
# MAGIC In project you and your group will be developing an end to end Data Intensive Application (DIA) that receommends ERC-20 Tokens for a given user address (wallet) on the Ethereum blockchain.
# MAGIC 
# MAGIC Consider this illustration of the ethereum network as well as the familiar process and tech stack that we have covered in this course.
# MAGIC <table border=0>
# MAGIC   <tr><td><h2>Application</h2></td><td><h2>Process</h2></td></tr>
# MAGIC   <tr><td>![Image]()</td><td><img src='https://data-science-at-scale.s3.amazonaws.com/images/DIA+Framework-DIA+Process+-+1.png' width=680></td></tr>
# MAGIC </table>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Databricks Resources and Naming Conventions
# MAGIC 
# MAGIC - Each group has been assigned a specific Databricks Spark Cluster named **dscc202-groupxx-cluster** (all provisioned the same) with 1 driver node and up to 8 workers.
# MAGIC - Each group has a specific AWS S3 bucket they will mount to store their project tables and model artifacts.  The project starting archive has code to do this. **/mnt/dscc202-groupxx-datasets**
# MAGIC - Each group has a pre-provisioned database **dscc202_groupxx_db** that they should use for all of their hive metastore tables.
# MAGIC - Each group should create a specific model in the mlflow registry **groupxx_model** that they should use for their project.
# MAGIC - Each group should create a specific MLflow experiment for all of their project runs and the model artifacts should be stored in their group specific bucket as well. **dscc202_groupxx_experiment**
# MAGIC 
# MAGIC ![Image](https://data-science-at-scale.s3.amazonaws.com/images/Flight+Project+Resources.png)
# MAGIC 
# MAGIC **IMPORTANT**: See the configuration notebook under **includes** to set your group designation

# COMMAND ----------

# DBTITLE 0,Project Structure
# MAGIC %md
# MAGIC ## Project Structure
# MAGIC Each group is expected to divide their work among a set of notebooks within the Databricks workspace.  A group specific project archive should be setup in github and each group member can work on their specific branch of that repository and then explicitly merge their work into the “master” project branch when appropriate. (see the class notes on how to do this).  The following illustration highlights the recommended project structure.  This approach should make it fairly straight forward to coordinate group participation and work on independent pieces of the project while having a well identified way of integrating those pieces into a whole application that meets the requirements specified for the project.
# MAGIC 
# MAGIC ![Image](https://data-science-at-scale.s3.amazonaws.com/images/Flight+Project+Structure.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Grading
# MAGIC **Project is Due no later than May 6th 2022**
# MAGIC <p>Each student in a given group should participate in the design and development of the application.  
# MAGIC The group should coordinate and divide up the responsibilities needed to complete the project.  Each group should submit a link to their GitHub repositiory for their project via blackboard.
# MAGIC </p>
# MAGIC 
# MAGIC #### Points Allocation
# MAGIC - Group - Extract Tansform and Load (ETL) - 10 points
# MAGIC - Group - Exploratory Data Analysis (EDA) - 10 points
# MAGIC - Group - Modeling - 10 points
# MAGIC - Group - Application - 10 points
# MAGIC 
# MAGIC Total of 40 points.  Good luck and have fun!

# COMMAND ----------

# MAGIC %run ./includes/utilities

# COMMAND ----------

# MAGIC %run ./includes/configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ### Application Widgets
# MAGIC - Airport Code - dropdown list to select the airport of interest.
# MAGIC - Training Start Date - when the training should start (note 6 years of data in the archive)
# MAGIC - Training End Date - when the training should end
# MAGIC - Inference Date - this will be set to 1 day after the training end date although any date after training can be entered.

# COMMAND ----------

training_start_date, training_end_date, inference_date, airport_code = Utils.create_widgets()
print(training_start_date, training_end_date, inference_date, airport_code)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extract Transform and Load (ETL)
# MAGIC - Bronze data sources of airline data and weather data are already in the dscc202_db.
# MAGIC - Document your pipeline according to the object and class notation that we used in the Moovio example.
# MAGIC - Feature engineering should be well documented.  E.g. what transformations are being employed to form the Silver data from the Bronze data.
# MAGIC - Schema validation and migration should be included in the Bronze to Silver transformation.
# MAGIC - Optimization with partitioning and Z-ordering should be appropriately employed.
# MAGIC - Streaming construct should be employed so any data added to Bronze data sources will be automatically ingested by your application by running the ELT code.
# MAGIC - ELT code should be idempotent.  No adverse effects for multiple runs.

# COMMAND ----------

# run link to the ETL notebook
result_etl = dbutils.notebook.run("01 ETL", 7200, {"00.Airport_Code":airport_code,"01.training_start_date":training_start_date,"02.training_end_date":training_end_date,"03.inference_date":inference_date})

# Check for success
#assert json.loads(result_etl)["exit_code"] == "OK", "ETL Failed!" # Check to see that it worked

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exploratory Data Analysis (EDA)
# MAGIC - Follow the guidelines in [Practical Advice for the analysis of large data](https://www.unofficialgoogledatascience.com/2016/10/practical-advice-for-analysis-of-large.html)
# MAGIC - Clear communication of findings and filtering of Bronze/Silver data.
# MAGIC - Pandas Profiling and Tensorflow Data Validation libraries can be very helpful here... hint!

# COMMAND ----------

# run link to the EDA notebook
result_eda = dbutils.notebook.run("02 EDA", 7200, {"00.Airport_Code":airport_code,"01.training_start_date":training_start_date,"02.training_end_date":training_end_date,"03.inference_date":inference_date})

# Check for success
assert json.loads(result_eda)["exit_code"] == "OK", "EDA Failed!" # Check to see that it worked

# COMMAND ----------

# MAGIC %md
# MAGIC ### MLops Lifecycle
# MAGIC - Use the training start and end date widgets to specify the date range to be used in the training and test of a given model.
# MAGIC - Training model(s) at scale to estimate the arrival or departure time of a flight before it leaves the gate.
# MAGIC - Register training and test data versions as well as parameters and metrics using mlflow
# MAGIC - Including model signature in the published model
# MAGIC - Hyperparameter tuning at scale with mlflow comparison of performance
# MAGIC - Orchestrating workflow staging to production using clear test methods
# MAGIC - Parameterize the date range that a given model is covering in its training set.

# COMMAND ----------

# run link to the modeling notebook
result_mlops = dbutils.notebook.run("04 Modeling", 0, {"00.Airport_Code":airport_code,"01.training_start_date":training_start_date,"02.training_end_date":training_end_date,"03.inference_date":inference_date})

# Check for success
assert json.loads(result_mlops)["exit_code"] == "OK", "Modeling Failed!" # Check to see that it worked

# COMMAND ----------

# MAGIC %md
# MAGIC ## The Token Recommender
# MAGIC ![Image](https://data-science-at-scale.s3.amazonaws.com/images/Flight+Application+v2.png)
# MAGIC 
# MAGIC Your application will focus on 13 of the busiest airports in the US.  In particular, 
# MAGIC - JFK - INTERNATIONAL AIRPORT, NY US
# MAGIC - SEA - SEATTLE TACOMA INTERNATIONAL AIRPORT, WA US
# MAGIC - BOS - BOSTON, MA US
# MAGIC - ATL - ATLANTA HARTSFIELD INTERNATIONAL AIRPORT, GA US
# MAGIC - LAX - LOS ANGELES INTERNATIONAL AIRPORT, CA US
# MAGIC - SFO - SAN FRANCISCO INTERNATIONAL AIRPORT, CA US
# MAGIC - DEN - DENVER INTERNATIONAL AIRPORT, CO US
# MAGIC - DFW - DALLAS FORT WORTH AIRPORT, TX US
# MAGIC - ORD - CHICAGO O’HARE INTERNATIONAL AIRPORT, IL US
# MAGIC - CVG - CINCINNATI NORTHERN KENTUCKY INTERNATIONAL AIRPORT, KY US'
# MAGIC - CLT - CHARLOTTE DOUGLAS AIRPORT, NC US
# MAGIC - DCA - WASHINGTON, DC US
# MAGIC - IAH - HOUSTON INTERCONTINENTAL AIRPORT, TX US
# MAGIC 
# MAGIC Your application should allow the airport and time mark to be chosen using user interface widgets.  The time mark for predictions should of course be restricted to a period of time after the training data that was used to build the model. The application should form an estimate for each flight that is due to arrive or depart the specified airport over the next 24 hours after the time mark.  It is important to note that some flights that are inbound will have already left their originating airport at the time mark.  In these cases, the application should form an estimate that is focused on “flight duration” which is likely different than estimating delay when the period of interest includes time before the airplane has departed the originating airport.

# COMMAND ----------

# run link to the application notebook
result_dash = dbutils.notebook.run("06 Flight Dash", 7200, {"00.Airport_Code":airport_code,"01.training_start_date":training_start_date,"02.training_end_date":training_end_date,"03.inference_date":inference_date})

# Check for success
assert json.loads(result_dash)["exit_code"] == "OK", "Flight Deltay Application Failed!" # Check to see that it worked

