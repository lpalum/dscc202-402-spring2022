# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC <img src="https://files.training.databricks.com/images/Apache-Spark-Logo_TM_200px.png" style="float: left: margin: 20px"/>
# MAGIC # Overview and Setup
# MAGIC ## MLflow
# MAGIC ### Managing the Machine Learning Lifecycle
# MAGIC 
# MAGIC In this course data scientists and data engineers learn the best practices for managing experiments, projects, and models using MLflow.
# MAGIC 
# MAGIC By the end of this course, you will have built a pipeline to log and deploy machine learning models using the environment they were trained with.
# MAGIC 
# MAGIC ## Lessons
# MAGIC 
# MAGIC 0. Course Overview and Setup
# MAGIC 0. Experiment Tracking
# MAGIC 0. Packaging ML Projects
# MAGIC 0. Multistep Workflows
# MAGIC 0. Model Management 
# MAGIC 
# MAGIC ## Audience
# MAGIC * Primary Audience: Data Scientists and Data Analysts
# MAGIC * Additional Audiences: Data Engineers
# MAGIC 
# MAGIC ## Prerequisites
# MAGIC - Web browser: Chrome
# MAGIC - A cluster configured with **8 cores** and **DBR 7.0 ML**
# MAGIC - Python (`pandas`, `sklearn`, `numpy`)
# MAGIC - Background in machine learning and data science

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC <h2><img src="https://files.training.databricks.com/images/105/logo_spark_tiny.png"> Before You Start</h2>
# MAGIC 
# MAGIC Before starting this course, you will need to create a cluster and attach it to this notebook.
# MAGIC 
# MAGIC Please configure your cluster to use the Databricks **ML Runtime** version **7.0 ML** which includes:
# MAGIC - Python Version 3.x
# MAGIC - Scala Version 2.11
# MAGIC - Apache Spark 2.4.4
# MAGIC 
# MAGIC <img alt="Caution" title="Caution" style="vertical-align: text-bottom; position: relative; height:1.3em; top:0.0em" src="https://files.training.databricks.com/static/images/icon-warning.svg"/> Do not use the stock or GPU accelerated runtimes
# MAGIC 
# MAGIC Step-by-step instructions for creating a cluster are included here:
# MAGIC - <a href="https://www.databricks.training/step-by-step/creating-clusters-on-azure" target="_blank">Azure Databricks</a>
# MAGIC - <a href="https://www.databricks.training/step-by-step/creating-clusters-on-aws" target="_blank">Databricks on AWS</a>
# MAGIC - <a href="https://www.databricks.training/step-by-step/creating-clusters-on-ce" target="_blank">Databricks Community Edition (CE)</a>
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> This courseware has been tested against the specific DBR listed above. Using an untested DBR may yield unexpected results and/or various errors. If the required DBR has been deprecated, please <a href="https://academy.databricks.com/" target="_blank">download an updated version of this course</a>.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Classroom-Setup
# MAGIC In general, all courses are designed to run on one of the following Databricks platforms:
# MAGIC * Databricks Community Edition (CE)
# MAGIC * Databricks (an AWS hosted service)
# MAGIC * Azure-Databricks (an Azure-hosted service)
# MAGIC 
# MAGIC <img alt="Caution" title="Caution" style="vertical-align: text-bottom; position: relative; height:1.3em; top:0.0em" src="https://files.training.databricks.com/static/images/icon-warning.svg"/> Some features are not available on the Community Edition, which limits the ability of some courses to be executed in that environment. Please see the course's prerequisites for specific information on this topic.
# MAGIC 
# MAGIC <img alt="Caution" title="Caution" style="vertical-align: text-bottom; position: relative; height:1.3em; top:0.0em" src="https://files.training.databricks.com/static/images/icon-warning.svg"/> Additionally, private installations of Databricks (e.g., accounts provided by your employer) may have other limitations imposed, such as aggressive permissions and or language restrictions such as prohibiting the use of Scala which will further inhibit some courses from being executed in those environments.
# MAGIC 
# MAGIC For each lesson to execute correctly, please make sure to run the **`Classroom-Setup`** cell at the start of each lesson (see the next cell) and the **`Classroom-Cleanup`** cell at the end of each lesson.

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Classroom-Cleanup<br>
# MAGIC 
# MAGIC During the course of this lesson, files, tables, and other artifacts may have been created.
# MAGIC 
# MAGIC These resources create clutter, consume resources (generally in the form of storage), and may potentially incur some [minor] long-term expense.
# MAGIC 
# MAGIC You can remove these artifacts by running the **`Classroom-Cleanup`** cell below.

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Cleanup"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC Start the lesson [Experiment Tracking]($./02-Experiment-Tracking).

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
