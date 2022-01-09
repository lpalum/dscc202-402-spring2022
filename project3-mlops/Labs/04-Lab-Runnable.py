# Databricks notebook source
# TODO
Create 3 widgets for parameter passing into the notebook:
  - n_estimators with a default of 100
  - learning_rate with a default of .1
  - max_depth with a default of 1 
Note that only strings can be used for widgets

# COMMAND ----------

# TODO
Read from the widgets to create 3 variables.  Be sure to cast the values to numeric types
n_estimators = FILL_IN
learning_rate = FILL_IN
max_depth = FILL_IN

# COMMAND ----------

# TODO
Train and log the results from a model.  Try using Gradient Boosted Trees
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor

# COMMAND ----------

# TODO
Report the model output path to the parent notebook


# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
