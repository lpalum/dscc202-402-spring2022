# Databricks notebook source
# MAGIC %md
# MAGIC ## Rubric for this module
# MAGIC - Using the silver delta table(s) that were setup by your ETL module train and validate your token recommendation engine. Split, Fit, Score, Save
# MAGIC - Log all experiments using mlflow
# MAGIC - capture model parameters, signature, training/test metrics and artifacts
# MAGIC - Tune hyperparameters using an appropriate scaling mechanism for spark.  [Hyperopt/Spark Trials ](https://docs.databricks.com/_static/notebooks/hyperopt-spark-ml.html)
# MAGIC - Register your best model from the training run at **Staging**.

# COMMAND ----------

# MAGIC %run ./includes/utilities

# COMMAND ----------

# MAGIC %run ./includes/configuration

# COMMAND ----------

# Grab the global variables
wallet_address,start_date = Utils.create_widgets()
print(wallet_address,start_date)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Your Code starts here...

# COMMAND ----------

# Import the necessary libraries
import random
import datetime
 
import delta
import mlflow
import pyspark

# COMMAND ----------

# Define a class for model training and recommendation
class EthereumTokenRecommender:
    def __init__(self, group_name, model_name, mlflow_exp_dir, wallet_dir, token_dir, test_split, metric, seed):
        # Define member variables
        self.seed = seed
        self.metric = metric
        self.group_name = group_name
        self.model_name = model_name
        self.wallet_dir = wallet_dir
 
        # Set an MLFlow experiment
        mlflow.set_experiment(mlflow_exp_dir)
 
        # Load and split the dataset (no need of a val set since we will be using cross validation)
        self.data = {'wallet': spark.read.format('delta').load(wallet_dir)}
        self.data['token'] = spark.read.format('delta').load(token_dir).cache()
        self.data['train'], self.data['test'] = self.data['wallet'].randomSplit(
            [1 - test_split, test_split], seed=self.seed)
 
        # Cache the dataset splits
        self.data['train'] = self.data['train'].cache()
        self.data['test'] = self.data['test'].cache()
 
        # Define the model signature
        self.signature = mlflow.models.signature.ModelSignature(
            inputs=Schema([ColSpec('integer', 'TokenID'), ColSpec('integer', 'WalletID')]),
            outputs=Schema([ColSpec('double')])
        )
 
        # Define the model
        self.model = pyspark.ml.recommendation.ALS(
            userCol='WalletID',
            itemCol='TokenID',
            ratingCol='Balance',
            coldStartStrategy='drop',
            nonnegative=True,
            seed=self.seed
        )
 
        # Define the evaluator
        self.evaluator = pyspark.ml.evaluation.RegressionEvaluator(
            predictionCol='prediction',
            labelCol='Balance',
            metricName=self.metric
        )
 
    def _stage_model(self):
        # Create an MLFlow Client
        versions = []
        mlflow_client = mlflow.tracking.MlflowClient()
 
        # Archive the current staged model
        for version in mlflow_client.search_model_versions(f'name="{self.model_name}"'):
            version = dict(version)
            versions.append(version['version'])
            if version['current_stage'] == 'Staging':
                mlflow_client.transition_model_version_stage(
                    name=self.model_name,
                    version=version['version'],
                    stage='Archived'
                )
 
        # Stage the latest model
        mlflow_client.transition_model_version_stage(
            name=self.model_name,
            version=versions[0],
            stage='Staging'
        )
 
    def train(self, ranks, max_iterations, regularization_params, num_user_blocks, num_item_blocks):
        # Log the time of the experiment
        time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
 
        # Setup grid for hyperparameter tuning
        self.param_grid = pyspark.ml.tuning.ParamGridBuilder(
        ).addGrid(self.model.rank, ranks
        ).addGrid(self.model.maxIter, max_iterations
        ).addGrid(self.model.regParam, regularization_params
        ).addGrid(self.model.numUserBlocks, num_user_blocks
        ).addGrid(self.model.numItemBlocks, num_item_blocks
        ).build()
 
        # Setup Cross Validator
        self.model = pyspark.ml.tuning.CrossValidator(
            estimator=self.model, 
            evaluator=self.evaluator,
            estimatorParamMaps=grid,
            numFolds=10,
            seed=self.seed
        )
 
        # Start the MLFlow run
        with mlflow.start_run(run_name=f'{self.group_name}_{self.model_name}_{time}') as run:
            # Set tags
            mlflow.set_tags({
                'group': self.group_name,
                'seed': self.seed
            })
 
            # Log params
            mlflow.log_params({
                'ranks': ranks,
                'max_iterations': max_iterations,
                'regularization_param': regularization_param,
                'num_user_blocks': num_user_blocks,
                'num_item_blocks': num_item_blocks,
                'data_version': DeltaTable.forPath(spark, self.wallet_dir).history().head(1)[0]['version']
            })
 
            # Train the model
            trained_model = self.model.fit(self.data['train'])
 
            # Evaluate the model
            train_metric = self.evaluator.evaluate(trained_model.transform(self.data['train']))
            test_metric = self.evaluator.evaluate(trained_model.transform(self.data['test']))
 
            # Log the evaluation metrics
            mlflow.log_metric(f'train_{self.metric}', train_metric)
            mlflow.log_metric(f'test_{self.metric}', test_metric)
 
            # Log the model version with the best metrics
            mlflow.spark.log_model(
                spark_model=trained_model.bestModel,
                signature=self.signature,
                artifact_path='recommendation_model',
                registered_model_name=self.model_name
            )
 
        # Stage the recently trained model
        self._stage_model()
 
    def test(self, subset='test'):
        # Load the model 
        model = mlflow.spark.load_model(f'models:/{self.model_name}/Staging')
 
        # Compute the metrics on the test dataset
        metric = self.evaluator.evaluate(model.transform(self.data[subset]))
 
        # Log the metric
        print(f'The staged model\'s {self.metric} value on the {subset} dataset: {metric}')
 
    def infer(self, wallet_id):
        # Extract the tokens purchased by a user in his/her wallet ID
        purchased_tokens = self.data['total'].filter(
            self.data['wallet']('WalletID') == userId
        ).join(self.data['token'], 'TokenID'
        ).select('TokenID', 'name', 'image', 'links')
 
        # Extract the tokens that haven't been purchased by a user in his/her wallet ID
        unpurchased_tokens = self.data['wallet'].filter(
            ~ self.data['wallet']['TokenID'].isin([token['TokenID'] for token in purchased_tokens.collect()])
        ).select('tokens_holding').withColumn('WalletID', pyspark.sql.functions.lit(WalletID)).distinct()
 
        # Generate recommendations for the unpurchased tokens
        model = mlflow.spark.load_model(f'models:/{self.model_name}/Staging')
        return model.transform(unpurchased_tokens)

# COMMAND ----------

# Create an instance of the EthereumTokenRecommender class
ethereum_token_recommender = EthereumTokenRecommender(
    group_name='G07',
    model_name='model_1',
    mlflow_exp_dir,
    wallet_dir,
    token_dir,
    test_split=0.2,
    metric='rmse',
    seed=42
)

# COMMAND ----------

# Train the EthereumTokenRecommender on the train set
ethereum_token_recommender.train(
    ranks=,
    max_iterations=,
    regularization_params=,
    num_user_blocks=,
    num_item_blocks=
)

# COMMAND ----------

# Test the EthereumTokenRecommender on the test set
ethereum_token_recommender.test()

# COMMAND ----------

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
