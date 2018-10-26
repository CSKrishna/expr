# Fraud Detection using Machine Learning on Google Cloud ML Engine

## Wide & Deep Modeling Architecture
## <img src="image04.png" />
The model adopts the [Deep & Wide modelling architecture](https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html): most categorical features with low cardinality of the underlying category sets are not passed through hidden activations. These constitute the wide aspect and the model only learns the weights to assign to these features. On the other hand, the model learns dense embeddings for features with high cardinality of their underlying category sets. These dense embeddings are then passed through a series of hidden activations. This enables the model to map these highly sparse features into a lower dimensional space, each dimension of which encodes different aspects of the propensity for fraud. Thus, the architecture combines the best of a simple linear classifier and a deep learning classifier to train high performance models.

## Best Practice Template	
We describe how to train and deploy a wide & deep model for fraud detection using Google Cloud MLE. For this purpose, we repurpose the Google Cloud Platform [template](https://github.com/GoogleCloudPlatform/cloudml-samples/tree/master/cloudml-template/examples/census-classification), which uses high level Estimator APIs and follows a recommended folder structure that permits easy training and serving on Google Cloud MLE. This obviates the need for writing boilerplate code to support generic aspects of scalable, production grade model training and deployment such as model checkpointing, data ingestion, data parsing, logging, monitoring of training, distributed training, hyperparameter turning, model serving.

This allows the Data Scientist to focus on core Machine Learning tasks: model design and feature engineering.

### Rough Notes to Clean up Tomorrow
1. **Wide & Deep**
    * **Best Practice Template**: Followed recommended Google/TF patterns with Estimators
    * **Feature Engineering**:
        * _BQ Engineering_ : Build the features, build vocabulary set, max & min
        * _TF Engineering_ : Embedding, Treated aggegrations as categories as compared to continuous like in BQ
    * **HyperParameter Tuning** : Learning Rate
    * **Curves**
    * **Weights**
    * **Ingestion Pipeline**: `tf.Data` API used to efficiently train at scale using batched data loads **Lookup or Steal Blog Post**
    * **Local vs Cloud**: Template allowed for local vs cloud PLUS easy deployment 
    * **Distributed Training**: Config.yaml - parameter how easy this was compared to on-prem - **Krishna blown away** Paramet
    * _Deployed as REST on Cloud MLE_
        * Samples JSON request executed
2. **BoostedTreeClassifer** : Easy to swap due to power template Estimastor API - PreMade Estimar
3. **RoadMap**: Custom DL 


## Repository Structure
The code-base is organized as follows:
1. trainer directory contains the following python scripts - metadata.py, input.py, model.py, task.py -  to adapt the data, specify the training configuration and hyperparameters values, and the script to train and validate the model    
2. config.yaml contains the specs for running a distributed training job including:
   1) number of parameter servers,
   2) number of workers
   3) concomitant hardware specs    
3. config_hyperparam.yaml file, in addition to the specs for the cluster, also specifies the training regime for hyperparameters, including:
   1) hyperparameters to tune
   2) range of values to explore for each hyperparameter
   3) number of trials to run
   4) level of parallelism (applicable only for grid/random search)
   5) tuning algorithm - Bayesian Optimization, Grid Search, Random Search       
4. local_run.sh is the shell script to lauch a training job locally    
5. distributed_run.sh is the shell script to launch a model training job over Google Cloud MLE. It references config.yaml to specify the cluster configuration    
6. distributed_run_hyperparm_tuning.sh is the shell script to launch a set of training jobs either in parallel or in sequence so as to produce a model that is optimized over both trainable and hyperparameters. The script references config_hyperparams.yaml to specify the hyperparameter tuning regime. 

### Trainer Template Modules

|File Name| Purpose| Do You Need to Change?
|:---|:---|:---
|[metadata.py](template/trainer/metadata.py)|Defines: 1) task type, 2) input data header, 3) numeric and categorical feature names, 4) target feature name (and labels, for a classification task), and 5) unused feature names. | **Yes**, as you will need to specify the metadata of your dataset. **This might be the only module to change!**
|[input.py](template/trainer/input.py)| Includes: 1) data input functions to read data from csv and tfrecords files, 2) parsing functions to convert csv and tf.example to tensors, 3) function to implement your features custom  processing and creation functionality, and 4) prediction functions (for serving the model) that accepts CSV, JSON, and tf.example instances. | **Maybe**, if you want to implement any custom pre-processing and feature creation during reading data.
|[featurizer.py](template/trainer/featurizer.py)| Creates: 1) tensorflow feature_column(s) based on the dataset metadata (and other extended feature columns, e.g. bucketisation, crossing, embedding, etc.), and 2) deep and wide feature column lists. | **Maybe**, if you want to change your feature_column(s) and/or change how deep and wide columns are defined (see next section). 
|[model.py](template/trainer/model.py)|Includes: 1) function to create DNNLinearCombinedRegressor, 2) DNNLinearCombinedClassifier, and 2) function to implement for a custom estimator model_fn.|**No, unless** you want to change something in the estimator, e.g., activation functions, optimizers, etc., or to implement a custom estimator. 
|[task.py](template/trainer/task.py) |Includes: 1 experiment function that executes the model training and evaluation, 2) initialise and parse task arguments (hyper parameters), and 3) Entry point to the trainer. | **No, unless** you want to add/remove parameters, or change parameter default values.

## Data Preprocessing & Feature Engineering
The training and evaluation datasets are prepared in BQ. Features are created based on business rules currently in use for detecting fraud, and judgement. We provide an illustrative query for preparing the training data set, which could be further split into training and eval datasets:

```
BUCKET=gs://recommender_$GOOGLE_CLOUD_PROJECT
gsutil mb ${BUCKET}
bq mk GA360_MerchStore
cd gcp-retail-workshop-2018/recommender
```
