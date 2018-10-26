# Fraud Detection using Machine Learning on Google Cloud ML Engine

## Wide & Deep Modeling Architecture
## <img src="image04.png" />
The model adopts the [Deep & Wide modelling architecture](https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html): most categorical features with low cardinality of the underlying category sets are not passed through hidden activations. These constitute the wide aspect and the model only learns the weights to assign to these features. On the other hand, the model learns dense embeddings for features with high cardinality of their underlying category sets. These dense embeddings are then passed through a series of hidden activations. This enables the model to map these highly sparse features into a lower dimensional space, each dimension of which encodes different aspects of the propensity for fraud. Thus, the architecture combines the best of a simple linear classifier and a deep learning classifier to train high performance models.

## Best Practice Template	
We describe how to train and deploy a wide & deep model for fraud detection using Google Cloud MLE. For this purpose, we repurpose the Google Cloud Platform [template](https://github.com/GoogleCloudPlatform/cloudml-samples/tree/master/cloudml-template), which uses high level Estimator APIs and follows a recommended folder structure that permits easy training and serving on Google Cloud MLE. This obviates the need for writing boilerplate code to support generic aspects of scalable production grade model training and deployment such as model checkpointing, data ingestion, data parsing, logging, monitoring of training, distributed training, hyperparameter turning and model serving.

This allows the Data Scientist to focus on core Machine Learning tasks: model design and feature engineering.

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

|File Name| Purpose|Changes  Made
|:---|:---|:---
|[metadata.py](template/trainer/metadata.py)|Defines: 1) task type, 2) input data header, 3) numeric and categorical feature names, 4) target feature name (and labels, for a classification task), and 5) unused feature names. | Metadata for the features registered here to enable parsing
|[input.py](template/trainer/input.py)| Includes: 1) data input functions to read data from csv and tfrecords files, 2) parsing functions to convert csv and tf.example to tensors, 3) function to implement your features custom  processing and creation functionality, and 4) prediction functions (for serving the model) that accepts CSV, JSON, and tf.example instances. | Feature scaling for numerical features
|[featurizer.py](template/trainer/featurizer.py)| Creates: 1) tensorflow feature_column(s) based on the dataset metadata (and other extended feature columns, e.g. bucketisation, crossing, embedding, etc.), and 2) deep and wide feature column lists. | Feature embeddings for features with high cardinality 
|[model.py](template/trainer/model.py)|Includes: 1) function to create DNNLinearCombinedRegressor, 2) DNNLinearCombinedClassifier, and 2) function to implement for a custom estimator model_fn.|No changes made here
|[task.py](template/trainer/task.py) |Includes: 1 experiment function that executes the model training and evaluation, 2) initialise and parse task arguments (hyper parameters), and 3) Entry point to the trainer. | Reset hyperparameter values: batch-size, num-epochs, dropout, learning rate

## Data Preprocessing & Feature Engineering
The training and evaluation datasets are prepared in BQ. Features are created based on business rules currently in use for detecting fraud, and judgement. We provide an illustrative query for preparing the training data set, which could be further split into training and eval datasets:

```
%%time
%%bigquery
create TABLE ml.deep_features
as
SELECT
actual_fraud AS label,
CONCAT(case when send_cntry_code is not null then send_cntry_code else "" end ,
     "-",
     case when recv_cntry_code is not null then recv_cntry_code else "" end ) as corridor,
tran_amt_bin,
corr_cross_tran_bin,
sender_age_group,
receiver_age_group,
time_span_secs,
complained_fraud_last_5_for_sender,
complained_fraud_last_5_for_receiver,
prevented_fraud_last_5_for_sender,
prevented_fraud_last_5_for_receiver,        
txn_count_last_5_for_sender,
txn_count_last_5_for_receiver,  
send_recv_have_same_last_name,
unq_recs_last_5_for_sender,
unq_senders_last_5_for_receiver,
unq_agents_last_5_for_sender,
unq_agents_last_5_for_receiver,
unq_recv_cntry_last_5_for_sender,
unq_send_cntry_last_5_for_receiver
FROM
  ml.features
```
The table can then be exported to Google Cloud Storage where the files are sharded or downloaded to local storage

```
TRAIN_FILE=$BUCKET/deep_features_noheader/deep_features_train.csv
bq extract ml.deep_features_train $TRAIN_FILE
```

### Metadata Collection
We also need to collect the following metadata to enable parsing of data during and feature scaling for numerical features during training:
 * Type of feature – numerical, categorical 
 * Category set for categorical features low high cardinality
 * Number of features for high cardinality categorical features
 * Mean and Standard Deviation for numerical features

We generate these statistics using BQ. An illustrative query to compute the cardinality of a feature is shown below:

```
%%time
%%bigquery statDF
select 
count(distinct corridor) as num_unq_corridors
from ml.deep_features
```

### Weights for Class Imbalance
To deal with severe class imbalance, we compute the ratio of ‘fraud’ to ‘non-fraud’ instances. We use this ratio to up weight fraud instances in the dataset and down weight ‘non-fraud’ instances. This weight column must be appended to the dataset so that the training template can implement weighting while computing the loss function.

### Data Parsing
The Cloud MLE template provides hooks in training/metadata.py to register the meta-data in the training template to enable data-parsing, and batch creation. For high cardinality features such as corridor which has 25000 codes, it is impractical to explicitly register the category set in metadata.pyy. A workaround is to register such features using hash buckets, like so:

```
INPUT_CATEGORICAL_FEATURE_NAMES_WITH_HASH_BUCKET = {
    'corridor': 100000,
    'corr_cross_tran_bin': 17000000    
}
```
Here, we are telling the model to create a hash bucket of size 100000 for the corridor feature. The recommended practice is for the hash bucket size to be much bigger than the feature’s cardinality. The template also provides hooks to create additional features based on feature crosses, bin numerical features if their values don’t have meaningful magnitude, and finally, implement additional feature transformations such as feature scaling. 

## Data Ingestion
The template is configured such it can deal with datasets stored locally or on Google Cloud Storage. To register the data source, simply specify the local path in local_run.sh 

```
TRAIN_FILES=/home/jupyter/data/deep_features/noHeaders/deep_features_train-*.csv
VALID_FILES=/home/jupyter/data/deep_features/noHeaders/deep_features_eval-*.csv
```
or the path in GCS:

```
TRAIN_FILES=gs://${BUCKET}/deep_features_noheader/deep_features_train-*.csv
EVAL_FILES=gs://${BUCKET}/deep_features_noheader/deep_features_eval-*.csv
```

The template then uses tf.data to progressively load data into memory, parse it, apply additional transformation steps, and create batches of the appropriate shape to be passed on to GPUs for implementing the forwardprop/brackprop operations during training. 
The boilerplate associated with efficient data loading including buffering in memory, shuffling of data, and pre-fetching to avoid GPU starvation has been abstracted away.

## Local Training
Run the loca-run.sh script to launch from the bash to launch a local training job. 

The necessary parameters are path to the training and datasets, and specification of a directory for the model. After the training job is launched, model checkpoints and event files (required to monitor training and evaluation using Tensorboard) are persisted in this directory.


```
CURRENT_DATE=`date +%Y%m%d_%H%M%S`
MODEL_NAME="linear_fraud_model"_${CURRENT_DATE}

PACKAGE_PATH=trainer
TRAIN_FILES=/home/jupyter/data/deep_features/noHeadersSmall/deep_features_train-*.csv
VALID_FILES=/home/jupyter/data/deep_features/noHeadersSmall/deep_features_eval-*.csv
MODEL_DIR=latest

gcloud ml-engine local train \
        --module-name=trainer.task \
        --package-path=${PACKAGE_PATH} \
        -- \
        --train-files=${TRAIN_FILES} \
        --num-epochs=1 \
        --train-batch-size=512 \
        --eval-files=${VALID_FILES} \
        --eval-batch-size=512 \
        --learning-rate=0.001 \
        --hidden-units="62,20" \
        --layer-sizes-scale-factor=0.5 \
        --num-layers=3 \
        --job-dir=${MODEL_DIR} \
```
The user can also specify key parameters that impact model architecture (number of layers, number of hidden units per layer), hyperparameters (learning rate, dropout, batch size), and training specs (size of training data, number of epochs to run, batch-size) here. All these parameters also come with robust defaults.

## Distributed Training
To launch distributed training on Cloud MLE, run distributed_run.sh.
The cluster configuration needs to be specified in config.yml which is picked up by distributed_run.sh.

```
trainingInput:
  #scaleTier: BASIC # BASIC | BASIC_GPU | STANDARD_1 | PREMIUM_1
  scaleTier: CUSTOM
  masterType: complex_model_m_gpu
  workerType: complex_model_m_gpu
  parameterServerType: complex_model_m_gpu
  workerCount: 4
  parameterServerCount: 1
```
Cloud MLE launches the cluster as per the specified configuration and executes the distributed training regime. Under the hood, Cloud MLE implements state of the art practices – the ring reduce algorithm for efficiently disseminating gradient information across worker nodes, efficient communication between disk and compute during data ingestion, gracefuly restarting in case of node failure etc. 

Thus, the complexity of managing distributed training including setting up and launching a cluster is completely abstracted away and the user can focus on reasoning about the optimal cluster configuration including the underlying hardware by trading off time against computing cost.

The speedup of launching a distributed training job vs a local training job can be substantial as the training data size increases. The speedup for training a dataset of size 5.64 GB (43 million records) is shown below:

| Training Mode       | Hard Ware Specs           |Training Time for 2 epochs (minutes)  |
| ------------- |:-------------:| -----:|
| Local Training with Cloud MLE     | Machine type n1-highmem-16 (16 vCPUs, 104 GB memory) CPU platform Intel Haswell GPUs 2 x NVIDIA Tesla K80 | ~198 minutes |
| Distributed Training with Cloud MLE    | 1 Parameter Server, 4 Workers All instances of type: complex_model_m_gpu|   63 minutes including ~20 minutes for cluster set up |

## Hyperparameter tuning
Finally, Cloud MLE provides very good support for launching many training jobs in sequence or in parallel depending on the metalearning algorithm to tune the specified set of hyperparameters. To do so, we need to specify the set of hyperparameters to tune, the ranges to explore for them, the exploration scale (log or linear).
This is specified in the config_hyperparam.yaml file.
```
trainingInput:
  #scaleTier: BASIC # BASIC | BASIC_GPU | STANDARD_1 | PREMIUM_1
  scaleTier: CUSTOM
  masterType: complex_model_m_gpu
  workerType: complex_model_m_gpu
  parameterServerType: complex_model_m_gpu
  workerCount: 4
  parameterServerCount: 1
  hyperparameters:
    goal: MAXIMIZE
    hyperparameterMetricTag: auc_precision_recall
    maxTrials: 6
    maxParallelTrials: 1
    enableTrialEarlyStopping: True
    params:
    - parameterName: learning-rate
      type: DOUBLE
      minValue: 0.0001
      maxValue: 0.1
      scaleType: UNIT_LOG_SCALE
  ```
The shell script references this file and can run from bash to launch model training cum hyperaparameter tuning.

As long as the hyperparameters have been defined in trainer/task.py under def initialise_hyper_params(), Cloud MLE will be able to pick them up and tune them.
    
## Training Monitoring and Results:
Training and evaluation data is logged in events files. It suffices to launch Tensorboard and point it to the directory containing the events files, (in our case, the model directory), like so:
```
tensorboard --logdir=$MODEL_DIR
```
## Results and Curves
It  is also possible to log data to generate important graphs such as the PR-curve and the AUC curve. This data can easily be exported to a plotting tool to generate the requisite curves.
## <img src="image04.png" />




