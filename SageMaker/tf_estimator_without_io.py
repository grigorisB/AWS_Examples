import os
from sagemaker.tensorflow import TensorFlow

# ESTIMATOR CONFIGURATION
JOB_NAME='example-tf-estimator-without-io'
TAGS=[
    {"Key":"info:creator",
     "Value":"theodoros"},
    {"Key":"info:maintainer",
     "Value":"theodoros"},
    {"Key":"info:product",
     "Value":"datascience"},
    {"Key":"info:env",
     "Value":"dev"},
]

ROLE_SAGEMAKER='arn:aws:iam::513905722774:role/service-role/AmazonSageMaker-ExecutionRole-20200914T153095'
INSTANCE_TYPE='ml.p3.2xlarge'
INSTANCE_COUNT=1

PYTHON_VERSION='py3'
TF_VERSION='1.15'

MODEL_FILE=os.path.join("..","Scripts","audio_tf_training_without_io.py")

# MODEL CONFIGURATION
HYPERPARAMETERS = {
    'epochs': 50,
    'batch_size': 4,
    'learning_rate': 0.0001
}

# ESTIMATOR BUILD AND FIT
estimator = TensorFlow(
    hyperparameters=HYPERPARAMETERS,
    base_job_name=JOB_NAME,
    tags=TAGS,
    py_version=PYTHON_VERSION,
    framework_version=TF_VERSION,
    role=ROLE_SAGEMAKER,
    instance_type=INSTANCE_TYPE,
    instance_count=INSTANCE_COUNT,
    entry_point=MODEL_FILE,
)
estimator.fit()
