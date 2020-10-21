import os
from sagemaker.processing import ScriptProcessor

# PROCESSOR CONFIGURATION
JOB_NAME='example-script-processor-without-io'
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
INSTANCE_TYPE='ml.m5.xlarge'
INSTANCE_COUNT=1

IMAGE_URI='513905722774.dkr.ecr.us-east-1.amazonaws.com/sagemaker-processing-ffmpeg:latest'

# SCRIPT CONFIGURATION
SCRIPT=os.path.join("..","Scripts","audio_preprocessing_without_io.py")
ARGUMENTS=["--dataset-len","10","--train-test-split-ratio", "0.8","--genre","rock"]


# PROCESSOR BUILD AND RUN
processor = ScriptProcessor(
    base_job_name=JOB_NAME,
    tags=TAGS,
    role=ROLE_SAGEMAKER,
    instance_type=INSTANCE_TYPE,
    instance_count=INSTANCE_COUNT,
    image_uri=IMAGE_URI,
    command=['python3']
)

processor.run(
    code=SCRIPT,
    arguments=ARGUMENTS
)