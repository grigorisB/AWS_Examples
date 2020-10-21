import boto3
import glob
import os

s3 = boto3.resource('s3')

bucket_dst = "sagemaker-us-east-1-513905722774"
prefix_dst = "sagemaker_examples/data/temp_audio_raw"

folder_src = "temp_downloads"

for i,filepath in enumerate(glob.glob(os.path.join(folder_src,"*/*"))):
    filename = filepath.split(os.sep)[-1]
    s3_filename = '/'.join([prefix_dst,filename])
    print(i,"- UPLOADING [",filepath,"] TO [",s3_filename,"]")
    s3.meta.client.upload_file(filepath, bucket_dst,s3_filename)