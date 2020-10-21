import boto3
import os


def get_all_s3_objects(s3,bucket,prefix):
    continuation_token = None
    filenames = []
    while True:
        if continuation_token:
            response = s3.list_objects_v2(Bucket=bucket,ContinuationToken=continuation_token,Prefix=prefix)
        else:
            response = s3.list_objects_v2(Bucket=bucket,Prefix=prefix)

        if 'Contents' in response:
            for obj in response['Contents']:
                if obj['Key'][-1]!='/': # Filter out folder objects
                    filenames.append(obj['Key'])
        if not response.get('IsTruncated'):
            return filenames
        continuation_token = response['NextContinuationToken']


s3 = boto3.client('s3')
bucket_src = "audio-video-datasets"
prefix_src = "public/gtzan/data/genres/rock"

s3_filenames = get_all_s3_objects(s3, bucket_src, prefix_src)


s3 = boto3.resource('s3')

bucket_src = "audio-video-datasets"

folder_dst = "temp_downloads"
if not os.path.isdir(folder_dst):
    os.mkdir(folder_dst)

for i, s3_filename in enumerate(s3_filenames[:20]):
    filename = s3_filename.split("/")[-1]
    subfolder = s3_filename.split("/")[-2]

    sub_path = os.path.join(folder_dst, subfolder)
    if not os.path.isdir(sub_path):
        os.mkdir(sub_path)

    down_path = os.path.join(folder_dst, subfolder, filename)
    print(i, "- DOWNLOADING [", s3_filename, "] TO [", down_path, "]")
    s3.meta.client.download_file(bucket_src, s3_filename, down_path)