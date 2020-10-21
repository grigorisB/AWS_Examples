import boto3
import pprint


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
prefix_src = "public/gtzan/data/genres/metal"

s3_filenames = get_all_s3_objects(s3, bucket_src, prefix_src)
print(len(s3_filenames))
pprint.pprint(s3_filenames[:10])

