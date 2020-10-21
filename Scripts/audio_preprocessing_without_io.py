import argparse
import librosa
import boto3
import os
import numpy as np


def get_all_s3_objects(s3, bucket, prefix):
    continuation_token = None
    filenames = []
    while True:
        if continuation_token:
            response = s3.list_objects_v2(Bucket=bucket, ContinuationToken=continuation_token, Prefix=prefix)
        else:
            response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

        if 'Contents' in response:
            for obj in response['Contents']:
                if obj['Key'][-1] != '/': # Filter out folder objects
                    filenames.append(obj['Key'])
        if not response.get('IsTruncated'):
            return filenames
        continuation_token = response['NextContinuationToken']


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-len', type=int, default=20)
    parser.add_argument('--genre', type=str, default='')
    parser.add_argument('--train-test-split-ratio', type=float, default=0.7)
    args, _ = parser.parse_known_args()

    print(f'Arguments {args}')
    print()


    # Get raw audio s3 filenames
    s3 = boto3.client('s3')
    bucket_src = "sagemaker-us-east-1-513905722774"
    prefix_src = "sagemaker_examples/data/temp_audio_raw"

    s3_filenames = get_all_s3_objects(s3, bucket_src, prefix_src)

    s3_filenames = [s3_fn for s3_fn in s3_filenames if args.genre in s3_fn]
    s3_filenames = s3_filenames[:args.dataset_len]

    # Create local directories
    down_folder = "local_audio_raw"
    if not os.path.isdir(down_folder):
        os.mkdir(down_folder)

    feat_folder = "local_audio_features"
    if not os.path.isdir(feat_folder):
        os.mkdir(feat_folder)

    # Setup S3
    s3 = boto3.resource('s3')
    bucket_src = "sagemaker-us-east-1-513905722774"
    bucket_dst = "sagemaker-us-east-1-513905722774"

    # Process data
    for i, s3_filename in enumerate(s3_filenames):
        filename = s3_filename.split("/")[-1]

        print("==========",i,'-',s3_filename,"==========")

        down_path = os.path.join(down_folder, filename)
        print("DOWNLOADING TO [", down_path, "]")
        s3.meta.client.download_file(bucket_src, s3_filename, down_path)

        print("LOADING AUDIO")
        y,sr=librosa.load(down_path)

        print("EXTRACTING MFCC")
        mfcc=librosa.feature.mfcc(y)
        feat_path=os.path.join(feat_folder, filename).replace('.wav','.npy')
        print("SAVING FEATURES TO [", feat_path, "]")
        np.save(feat_path,mfcc)

        feat_filename = feat_path.split(os.sep)[-1]
        if i < args.train_test_split_ratio * len(s3_filenames):
            prefix_dst = "sagemaker_examples/data/temp_audio_features/train"
        else:
            prefix_dst = "sagemaker_examples/data/temp_audio_features/test"
        s3_filename_up = '/'.join([prefix_dst,feat_filename])
        print("UPLOADING TO [", s3_filename_up, "]")
        s3.meta.client.upload_file(feat_path, bucket_dst, s3_filename_up)

        print("REMOVING LOCAL FILES")
        os.remove(down_path)
        os.remove(feat_path)

        print()

