import argparse
import os
import numpy as np
import tensorflow as tf
import boto3

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
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    args, _ = parser.parse_known_args()

    print(f'Arguments {args}')
    print()

    # Create local directories
    feat_folder = "local_audio_features"
    if not os.path.isdir(feat_folder):
        os.mkdir(feat_folder)

    # Load train data
    s3 = boto3.client('s3')
    bucket_src = "sagemaker-us-east-1-513905722774"
    prefix_src = "sagemaker_examples/data/temp_audio_features/train"

    s3_filenames = get_all_s3_objects(s3, bucket_src, prefix_src)

    s3 = boto3.resource('s3')
    bucket_src = "sagemaker-us-east-1-513905722774"

    X_train = []
    Y_train = []

    for i, s3_filename in enumerate(s3_filenames):
        filename = s3_filename.split("/")[-1]

        print("==========", i, '-', s3_filename, "==========")

        down_path = os.path.join(feat_folder, filename)
        print("DOWNLOADING TO [", down_path, "]")
        s3.meta.client.download_file(bucket_src, s3_filename, down_path)

        feat = np.load(down_path)
        X_train.append(np.reshape(feat, (20, 1293, 1)))
        if 'metal' in s3_filename:
            Y_train.append([1, 0])
        elif 'rock' in s3_filename:
            Y_train.append([0, 1])

        print("REMOVING LOCAL FILES")
        os.remove(down_path)

        print()

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)


    print()


    # Load test data
    s3 = boto3.client('s3')
    bucket_src = "sagemaker-us-east-1-513905722774"
    prefix_src = "sagemaker_examples/data/temp_audio_features/test"

    s3_filenames = get_all_s3_objects(s3, bucket_src, prefix_src)

    s3 = boto3.resource('s3')
    bucket_src = "sagemaker-us-east-1-513905722774"

    X_test = []
    Y_test = []

    for i, s3_filename in enumerate(s3_filenames):
        filename = s3_filename.split("/")[-1]

        print("==========", i, '-', s3_filename, "==========")

        down_path = os.path.join(feat_folder, filename)
        print("DOWNLOADING TO [", down_path, "]")
        s3.meta.client.download_file(bucket_src, s3_filename, down_path)

        feat = np.load(down_path)
        X_test.append(np.reshape(feat, (20, 1293, 1)))
        if 'metal' in s3_filename:
            Y_test.append([1, 0])
        elif 'rock' in s3_filename:
            Y_test.append([0, 1])

        print("REMOVING LOCAL FILES")
        os.remove(down_path)

        print()

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    # Build and fit model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same',input_shape=(20, 1293,1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax'),
    ])
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X_train,
        Y_train,
        validation_split=0.25,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2
    )

    # Evaluate model
    scores = model.evaluate(X_test, Y_test, verbose=2)
    print("Test evaluation (Loss,Accuracy):", scores)
    print()

    # Save and upload model
    model_filename="example_model.h5"

    model_folder = "local_model"
    if not os.path.isdir(model_folder):
        os.mkdir(model_folder)

    model_path=os.path.join(model_folder,model_filename)

    model.save(model_path)

    s3 = boto3.resource('s3')
    bucket_dst = "sagemaker-us-east-1-513905722774"
    prefix_dst = "sagemaker_examples/trained_model"
    s3_filename_up = '/'.join([prefix_dst, model_filename])
    print("UPLOADING MODEL TO [", s3_filename_up, "]")
    s3.meta.client.upload_file(model_path, bucket_dst, s3_filename_up)

    print("REMOVING LOCAL FILES")
    os.remove(model_path)



