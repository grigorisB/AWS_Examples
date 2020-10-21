import argparse
import os
import glob
import numpy as np
import tensorflow as tf


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    args, _ = parser.parse_known_args()

    print(f'Arguments {args}')
    print()

    # Load data
    X_train = []
    Y_train = []
    for i,filepath in enumerate(glob.glob(args.train+"/*")):
        print("==========", i, '-', filepath, "==========")
        feat=np.load(filepath)
        X_train.append(np.reshape(feat, (20, 1293,1)))
        if 'metal' in filepath:
            Y_train.append([1,0])
        elif 'rock' in filepath:
            Y_train.append([0,1])
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    print()

    X_test = []
    Y_test = []
    for i,filepath in enumerate(glob.glob(args.test+"/*")):
        print("==========", i, '-', filepath, "==========")
        feat=np.load(filepath)
        X_test.append(np.reshape(feat, (20, 1293,1)))
        if 'metal' in filepath:
            Y_test.append([1,0])
        elif 'rock' in filepath:
            Y_test.append([0,1])
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

    # Save model
    model.save(args.model_dir+"/example_model.h5")
