import argparse
import librosa
import os
import glob
import numpy as np


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-len', type=int, default=20)
    parser.add_argument('--genre', type=str, default='')
    parser.add_argument('--train-test-split-ratio', type=float, default=0.7)
    args, _ = parser.parse_known_args()

    print(f'Arguments {args}')
    print()

    # Process data
    filepaths = [fn for fn in sorted(glob.glob("/opt/ml/processing/input/data/*")) if args.genre in fn]
    filepaths = filepaths[:args.dataset_len]
    for i, filepath in enumerate(filepaths):
        filename = filepath.split("/")[-1]

        print("==========", i, '-', filepath, "==========")

        print("LOADING AUDIO")
        y, sr = librosa.load(filepath)

        print("EXTRACTING MFCC")
        mfcc = librosa.feature.mfcc(y)

        if i < args.train_test_split_ratio * len(filepaths):
            feat_folder = "/opt/ml/processing/output/train"
        else:
            feat_folder = "/opt/ml/processing/output/test"
        feat_path = os.path.join(feat_folder, filename).replace('.wav', '.npy')
        print("SAVING FEATURES TO [", feat_path, "]")
        np.save(feat_path, mfcc)

        print()