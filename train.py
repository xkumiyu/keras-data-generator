import argparse
import os
import pathlib

import keras
import numpy as np
import pandas as pd

from data_generator import ImageDataGenerator
from net import build_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    parser = argparse.ArgumentParser(description='Example of Keras Data Generator')
    parser.add_argument('--train_dir', default='data/cifar/train/')
    parser.add_argument('--test_dir', default='data/cifar/train/')
    parser.add_argument('--label_file', default='data/cifar/labels.txt')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    args = parser.parse_args()

    print('batchsize: {}'.format(args.batchsize))
    print('epoch: {}'.format(args.epoch))

    # Datasets
    train_dir = pathlib.Path(args.train_dir)
    train_datagen = ImageDataGenerator()
    test_dir = pathlib.Path(args.test_dir)
    test_datagen = ImageDataGenerator()
    classes = list(pd.read_csv(args.label_file, header=-1)[0])
    input_shape = (32, 32, 3)

    # Model
    model = build_model(input_shape, len(classes))
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9),
        metrics=['accuracy'])
    model.summary()

    # Train
    model.fit_generator(
        generator=train_datagen.flow_from_directory(train_dir, classes),
        steps_per_epoch=int(np.ceil(len(list(train_dir.iterdir())) / args.batchsize)),
        epochs=args.epoch,
        verbose=1,
        validation_data=test_datagen.flow_from_directory(test_dir, classes),
        validation_steps=int(np.ceil(len(list(test_dir.iterdir())) / args.batchsize)))


if __name__ == '__main__':
    main()
