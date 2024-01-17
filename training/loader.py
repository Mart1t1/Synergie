import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import constants


# TODO: integrate data augmentation in the pipeline


class loader:
    """
    This class is meant to load the data from the csv files,
    and make it ready to be used by the model for training
    """

    def print_stats(self):
        print("Dataset stats:")
        print("Train dataset size: {}".format(len(self.features_train)))
        print("Val dataset size: {}".format(len(self.features_test)))
        print("Train dataset shape: {}".format(self.features_train.shape))
        print("Val dataset shape: {}".format(self.features_test.shape))
        print("Train labels shape: {}".format(self.labels_train.shape))
        print("Val labels shape: {}".format(self.labels_test.shape))
        print("Train dataset type: {}".format(self.features_train.dtype))
        print("Val dataset type: {}".format(self.features_test.dtype))
        print("Train labels type: {}".format(self.labels_train.dtype))
        print("Val labels type: {}".format(self.labels_test.dtype))

        # print the distribution of the labels
        print("Labels distribution: {}".format(np.sum(self.labels_train, axis=0) / len(self.labels_train)))

    def _normalize(self, jumps):
        # jumps being a temporal series, I want to normalize every jump based on mean and std of the whole dataset
        dataset_concat = jumps.reshape(-1, jumps.shape[-1])


        means = np.mean(dataset_concat, axis=0)

        stds = np.std(dataset_concat, axis=0)

        jumps = (jumps - means) / stds

        return jumps



    def __init__(self, folder_path: str, train_ratio: float = 0.8):
        assert 0 <= train_ratio <= 1

        self.folder_path = folder_path

        main_csv = os.path.join(folder_path, "jumplist.csv")

        mainFrame = pd.read_csv(main_csv)

        jumps = []
        labels = []

        fields_to_keep = constants.fields_to_keep

        for index, row in mainFrame.iterrows():
            if row["type"] != 8:
                # print(row)
                jumpFrame = pd.read_csv(os.path.join(folder_path, row['path']))
                jumpFrame = jumpFrame[fields_to_keep]

                # make (2 * n) + 1 rows equal 2 * n

                for i in jumpFrame.index:
                    if i % 2 == 1: # if i is odd, it should be equal to previous row
                        jumpFrame.loc[i] = jumpFrame.loc[i - 1]

                jumps.append(jumpFrame)
                labels.append(row['type'])

        jumps = np.array(jumps)

        # label one hot encoding
        labelEncoder = LabelEncoder()
        labels = np.eye(len(set(labels)))[labelEncoder.fit_transform(labels)]

        # remove lines with nan field

        jumps = np.nan_to_num(jumps, nan=0.0, posinf=0.0, neginf=0.0)

        # TODO: normalize data

        jumps = self._normalize(jumps)

        dataset = tf.data.Dataset.from_tensor_slices((jumps, labels))

        # make a training and validation dataset

        self.dataset = dataset.shuffle(len(jumps))

        features_train, features_val, labels_train, labels_val = train_test_split(jumps, labels, train_size=train_ratio, shuffle=True)


        self.features_test, self.labels_test = features_val, labels_val
        self.features_train, self.labels_train = features_train, labels_train

        self.train_dataset = tf.data.Dataset.from_tensor_slices((features_train, labels_train)).batch(16)
        self.val_dataset = tf.data.Dataset.from_tensor_slices((features_val, labels_val)).batch(16)



        # TODO: make data augmentation. Mostly by adding noise and some rotations
