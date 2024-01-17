import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
from sklearn.metrics import confusion_matrix

import constants
import model.model
import training.loader as loader


class Trainer:
    def __init__(self, dataset: loader, model: keras.models.Model, model_filepath: str = "saved_models/checkpoint"):
        self.dataset = dataset
        self.model = model

        self.loss_fn = keras.losses.CategoricalCrossentropy()  # from_logits?

        self.accuracy = tf.keras.metrics.CategoricalAccuracy()

        self.model_filepath = model_filepath

    def model_save_best(self):
        return keras.callbacks.ModelCheckpoint(
            self.model_filepath,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )

    def model_load_best(self):
        return keras.models.load_model(self.model_filepath)

    def plot(self, trainin):
        self.plot_confusion_matrix()
        plt.plot(trainin.history['loss'], label='training loss')
        plt.plot(trainin.history['val_loss'], label='val loss')
        plt.show(title="losses")

        plt.plot(trainin.history['accuracy'], label='training accuracy')
        plt.plot(trainin.history['val_accuracy'], label='val accuracy')

        plt.show(title="accuracy")

    def plot_confusion_matrix(self):
        model = self.model_load_best()

        y_pred = model.predict(self.dataset.features_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(self.dataset.labels_test, axis=1)
        results = confusion_matrix(y_true, y_pred)

        labelled_rows = [constants.jumpType(i).name for i in range(7)]

        df_cm = pd.DataFrame(results, labelled_rows, labelled_rows)
        # plt.figure(figsize=(10,7))
        sn.set(font_scale=0.9)  # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size

        plt.show()

        TP = np.diag(results)

        FP = np.sum(results, axis=0) - TP

        FN = np.sum(results, axis=1) - TP

        # display sensitivity and specificity global and for each class

        print("sensitivity: ", TP / (TP + FN))

        print("specificity: ", TP / (TP + FP))

        for i in range(constants.NB_CLASSES_USED):
            print(
                f"{constants.jumpType(i).name}: sensitivity: {TP[i] / (TP[i] + FN[i])}, specificity: {TP[i] / (TP[i] + FP[i])}")

    def train(self, epochs: int = 100, plot: bool = True):
        """
        Do the training, and plot the confusion matrix and losses through epochs
        :param epochs:
        :param plot:
        :return: none
        """

        self.model.summary()

        # callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
        # signal.signal(signal.SIGINT, signal_handler)
        try:
            trainin = self.model.fit(
                self.dataset.features_train,
                self.dataset.labels_train,
                epochs=epochs,
                validation_data=self.dataset.val_dataset,
                callbacks=[self.model_save_best()],
            )
        except KeyboardInterrupt:
            self.plot(trainin)

        if plot:
            self.model = model.model.load_model("saved_models/checkpoint")
            self.plot(trainin)
            self.plot_confusion_matrix()
