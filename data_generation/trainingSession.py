import numpy as np
import pandas as pd
import scipy as sp

import constants
from utils import plot
from utils.jump import Jump


def gather_jumps(df: pd.DataFrame) -> list:
    """
    detects and gathers all the jumps in a dataframe
    :param df: the dataframe containing the session data
    :return: list of jumps done
    """
    jumps = []

    # Find indices where 'X_gyr_second_derivative_crossing' transitions from False to True
    begin = np.where(np.diff(df['X_gyr_second_derivative_crossing'].astype(int)) == 1)[0]

    # Find indices where 'X_gyr_second_derivative_crossing' transitions from True to False
    end = np.where(np.diff(df['X_gyr_second_derivative_crossing'].astype(int)) == -1)[0]

    for i in range(len(end)):
        # remove the first end marks that happens before the first begin mark
        if end[i] < begin[0]:
            end = np.delete(end, i)
            break

    for i in range(len(begin)):
        jumps.append(Jump(begin[i], end[i], df))

    return jumps

class trainingSession:
    """
    This class is meant to describe a training session in a sport context. Not to be confused with a training session in a machine learning context (class training)
    contains the preprocessed dataframe and the jumps
    """

    def __load_and_preprocess_data(self, path: str, sampleTimefineSynchro: int = 0) -> pd.DataFrame:
        """
        meant to be static and private, if python was a decent programming language I wouldn't need to type this

        loads a dataframe from a csv, and preprocess data
        :param self: path to the csv file
        :return: the dataframe with preprocessed fields
        """

        df = pd.read_csv(path, sep=',')

        df = df.astype({'PacketCounter': 'int64', 'SampleTimeFine': 'ulonglong', 'Euler_X': 'float64', 'Euler_Y': 'float64','Euler_Z': 'float64', 'Acc_X': 'float64', 'Acc_Y': 'float64', 'Acc_Z': 'float64', 'Gyr_X': 'float64', 'Gyr_Y': 'float64', 'Gyr_Z': 'float64'})


        if sampleTimefineSynchro != 0:
            # slice the list from sampleTimefineSynchro

            synchroIndex = df[df['SampleTimeFine'] >= sampleTimefineSynchro].index[0]

            df = df[synchroIndex:].reset_index(drop=True)
        df = df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        # adding the ms field, indicating how much ms has past since the beginning of the recording
        # we are using the SampleTimeFine field, which is a timestamp in microsecond

        # add 2^32 to the SampleTimeFine field when it is smaller than the previous one, because it means that the counter has overflowed
        initial_timeStamp = df['SampleTimeFine'][0]

        df.loc[df["SampleTimeFine"] < initial_timeStamp, 'SampleTimeFine'] += 4294967296


        initialSampleTimeFine = df['SampleTimeFine'][0]
        df['ms'] = (df['SampleTimeFine'] - initialSampleTimeFine) / 1000

        # df.set_index(df['ms'], inplace=True)



        df['X_acc_derivative'] = df['Acc_X'].diff()
        df['Y_acc_derivative'] = df['Acc_Y'].diff()
        df['Z_acc_derivative'] = df['Acc_Z'].diff()

        df["Gyr_X_unfiltered"] = df["Gyr_X"].copy(deep=True)

        df["Gyr_X_smoothed"] = sp.ndimage.gaussian_filter1d(df["Gyr_X"], sigma=30)
        df['X_gyr_derivative'] = df['Gyr_X_smoothed'].diff()
        df['Y_gyr_derivative'] = df['Gyr_Y'].diff()
        df['Z_gyr_derivative'] = df['Gyr_Z'].diff()

        df["X_gyr_second_derivative"] = df['X_gyr_derivative'].diff()

        # add markers when the value is crossing -0.2

        df['X_gyr_second_derivative_crossing'] = [False if x > constants.treshold else True for x in
                                                  df['X_gyr_second_derivative']]

        return df
    def initFromDataFrame(self, df: pd.DataFrame):
        """
        can be called as a constructor, provided that the dataframe correctly been preprocessed
        this function was meant to be a constructor overload. Things would be simpler if python was a decent programming language
        :param df: the dataframe containing the whole session
        """
        self.df = df

        # self.rotation_matrix = sp.spatial.transform.Rotation.from_euler('xyz', df[['Euler_X', 'Euler_Y', 'Euler_Z']][0].to_numpy(), degrees=True).as_matrix()

        self.jumps = gather_jumps(df)




    def __init__(self, path: str, sampleTimefineSynchro: int = 0):
        """
        :param path: path of the CSV
        :param synchroFrame: the frame where the synchro tap is
        """

        df = self.__load_and_preprocess_data(path, sampleTimefineSynchro)



        self.initFromDataFrame(df)
        self.filename = path.split('/')[-1]

    def plot(self):
        timestamps = [i.startTimestamp for i in self.jumps] + [i.endTimestamp for i in self.jumps]
        plot.plot_data(self.df, timestamps, str(self))
