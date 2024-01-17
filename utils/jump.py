import numpy as np
import pandas as pd
import scipy as sp

import constants
from constants import jumpType
from utils import plot


class Jump:
    def __init__(self, start: int, end: int, df: pd.DataFrame, jump_type: jumpType = jumpType.NONE):
        """
        :param start: the frame index where the jump starts
        :param end: the frame index where the jump ends
        :param df: the dataframe containing the session where the jump is
        :param jump_type: the type of the jump (has to be set to NONE before annotation)
        """
        self.start = start
        self.end = end
        self.type = jump_type

        self.startTimestamp = (df['SampleTimeFine'][start] - df['SampleTimeFine'][0]) / 1000
        self.endTimestamp = (df['SampleTimeFine'][end] - df['SampleTimeFine'][0]) / 1000

        # timestamps are in microseconds, I want to have the lenghs in seconds
        self.length = np.longlong(df['ms'][end] - df['ms'][start]) / 1000

        self.rotation = self.calculate_rotation(df[self.start:self.end].copy().reset_index(),
                                                df.iloc[constants.SYNCHROFRAME])  # TODO: replace with constant

        self.df = self.dynamic_resize(df)  # The dataframe containing the jump

        self.max_rotation_speed = df['Gyr_X_unfiltered'][start:end].mean()

    def plot(self):
        plot.plot_data(self.df, [self.startTimestamp, self.endTimestamp], str(self))

    def calculate_rotation(self, df, initial_frame):
        """
        calculates the rotation in degrees around the vertical axis, the initial frame is a frame where the skater is
        standing still

        TODO: test this function, and make it work (it doesnt even properly recognize the beginning and the end of the jump)
        :param df: the dataframe containing the jump
        :param initial_frame: the frame where the skater is standing still
        :return: the absolute value of the rotation in degrees
        """

        # initial frame is the reference frame, I want to compute rotations around the "Euler_X" axis

        df_rots = df[["Euler_X", "Euler_Y", "Euler_Z"]]
        initial_frame_euler = initial_frame[["Euler_X", "Euler_Y", "Euler_Z"]]

        r = sp.spatial.transform.Rotation.from_euler('xyz', initial_frame_euler.to_numpy(), degrees=True).as_matrix()

        # Apply the inverse rotation to align with the initial frame
        df_rots_aligned = np.dot(df_rots.to_numpy(), np.linalg.inv(r).T)

        # Initialize variables
        total_rotation_x = 0
        prev_orientation = df_rots_aligned[0, 0]

        # Custom unwrapping and rotation counting loop
        for orientation in df_rots_aligned[:, 0]:
            orientation_diff = orientation - prev_orientation

            # Adjust for wrapping, a treshold of 180 handles every corner cases to my knowledge

            threshold = 180


            if orientation_diff > threshold:
                total_rotation_x -= 360
            elif orientation_diff < -threshold:
                total_rotation_x += 360

            # Update total rotation
            total_rotation_x += orientation_diff

            # Update previous orientation for the next iteration
            prev_orientation = orientation

        # Return the absolute value of the total rotation
        return np.abs(total_rotation_x)/360

    def dynamic_resize(self, df: pd.DataFrame = None):
        """
        normalize the jump to a given time fram. I will need to resample the data so the take-off is at frame 200,
        and landing at frame 300
        :param df: the dataframe containing the session where the jump is
        :param length_between_takeoff_and_reception: the number of frames between the takeoff  and the reception
        The middle of the jump is the beginning (takeoff), middle + length_between_takeoff_and_reception is the reception
        The timeframe should be a resampled version of the original timeframe
        :return: the new dataframe
        """

        begin_df = self.start - 200

        resampled_df = df[begin_df:begin_df + 400].copy(deep=True)

        return resampled_df

    def generate_csv(self, path: str):
        """
        exports the jump to a csv file
        :param path:
        :return:
        """
        self.df.to_csv(path, index=False)

    def __str__(self):
        return f"Jump: {self.type}, {self.length:.2f}s, {self.max_rotation_speed:.0f} deg/s"
