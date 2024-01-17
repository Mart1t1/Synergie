import pandas as pd


def emulate_60Hz(df: pd.DataFrame):
    """
    emulates a 60Hz frequency by interpolating a 120Hz dataframe
    :param df: the dataframe to emulate
    :return: the emulated dataframe
    """

    # df = (df.rolling(2).mean()[::2].dropna().reset_index(drop=True))
    df = df[::2].reset_index(drop=True)
    return df

