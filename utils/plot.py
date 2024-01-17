import matplotlib.pyplot as plt
import pandas as pd


def plot_data(df: pd.DataFrame, timestamps=[], title: str = "") -> None:
    """
    plots the dataframe using matplotlib.
    Plots three charts, each corresponding to the 3 vectors recorded by the sensor

    :param df: dataframe to plot
    :param timestamps: timestamps to plot as vertical lines
    :param title: title of the plot

    Notes:
    -----
    Reference: Xsens DOT Movella White paper (https://www.movella.com/hubfs/Downloads/Whitepapers/Xsens%20DOT%20WhitePaper.pdf)
    """

    def _sub_plot(ax, args, legend):
        subdf = df[args]
        subdf.set_index(df['ms'], inplace=True)

        for i in timestamps:
            ax.axvline(x=i, color='r', linestyle='-')

        subdf.plot(ax=ax, legend=legend)

    # Create a figure and multiple subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    # Define the arguments and legends for each subplot
    plot_args = [
        (axes[0, 0], ["Euler_X", "Euler_Y", "Euler_Z"], "Euler"),
        (axes[0, 1], ["Acc_X", "Acc_Y", "Acc_Z"], "Acceleration"),
        (axes[1, 0], ["Gyr_X", "Gyr_Y", "Gyr_Z"], "Gyroscope"),
        (axes[1, 1], ["X_gyr_second_derivative"], "Gyroscope second derivative")
    ]

    # Iterate over the subplots and plot the data
    for ax, args, legend in plot_args:
        _sub_plot(ax, args, legend)

    plt.suptitle(title)

    plt.tight_layout()
    plt.show()
