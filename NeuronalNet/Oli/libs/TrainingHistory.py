from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class TrainingHistory:
    """
    Manages the history for training a NN
    Includes functions for saving and loading a history from a csv file
    Also allows ploting stats
    """

    K_EPOCHS = 'epochs'
    K_ACC = 'accuracy'
    K_TDIFF = 'tdiff'

    FILENAME = "history.csv"

    def __init__(self, basepath):
        self.basepath: Path = Path(basepath)
        self.basepath.mkdir(parents=True, exist_ok=True)
        self.filepath = self.basepath.joinpath(self.FILENAME)
        self.df: pd.DataFrame = None

    def set_data(self, epo: np.ndarray, accu: np.ndarray, tdiff: np.ndarray):
        data = {
            plotter.K_ACC: pd.Series(accu, index=epo),
            plotter.K_TDIFF: pd.Series(tdiff, index=epo)
        }
        self.df = pd.DataFrame(data)
        self.df.index.name = plotter.K_EPOCHS

    def read_csv(self):
        self.df = pd.read_csv(str(self.filepath))

    def write_csv(self):
        self.df.to_csv(str(self.filepath))

    def plot_accuracy(self):
        plt.plot(self.df.index, self.df[self.K_ACC])
        plt.show()

    def plot_timediff(self):
        self.df.plot(y=self.K_TDIFF)
        plt.show()


# Small demo
if __name__ == '__main__':
    SIZE = 10
    epo = np.arange(0, SIZE)
    accu = np.linspace(0, 0.9, SIZE) + np.random.normal(0.1, 0.05, SIZE)
    t_dif = np.random.normal(40, 10, SIZE)

    plotter = TrainingHistory("HistoryDemo")
    plotter.set_data(epo, accu, t_dif)

    plotter.write_csv()
    plotter.plot_accuracy()