from pathlib import Path
import matplotlib.pyplot as plt
import keras
import pandas as pd
from datetime import datetime

# The names of the columns
K_VAL_LOSS = 'val_loss'
K_VAL_ACC = 'val_acc'
K_ACC = 'acc'
K_LOSS = 'loss'
K_TDIFF = 'tdiff'


class TrainingLogger(keras.callbacks.Callback):
    """
    This class logs all information while training a keras NN to a pandas dataframe.
    Can write the information to a csv file.
    Allows saving/ showing plot of data.
    """

    # The name of the saved history file
    CSV_FILENAME = "history.csv"
    PLOT_FILENAME = "stats.png"

    def __init__(self, basepath=None, frequent_write=False):
        """
        Constructor
        :param basepath: The path to the directory where to save the logs
        :param frequent_write: If True -> writes csv file of stats after each epoch to disk
        """
        super().__init__()
        self.write_frequent = frequent_write
        if basepath:
            self.set_basepath(basepath)

    def set_basepath(self, basepath: str):
        self.basepath: Path = Path(basepath)
        self.basepath.mkdir(parents=True, exist_ok=True)
        self.history_filepath = self.basepath.joinpath(self.CSV_FILENAME)
        self.plot_filepath = self.basepath.joinpath(self.PLOT_FILENAME)

    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        super().on_train_begin(logs)
        self.history = pd.DataFrame(columns=(K_VAL_LOSS, K_VAL_ACC, K_LOSS, K_ACC, K_TDIFF))
        self.history.index.name = 'epoch'

    def on_epoch_begin(self, epoch, logs=None):
        if logs is None:
            logs = {}
        super().on_epoch_begin(epoch, logs)
        self.t_start = datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        super().on_epoch_end(epoch, logs)
        t_diff = datetime.now() - self.t_start
        self.history.loc[epoch, K_VAL_LOSS] = logs.get(K_VAL_LOSS)
        self.history.loc[epoch, K_VAL_ACC] = logs.get(K_VAL_ACC)
        self.history.loc[epoch, K_ACC] = logs.get(K_ACC)
        self.history.loc[epoch, K_LOSS] = logs.get(K_LOSS)
        self.history.loc[epoch, K_TDIFF] = t_diff.total_seconds()

        # If frequent_write is activate write csv after each epoch to disk
        if(self.write_frequent):
            self.write_csv()

    def write_csv(self):
        """
        Writes the training history to a csv file
        :return: None
        """
        self.history.to_csv(str(self.history_filepath))

    def make_plots(self, save_to_disk=True, show=False):
        """
        Creates a plot of the training history
        :param save_to_disk: Saves the plots to disk
        :param show: Shows the plots to the user and blocks execution meanwhile
        :return: None
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=False)
        ax2.set_xlabel('epoch')
        fig.suptitle('Accuracy & Computation time')

        # Plot accuracy on epoch
        ax1.plot(self.history.index, self.history[K_ACC])
        ax1.set_ylabel('accuracy')
        #ax1.set_ylim(bottom=0.0, auto=True)

        # Plot calculation time on epoch
        ax2.plot(self.history.index, self.history[K_TDIFF])
        ax2.set_ylabel('time (s)')
        #ax2.set_ylim(bottom=0.0, auto=True)

        # Save to disk or show to user
        if save_to_disk:
            fig.savefig(str(self.plot_filepath), dpi=100)
        if show:
            plt.show()

