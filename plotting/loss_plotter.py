
import os
import csv

import matplotlib
matplotlib.use('Agg')   # Use non-interactive backend
import matplotlib.pyplot as plt


class LossPlotter:
    def __init__(self, loss_maintainer, config):
        self.config          = config
        self.loss_maintainer = loss_maintainer

        self.plot_filename      = os.path.join(self.config.model_folder, 'loss-plot.pdf')
        self.acc_plot_filename  = os.path.join(self.config.model_folder, 'accuracy-plot.pdf')
        self.mix_plot_filename  = os.path.join(self.config.model_folder, 'mixed-plot.pdf')

    def create_plot(self, include_loss=True, include_accuracy=False):
        assert include_loss or include_accuracy

        epochs, train_losses, val_losses, train_acc, val_acc = [], [], [], [], []
        for epoch, values in self.loss_maintainer.epochs.items():
            epochs.append(int(epoch))
            train_losses.append(values['train_loss'])
            val_losses.append(values['val_loss'])
            if include_accuracy:
                train_acc.append(values['train_accuracy'])
                val_acc.append(values['val_accuracy'])

        f = plt.figure()
        ax = f.add_subplot(1, 1, 1)
        if include_loss:
            ax.plot(epochs, train_losses, label='Training loss',   color='IndianRed')
            ax.plot(epochs, val_losses,   label='Validation loss', color='SteelBlue')
        if include_accuracy:
            ax.plot(epochs, train_acc, label='Training accuracy',   color='IndianRed', linestyle='--')
            ax.plot(epochs, val_acc,   label='Validation accuracy', color='SteelBlue', linestyle='--')

        ax.legend()
        ax.set_xlabel('Epoch')
        if include_loss and not include_accuracy:
            y_label = 'Loss'
        elif not include_loss and include_accuracy:
            y_label = 'Accuracy'
        else:
            y_label = 'Accuracy/Loss'
        ax.set_ylabel(y_label)
        ax.set_title(self.config.name)

        return f, ax


    def show(self):
        f, ax = self.create_plot()
        plt.show()

    def save(self):
        f, ax = self.create_plot(include_loss=True, include_accuracy=False)
        f.savefig(self.plot_filename, bbox_inches='tight')

        if self.loss_maintainer.include_accuracy:
            f, ax = self.create_plot(include_loss=False, include_accuracy=True)
            f.savefig(self.acc_plot_filename, bbox_inches='tight')

            f, ax = self.create_plot(include_loss=True, include_accuracy=True)
            f.savefig(self.mix_plot_filename, bbox_inches='tight')


class MixedLossPlotter:
    def __init__(self, mixed_loss_maintainer, config):
        self.config = config
        self.mixed_loss_maintainer = mixed_loss_maintainer

        self.plot_filename = os.path.join(self.config.model_folder, 'mixed-loss-plot.pdf')

    def create_plot(self):
        epochs, train_losses_1, val_losses_1, train_losses_2, val_losses_2 = [], [], [], [], []
        for epoch, values in self.mixed_loss_maintainer.epochs.items():
            epochs.append(int(epoch))
            train_losses_1.append(values['train_loss_1'])
            train_losses_2.append(values['train_loss_2'])
            val_losses_1.append(values['val_loss_1'])
            val_losses_2.append(values['val_loss_2'])

        f = plt.figure()
        ax = f.add_subplot(1, 1, 1)
        ax.plot(epochs, train_losses_1, label='Train seq loss',   color='IndianRed')
        ax.plot(epochs, train_losses_2, label='Train class loss', color='SteelBlue')
        ax.plot(epochs, val_losses_1,   label='val seq loss',   color='IndianRed', linestyle='--')
        ax.plot(epochs, val_losses_2,   label='val class loss', color='SteelBlue', linestyle='--')

        ax.legend()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')

        ax.set_yscale('log')

        ax.set_title(self.config.name)

        return f, ax


    def show(self):
        f, ax = self.create_plot()
        plt.show()

    def save(self):
        f, ax = self.create_plot()
        f.savefig(self.plot_filename, bbox_inches='tight')
