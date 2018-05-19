from itertools import product

from deepsense import neptune
from keras.callbacks import Callback

from steps.keras.callbacks import get_correct_channel_name


class NeptuneMonitorMultiOutput(Callback):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.ctx = neptune.Context()

        self.output_names = ['output_level1', 'output_level2', '']
        self.metrics = ['loss', 'categorical_accuracy']
        self.channels = self._channels()

        self.epoch_id = 0
        self.batch_id = 0

    @property
    def batch_channel_names(self):
        batch_channel_names = []
        for name, metric in product(self.output_names, self.metrics):
            batch_channel_name = 'batch_{}_{}_{}'.format(name, metric, self.model_name)
            if name == '':
                if metric == 'loss':
                    log_name = metric
                    batch_channel_names.append((batch_channel_name, log_name))
            else:
                log_name = '{}_{}'.format(name, metric)
                batch_channel_names.append((batch_channel_name, log_name))
        return batch_channel_names

    @property
    def epoch_channel_names(self):
        epoch_channel_names = []
        for name, metric in product(self.output_names, self.metrics):
            epoch_channel_name = '{}_{}_{}'.format(name, metric, self.model_name)
            if name == '':
                if metric == 'loss':
                    log_name = metric
                    epoch_channel_names.append((epoch_channel_name, log_name))
            else:
                log_name = '{}_{}'.format(name, metric)
                epoch_channel_names.append((epoch_channel_name, log_name))
        return epoch_channel_names

    @property
    def epoch_val_channel_names(self):
        epoch_val_channel_names = []
        for name, metric in product(self.output_names, self.metrics):
            epoch_val_channel_name = '{}_{}_{}_val'.format(name, metric, self.model_name)
            if name == '':
                if metric == 'loss':
                    log_name = 'val_{}'.format(metric)
                    epoch_val_channel_names.append((epoch_val_channel_name, log_name))
            else:
                log_name = 'val_{}_{}'.format(name, metric)
                epoch_val_channel_names.append((epoch_val_channel_name, log_name))
        return epoch_val_channel_names

    def _channels(self):
        channels = {}
        for channel_modality in [self.batch_channel_names,
                                 self.epoch_channel_names,
                                 self.epoch_val_channel_names]:
            for channel_name, _ in channel_modality:
                channels[channel_name] = get_correct_channel_name(self.ctx, channel_name)
        return channels

    def on_batch_end(self, batch, logs={}):
        self.batch_id += 1
        for channel_name, log_name in self.batch_channel_names:
            self.ctx.channel_send(self.channels[channel_name], self.batch_id, logs[log_name])

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_id += 1
        for channel_modality in [self.epoch_channel_names, self.epoch_val_channel_names]:
            for channel_name, log_name in channel_modality:
                self.ctx.channel_send(self.channels[channel_name], self.batch_id, logs[log_name])
