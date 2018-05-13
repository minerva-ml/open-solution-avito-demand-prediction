from functools import partial
from math import ceil
import os

import numpy as np
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator, Iterator, img_to_array, load_img
from keras.utils.np_utils import to_categorical
from sklearn.externals import joblib

from steps.base import BaseTransformer
from augmentation import fast_seq


class MultiOutputImageLoader(BaseTransformer):
    def __init__(self, loader_params):
        super().__init__()
        self.loader_params = loader_params

    def transform(self, X, y, X_valid=None, y_valid=None, train_mode=True):
        if train_mode:
            flow, steps = build_pandas_datagen(X, y, train_mode, self.loader_params['training'])
        else:
            flow, steps = build_pandas_datagen(X, y, train_mode, self.loader_params['inference'])

        if X_valid is not None and y_valid is not None:
            valid_flow, valid_steps = build_pandas_datagen(X_valid, y_valid, False,
                                                           self.loader_params['inference'])
        else:
            valid_flow = None
            valid_steps = None

        return {'datagen': (flow, steps),
                'validation_datagen': (valid_flow, valid_steps)}

    def load(self, filepath):
        params = joblib.load(filepath)
        self.loader_params = params['loader_params']
        return self

    def save(self, filepath):
        params = {'loader_params': self.loader_params}
        joblib.dump(params, filepath)


def build_pandas_datagen(X, y, train_mode, loader_params):
    datagen = PandasImageDataGenerator(preprocessing_function=partial(fast_seq, train_mode=train_mode))

    flow = datagen.flow_from_pandas(X, y, **loader_params)
    steps = ceil(X.shape[0] / loader_params['batch_size'])

    return flow, steps


class PandasImageDataGenerator(ImageDataGenerator):
    def flow_from_pandas(self, X, y, image_dir, num_classes,
                         target_size=(64, 64), color_mode='rgb', channel_order='tf',
                         batch_size=32, shuffle=True, seed=None):
        return PandasIterator(X, y, image_dir, num_classes, self,
                              target_size, color_mode, channel_order,
                              batch_size, shuffle, seed)


class PandasIterator(Iterator):
    def __init__(self, X, y, image_dir, num_classes,
                 image_data_generator,
                 target_size, color_mode, channel_order,
                 batch_size, shuffle, seed):
        self.X = X
        self.y = y
        self.image_dir = image_dir
        self.num_classes = num_classes
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.color_mode = color_mode
        self.channel_order = channel_order
        self.image_shape = self.target_size + (3,)
        self.data_format = K.image_data_format()

        self.samples = X.shape[0]

        super().__init__(self.samples, batch_size, shuffle, seed)
        """Note:
        Tensorflow channels order only rgb only
        """

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        batch_y_id = self.y[index_array]
        print('Unique classes in batch', [np.unique(batch_y_id[:, i]).shape for i in range(3)])
        batch_y = []
        for i, num_classes in enumerate(self.num_classes):
            batch_y_level = to_categorical(batch_y_id[:, i], num_classes=num_classes)
            batch_y.append(batch_y_level)

        for i, j in enumerate(index_array):
            img_id = self.X.iloc[j]
            img = self._load_img(img_id, target_size=self.target_size)
            x = img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x

        return batch_x, batch_y

    def _load_img(self, img_id, **kwargs):
        filepath = os.path.join(self.image_dir, '{}.jpg'.format(img_id))
        return load_img(filepath, **kwargs)

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)
