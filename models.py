import numpy as np
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.optimizers import Adam
import lightgbm as lgb

from steps.misc import LightGBM
from steps.keras.models import ClassifierGenerator
from steps.keras.callbacks import NeptuneMonitor, ReduceLR
from steps.utils import create_filepath
from callbacks import NeptuneMonitorMultiOutput


class LightGBMLowMemory(LightGBM):
    def fit(self, X, y, X_valid, y_valid, feature_names=None, categorical_features=None, **kwargs):
        X = X[feature_names].values.astype(np.float32)
        y = y.astype(np.float32)

        X_valid = X_valid[feature_names].values.astype(np.float32)
        y_valid = y_valid.astype(np.float32)

        train = lgb.Dataset(X, label=y,
                            feature_name=feature_names, categorical_feature=categorical_features)
        valid = lgb.Dataset(X_valid, label=y_valid,
                            feature_name=feature_names, categorical_feature=categorical_features)

        self.evaluation_results = {}
        self.estimator = lgb.train(self.model_config,
                                   train, valid_sets=[valid], valid_names=['valid'],
                                   evals_result=self.evaluation_results,
                                   feature_name=feature_names,
                                   categorical_feature=categorical_features,
                                   num_boost_round=self.training_config.number_boosting_rounds,
                                   early_stopping_rounds=self.training_config.early_stopping_rounds,
                                   verbose_eval=self.model_config.verbose,
                                   feval=self.evaluation_function)
        return self


class HierarchicalInceptionResnet(ClassifierGenerator):
    def _build_loss(self, **kwargs):
        return 'categorical_crossentropy'

    def _build_optimizer(self, lr, **kwargs):
        return Adam(lr=lr)

    def _build_model(self, target_size, num_classes, trainable_threshold, **kwargs):
        base_model = self._load_pretrained_model(target_size)

        for i, layer in enumerate(base_model.layers):

            if i < trainable_threshold:
                layer.trainable = False
            else:
                layer.trainable = True

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)

        level1_classes, level2_classes = num_classes
        predictions_level1 = Dense(level1_classes, activation='softmax', name='output_level1')(x)
        predictions_level2 = Dense(level2_classes, activation='softmax', name='output_level2')(x)

        model = Model(inputs=base_model.input, outputs=[predictions_level1, predictions_level2])
        return model

    def _compile_model(self, **architecture_config):
        model = self._build_model(**architecture_config)
        optimizer = self._build_optimizer(**architecture_config)
        loss = self._build_loss()
        model.compile(optimizer=optimizer, loss=loss,
                      loss_weights=architecture_config['loss_weights'],
                      metrics=['categorical_accuracy'])
        return model

    def _load_pretrained_model(self, target_size, **kwargs):
        return InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(target_size + (3,)))

    def _create_callbacks(self, **kwargs):
        lr_scheduler = ReduceLR(**kwargs['lr_scheduler'])
        early_stopping = EarlyStopping(**kwargs['early_stopping'])
        checkpoint_filepath = kwargs['model_checkpoint']['filepath']
        create_filepath(checkpoint_filepath)
        model_checkpoint = ModelCheckpoint(**kwargs['model_checkpoint'])
        neptune = NeptuneMonitorMultiOutput(**kwargs['neptune_monitor'])
        return [neptune, lr_scheduler, early_stopping, model_checkpoint]
