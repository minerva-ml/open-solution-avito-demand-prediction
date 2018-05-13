import numpy as np
import lightgbm as lgb

from steps.misc import LightGBM


class LightGBMLowMemory(LightGBM):
    def fit(self, X, y, X_valid, y_valid, feature_names=None, categorical_features=None, **kwargs):
        X = X[feature_names].values.astype(np.float32)
        y = y.astype(np.float32)

        X_valid = X_valid[feature_names].values.astype(np.float32)
        y_valid = y_valid.astype(np.float32)

        train = lgb.Dataset(X, label=y)
        valid = lgb.Dataset(X_valid, label=y_valid)

        self.evaluation_results = {}
        self.estimator = lgb.train(self.model_config,
                                   train, valid_sets=[valid], valid_names=['valid'],
                                   feature_name=feature_names,
                                   categorical_feature=categorical_features,
                                   evals_result=self.evaluation_results,
                                   num_boost_round=self.training_config.number_boosting_rounds,
                                   early_stopping_rounds=self.training_config.early_stopping_rounds,
                                   verbose_eval=self.model_config.verbose,
                                   feval=self.evaluation_function)
        return self


class HierarchicalInception:  # (BasicKerasClassifier):
    def _architecture(self, input_size, classes, trainable_threshold, loss_weights, **kwargs):
        base_model = self._load_pretrained_model(input_size)

        for i, layer in enumerate(base_model.layers):

            if i < trainable_threshold:
                layer.trainable = False
            else:
                layer.trainable = True

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)

        level1_classes, level2_classes = classes
        predictions_level1 = Dense(level1_classes, activation='softmax', name='output_level1')(x)
        predictions_level2 = Dense(level2_classes, activation='softmax', name='output_level2')(x)

        model = Model(inputs=base_model.input,
                      outputs=[predictions_level1, predictions_level2])
        return model

    def _optimizer(self, lr, momentum, **kwargs):
        return SGD(lr=lr, momentum=momentum, nesterov=True) if momentum else Adam(lr=lr)

    def _compile_model(self, loss_weights, **kwargs):
        if self.gpu_nr > 1:
            self.model = multi_gpu_model(self.model_template, gpus=self.gpu_nr)
        else:
            self.model = self.model_template
        optimizer = self._optimizer(**kwargs)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'],
                           loss_weights=loss_weights)

    def _load_pretrained_model(self, input_size, **kwargs):
        return InceptionV3(include_top=False, weights='imagenet', input_shape=(input_size, input_size, 3))

    def _create_callbacks(self, model_dirpath, patience, lr_reduction):
        model_filepath = self._create_model_filepath(model_dirpath)
        if self.gpu_nr > 1:
            model_checkpoint = ModelCheckpointMultiGPU(model_template=self.model_template,
                                                       filepath=model_filepath, save_best_only=True,
                                                       save_weights_only=True)
        else:
            model_checkpoint = ModelCheckpoint(filepath=model_filepath, save_best_only=True, save_weights_only=True)

        lr_schedule = ReduceLR(lr_reduction)
        # lr_schedule = ReduceLRBatch(lr_reduction, 2000)

        _, model_name = filepath_split(model_dirpath, last_is_dir=True)
        neptune = NeptuneHierarchicalMonitor(model_name)
        early_stopping = EarlyStopping(patience=patience)

        return [lr_schedule, model_checkpoint, early_stopping, neptune]

    def predict(self, X, **kwargs):
        test_flow, test_steps = X['X']

        hierarchical_predictions_sparse = [[], [], []]
        for cnt_steps, images in enumerate(test_flow):
            print_progress_bar(cnt_steps, test_steps)
            if test_steps == cnt_steps:
                break

            hierarchical_batch_predictions = self.model.predict_on_batch(images[0])

            for i, batch_predictions in enumerate(hierarchical_batch_predictions):
                batch_predictions_clipped = self._clip_predictions(batch_predictions)
                batch_predictions_sparse = sparse.csr_matrix(batch_predictions_clipped)
                hierarchical_predictions_sparse[i].append(batch_predictions_sparse)

        hierarchical_predictions_sparse = [sparse.vstack(level_predictions_sparse)
                                           for level_predictions_sparse in hierarchical_predictions_sparse]

        return hierarchical_predictions_sparse
