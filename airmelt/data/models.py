from os import path
import joblib
import logging

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from keras import models
from keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras.regularizers import l2, l1
from sklearn.linear_model import SGDClassifier, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.cluster import KMeans
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter, GammaGammaFitter
import numpy as np
import pandas as pd
import xgboost

from airmelt_data.dataprep import TrainTestSplit
from airmelt_system import toolbox
from airmelt_data import am_sys_definitions


def model_decorator(supervised=True):
    def wrapper(obj):
        obj.__supervised__ = supervised
        return obj

    return wrapper


def result_to_proba(values):
    # value: list: list of values to be converted to probabilities
    value_sum = sum(values)
    return [x / value_sum for x in values]


@model_decorator(supervised=True)
class RandomForestReg(object):
    def __init__(self, is_new=True, **kwargs):
        """
        Initializes a new RandomForest classifier object.
        is_new: bool: whether to create a new object or not (default: True)
        kwargs: keyword arguments for the object's properties
        """
        self.is_new = is_new
        if self.is_new:
            self.model = RandomForestRegressor(
                n_estimators=int(kwargs.get("trees", 100)),
                criterion=kwargs.get("loss_function", "mse"),
                min_samples_split=int(kwargs.get("min_samples_split", 30)),
                min_samples_leaf=int(kwargs.get("min_samples_leaf", 30)),
                max_features=kwargs.get("max_features", "auto"),
                bootstrap=kwargs.get("bootstrap", False),
                oob_score=kwargs.get("oob_score", False),
                n_jobs=int(kwargs.get("n_jobs", 1)),
                random_state=int(kwargs.get("random_state", 0)),
                ccp_alpha=float(kwargs.get("ccp_alpha", 0.0)),
            )

    def fit_model(self, X_train, y_train):
        return self.model.fit(X_train, y_train)

    def predict_model(self, model_obj, X):
        """
        Predicts the outcome of the data based on the model and the predefined behavior.
        model_obj: model object to be used for predictions
        X: input data
        """
        if self.is_new:
            return self.model.predict(X)
        else:
            return model_obj.predict(X)

    def save_model(self, file_path, model_obj):
        """
        Saves a model object to a specified file path.
        file_path: str: the file path where the model should be saved
        model_obj: model object to be saved
        """
        return joblib.dump(
            model_obj, path.join(file_path, am_sys_definitions.MODEL_PICKLE_FILE)
        )

    def load_model(self, model_path):
        """
        Loads a model object from a specified file path.
        model_path: str: the file path where the model is stored
        """
        return joblib.load(path.join(model_path, am_sys_definitions.MODEL_PICKLE_FILE))


@model_decorator(supervised=True)
class RandomForestClass(object):
    def __init__(self, is_new=True, **kwargs):
        if is_new:
            self.model = RandomForestClassifier(
                n_estimators=int(kwargs.get("trees", 100)),
                criterion=kwargs.get("loss_function", "mse"),
                min_samples_split=int(kwargs.get("min_samples_split", 30)),
                min_samples_leaf=int(kwargs.get("min_samples_leaf", 30)),
                max_features=kwargs.get("max_features", "auto"),
                bootstrap=kwargs.get("bootstrap", False),
                oob_score=kwargs.get("oob_score", False),
                n_jobs=int(kwargs.get("n_jobs", 1)),
                random_state=int(kwargs.get("random_state", 0)),
                ccp_alpha=float(kwargs.get("ccp_alpha", 0.0)),
            )

        self.prediction_behavior = kwargs.get("prediction_behavior", "predict_proba")

    def fit_model(self, X_train, y_train):
        return self.model.fit(X_train, y_train)

    def predict_model(self, model_obj, X):
        """
        Predicts the outcome of the data based on the model and the predefined behavior.
        model_obj: model object to be used for predictions
        X: input data
        """
        if self.prediction_behavior == "predict_proba":
            return model_obj.predict_proba(X)
        elif self.prediction_behavior == "predict_log_proba":
            return model_obj.predict_log_proba(X)
        else:
            return model_obj.predict(X)

    def save_model(self, file_path, model_obj):
        """
        Saves a model object to a specified file path.
        file_path: str: the file path where the model should be saved
        model_obj: model object to be saved
        """
        return joblib.dump(
            model_obj, path.join(file_path, am_sys_definitions.MODEL_PICKLE_FILE)
        )

    def load(self, model_path):
        """
        Loads a model object from a specified file path.
        model_path: str: the file path where the model is stored
        """
        return joblib.load(path.join(model_path, am_sys_definitions.MODEL_PICKLE_FILE))


@model_decorator(supervised=True)
class DotEmbedding(object):
    """
    `is_new` represents the way the class is executed.
    If the value is `True' then the new model will be created,
    Otherwise an existing model will be used
    """

    def __init__(self, is_new, **kwargs) -> None:
        self.prediction_behavior = kwargs.get("purpose")
        self.logger = logging.getLogger(__name__)
        if is_new:
            self.optimizer = kwargs.get("optimizer", "adam")
            self.metrics = kwargs.get("metrics", "accuracy")
            self.loss_function = kwargs.get("loss_function", "mse")
            self.embedding_size = kwargs.get("embedding_size")
            self.subject_input_name = kwargs.get("subject_input_name")
            self.item_input_name = kwargs.get("item_input_name")
            self.subject_embedding_name = kwargs.get("subject_embedding_name")
            self.item_embedding_name = kwargs.get("item_embedding_name")
            try:
                self.subject_input = tf.keras.layers.Input(
                    shape=(1,), name=kwargs.get("subject_input_name")
                )
                self.logger.info("Subject input layer created")
            except Exception as ex:
                self.logger.error(
                    f"Error while creating subject input layer: {ex}", exc_info=True
                )
            try:
                self.item_input = tf.keras.layers.Input(
                    shape=(1,), name=kwargs.get("item_input_name")
                )
                self.logger.info("Item input layer created")
            except Exception as ex:
                self.logger.error(
                    f"Error while creating item input layer: {ex}", exc_info=True
                )
            self.checkpoint_file = path.join(
                kwargs.get("work_dir"), kwargs.get("checkpoint_file")
            )
            self.epochs = kwargs.get("epochs", 10)
            self.batch_size = kwargs.get("batch_size", 500)

    def fit_model(self, X_train, y_train, **kwargs):

        # Perform train-test split on the data
        tts_val = TrainTestSplit(x=X_train, y=y_train, train_pct=0.8)
        x_train = tts_val.train_x
        y_train = tts_val.train_y
        x_val = tts_val.test_x
        y_val = tts_val.test_y

        """"
        User input name should be the same as the name of the column in the input data
        """
        subject_data_train = x_train[self.subject_input_name]  # array of subject ids
        item_data_train = x_train[self.item_input_name]  # array of item idsß
        for col in x_train.columns:
            x_val = x_val[x_val[col].isin(x_train[col])]
            y_val = y_val[y_val.index.isin(x_val.index)]

        subject_data_val = x_val[self.subject_input_name]
        item_data_val = x_val[self.item_input_name]

        # Model checkpoint callback to save the model with the best performance on validation set
        checkpointer = ModelCheckpoint(
            filepath=self.checkpoint_file, verbose=0, save_best_only=True
        )

        # Define early stopping callback
        early_stop = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        subject_input_dim = subject_data_train.max() + 1
        item_input_dim = item_data_train.max() + 1
        self.logger.info("Subject input cardinality: %s", subject_input_dim)
        self.logger.info("Item input cardinality: %s", item_input_dim)

        if subject_input_dim == 0 or item_input_dim == 0:
            raise ValueError("Input cardinality cannot be zero")

        try:
            self.subject_embedding = tf.keras.layers.Embedding(
                input_dim=subject_input_dim,
                output_dim=self.embedding_size,
                name=self.subject_embedding_name,
            )(self.subject_input)
            self.subject_embedding = tf.keras.layers.Flatten()(self.subject_embedding)
            self.logger.info("Subject embedding layer created")
        except Exception as e:
            self.logger.error("Error in subject embedding: {}".format(e))
            raise
        try:
            self.item_embedding = tf.keras.layers.Embedding(
                input_dim=item_input_dim,
                output_dim=self.embedding_size,
                name=self.item_embedding_name,
            )(self.item_input)
            self.item_embedding = tf.keras.layers.Flatten()(self.item_embedding)
            self.logger.info("Item embedding layer created")
        except Exception as e:
            self.logger.error("Error in item embedding: {}".format(e))
            raise

        # Flatten embeddings and concatenate
        subject_vecs = tf.keras.layers.Flatten()(self.subject_embedding)
        item_vecs = tf.keras.layers.Flatten()(self.item_embedding)
        tf.keras.layers.Concatenate()([subject_vecs, item_vecs])

        # Define dot product layer
        dot = tf.keras.layers.Dot(axes=1, name="dot")([subject_vecs, item_vecs])

        # Define model with dot product as output
        self.model = tf.keras.models.Model(
            inputs=[self.subject_input, self.item_input], outputs=[dot]
        )

        # Compile model with mean squared error loss function
        self.model.compile(
            optimizer=self.optimizer, loss=self.loss_function, metrics=[self.metrics]
        )

        self.history = self.model.fit(
            x=[subject_data_train, item_data_train],
            y=y_train,
            validation_data=(
                [subject_data_val, item_data_val],
                y_val,
            ),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop, checkpointer],
        )

        self.item_embedding_layer = self.model.get_layer(
            self.item_embedding_name
        ).get_weights()[0]

        self.subject_embedding_layer = self.model.get_layer(
            self.subject_embedding_name
        ).get_weights()[0]

        # load weights from best model
        self.model.load_weights(self.checkpoint_file)
        return self.model

    def save_model(self, file_path, model_obj):
        model_obj.save(file_path)


@model_decorator(supervised=True)
class KerasNN(object):
    """
    `is_new` represents the way the class is executed.
    If the value is `True' then the new model will be created,
    Otherwise an existing model will be used
    """

    def __init__(self, is_new, **kwargs) -> None:
        self.prediction_behavior = kwargs.get("purpose")
        if is_new:
            nodes = kwargs.get("nodes")
            input_dim = kwargs.get("input_dim")
            loss_function = kwargs.get("loss_function", "binary_crossentropy")
            hl_activation = kwargs.get("hl_activation", "relu")
            ol_activation = kwargs.get("ol_activation", "sigmoid")
            initializer = kwargs.get("initializer", "random_normal")
            optimizer = kwargs.get("optimizer", "adam")
            metrics = kwargs.get("metrics", "accuracy")
            self.checkpoint_file = path.join(
                kwargs.get("work_dir"), kwargs.get("checkpoint_file")
            )
            self.early_stopping_patience = kwargs.get("early_stopping_patience", 10)
            self.early_stopping_delta = kwargs.get("early_stopping_delta", 1e-4)
            self.epochs = kwargs.get("epochs", 100)
            self.batch_size = kwargs.get("batch_size", 100)
            ker_reg = kwargs.get("ker_reg", 1e-6)
            dropout = kwargs.get("dropout", 0.01)
            labels_dim = kwargs.get("labels_dim", 2)
            if kwargs.get("ker_reg_type", "l1") == "l1":
                kernel_regularizer = l1(ker_reg)
            else:
                kernel_regularizer = l2(ker_reg)

            # Initialize a new Sequential model
            self.model = models.Sequential()
            # hidden layers
            for h, _ in enumerate(nodes):
                if h == 0:
                    self.model.add(
                        Dense(
                            nodes[h],
                            input_dim=input_dim,
                            activation=hl_activation,
                            kernel_initializer=initializer,
                        )
                    )
                elif h < len(nodes) - 1:
                    self.model.add(Dropout(dropout))
                elif h > 0:
                    self.model.add(
                        Dense(
                            nodes[h],
                            activation=hl_activation,
                            kernel_regularizer=kernel_regularizer,
                        )
                    )
            # Adding output layer
            self.model.add(
                Dense(
                    labels_dim,
                    activation=ol_activation,
                    kernel_initializer=initializer,
                )
            )
            # Compile the model
            self.model.compile(
                loss=loss_function, optimizer=optimizer, metrics=[metrics]
            )

    def fit_model(self, X_train, y_train, **kwargs):
        # Early stopping callback to stop training when a specified metric has stopped improving
        monitor = EarlyStopping(
            monitor=kwargs.get("monitor", "val_loss"),  # specify metric to monitor
            min_delta=self.early_stopping_delta,  # minimum change in the monitored quantity to qualify as an improvement
            patience=self.early_stopping_patience,  # number of epochs with no improvement after which training will be stopped
            verbose=1,
            mode=kwargs.get("mode", "auto"),
        )
        # Model checkpoint callback to save the model with the best performance on validation set
        checkpointer = ModelCheckpoint(
            filepath=self.checkpoint_file, verbose=0, save_best_only=True
        )

        # Perform train-test split on the data
        tts_val = TrainTestSplit(x=X_train, y=y_train, train_pct=0.8)

        # Train the model
        self.model.fit(
            tts_val.train_x,
            tts_val.train_y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=2,
            callbacks=[
                monitor,
                checkpointer,
            ],  # use the callbacks defined above during training
            validation_data=(tts_val.test_x, tts_val.test_y),
        )
        # load weights from best model
        self.model.load_weights(self.checkpoint_file)
        return self.model

    def predict_model(self, model_obj, X):
        if self.prediction_behavior == "classification":
            results = model_obj.predict(X)
            return [result_to_proba(i) for i in results]
        else:
            return model_obj.predict(X)

    def save_model(self, file_path, model_obj):
        model_obj.save(file_path)

    def load_model(self, model_path):
        return models.load_model(model_path)


@model_decorator(supervised=True)
class StochasticGradDescentClass(SGDClassifier):
    def __init__(self, is_new=True, **kwargs):
        if is_new:
            super().__init__(
                loss_function=kwargs.get("loss_function", "log_loss"),
                penalty=kwargs.get("penalty", "l1"),
                tol=kwargs.get("tol"),
                seed=kwargs.get("seed"),
                n_jobs=int(kwargs.get("n_jobs", 2)),
                warm_start=toolbox.parse_bool(kwargs.get("warm_start", "True")),
                shuffle=toolbox.parse_bool(kwargs.get("shuffle", "True")),
                alpha=float(kwargs.get("alpha", 1e-4)),
                max_iter=int(kwargs.get("max_iter", 100)),
            )

    def fit_model(self, X_train, y_train):
        return self.fit(X_train, y_train)

    def predict_model(self, model_obj, X):
        return model_obj.predict_proba(X)

    def save_model(self, file_path, model_obj):
        return joblib.dump(model_obj, file_path)

    def load_model(self, model_path):
        return joblib.load(path.join(model_path, am_sys_definitions.MODEL_PICKLE_FILE))


@model_decorator(supervised=True)
class LinearReg(object):
    def __init__(self, is_new=True, **kwargs):

        if is_new:
            if kwargs.get("reg_type", "Lasso") == "Lasso":
                self.reg_class = Lasso
            else:
                self.reg_class = Ridge
            self.alpha = float(kwargs.get("alpha", 1e-6))
            self.max_iter = int(kwargs.get("max_iter", 1000))
            self.tol = kwargs.get("tol", 1e-3)
            self.fit_intercept = toolbox.parse_bool(kwargs.get("fit_intercept", "True"))
            self.sample_weight = kwargs.get("sample_weight")
            self.multioutput = kwargs.get("multioutput", "uniform_average")
            self.normalize = toolbox.parse_bool(kwargs.get("normalize", "True"))

            self.model = self.reg_class(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter,
                tol=self.tol,
                normalize=self.normalize,
            )

    def fit_model(self, X_train, y_train):
        return self.model.fit(X_train, y_train)

    def predict_model(self, model_obj, X):
        return model_obj.predict(X)

    def save_model(self, file_path, model_obj):
        return joblib.dump(
            model_obj, path.join(file_path, am_sys_definitions.MODEL_PICKLE_FILE)
        )

    def load_model(self, model_path):
        return joblib.load(path.join(model_path, am_sys_definitions.MODEL_PICKLE_FILE))


@model_decorator(supervised=True)
class TreeReg(DecisionTreeRegressor):
    def __init__(self, is_new=True, **kwargs):
        if is_new:
            super().__init__(
                reg_behavior=kwargs.get("reg_behavior", "regression"),
                criterion=kwargs.get("loss_function", "mse"),
                splitter=kwargs.get("splitter", "best"),
                max_depth=toolbox.parse_int(kwargs.get("max_depth")),
                min_samples_split=int(kwargs.get("min_samples_split", 30)),
                min_samples_leaf=int(kwargs.get("min_samples_leaf", 30)),
                min_weight_fraction_leaf=float(
                    kwargs.get("min_weight_fraction_leaf", 0.0)
                ),
                max_features=kwargs.get("max_features", "auto"),
                random_state=toolbox.parse_int(kwargs.get("random_state")),
                ccp_alpha=float(kwargs.get("ccp_alpha", 0.0)),
            )

    def fit_model(self, X_train, y_train):
        return self.fit(X_train, y_train)

    def predict_model(self, model_obj, X):
        return model_obj.predict(X)

    def save_model(self, file_path, model_obj):
        return joblib.dump(
            model_obj, path.join(file_path, am_sys_definitions.MODEL_PICKLE_FILE)
        )

    def load_model(self, model_path):
        return joblib.load(path.join(model_path, am_sys_definitions.MODEL_PICKLE_FILE))


@model_decorator(supervised=True)
class TreeClass(DecisionTreeClassifier):
    def __init__(self, is_new=True, **kwargs):
        self.logger = logging.getLogger(__name__)
        if is_new:
            super().__init__(
                reg_behavior=kwargs.get("reg_behavior", "regression"),
                criterion=kwargs.get("loss_function", "mse"),
                splitter=kwargs.get("splitter", "best"),
                max_depth=toolbox.parse_int(kwargs.get("max_depth")),
                min_samples_split=int(kwargs.get("min_samples_split", 30)),
                min_samples_leaf=int(kwargs.get("min_samples_leaf", 30)),
                min_weight_fraction_leaf=float(
                    kwargs.get("min_weight_fraction_leaf", 0.0)
                ),
                max_features=kwargs.get("max_features", "auto"),
                random_state=toolbox.parse_int(kwargs.get("random_state")),
                ccp_alpha=float(kwargs.get("ccp_alpha", 0.0)),
            )
        self.prediction_behavior = kwargs.get("prediction_behavior", "predict_proba")

    def fit_model(self, X_train, y_train):
        try:
            fitted_model = self.fit(X_train, y_train)
            self.logger.info("Model fitted successfully")
        except Exception as ex:
            self.logger.error("Error in fitting model: {}".format(ex))
            raise
        return fitted_model

    def predict_model(self, model_obj, X):
        if self.prediction_behavior == "predict_proba":
            return model_obj.predict_proba(X)
        elif self.prediction_behavior == "predict_log_proba":
            return model_obj.predict_log_proba(X)
        else:
            return model_obj.predict(X)

    def save_model(self, file_path, model_obj):
        return joblib.dump(
            model_obj, path.join(file_path, am_sys_definitions.MODEL_PICKLE_FILE)
        )

    def load_model(self, model_path):
        return joblib.load(path.join(model_path, am_sys_definitions.MODEL_PICKLE_FILE))


@model_decorator(supervised=True)
class XGboostClassifier(object):
    def __init__(self, is_new=True, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.is_new = is_new
        self.enable_categorical = kwargs.get("enable_categorical", False)
        if self.enable_categorical:
            tree_method = "hist"
        else:
            tree_method = kwargs.get("tree_method", "auto")
        if is_new:
            self.model = xgboost.XGBClassifier(
                n_estimators=toolbox.parse_int(kwargs.get("n_estimators", 100)),
                max_depth=toolbox.parse_int(kwargs.get("max_depth", 3)),
                learning_rate=float(kwargs.get("learning_rate", 0.1)),
                objective=kwargs.get("objective", "multi:softprob"),
                max_leaves=toolbox.parse_int(kwargs.get("max_leaves", 0)),
                enable_categorical=self.enable_categorical,
                tree_method=tree_method,
            )

    def fit_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self.model

    def predict_model(self, model_obj, X):
        try:
            if self.is_new:
                self.logger.info("Predicting with new model")
                predictions = self.model.predict_proba(X)
            else:
                self.logger.info(
                    "Predicting with old model, enable categorical: {}".format(
                        self.enable_categorical
                    )
                )
                dmatrix_x = xgboost.DMatrix(
                    X, enable_categorical=self.enable_categorical
                )
                self.logger.info("Converted to DMatrix")
                predictions = model_obj.predict(dmatrix_x)
            self.logger.info("Predictions made successfully")
        except Exception as ex:
            self.logger.error("Error in making predictions: {}".format(ex))
            raise
        return [result_to_proba(i) for i in predictions]

    def save_model(self, file_path, model_obj):
        return model_obj.save_model(
            path.join(file_path, am_sys_definitions.MODEL_JSON_FILE)
        )

    def load_model(self, model_path):
        loaded_model = xgboost.Booster()
        full_path = path.join(model_path, am_sys_definitions.MODEL_JSON_FILE)
        self.logger.info("Getting model from {}".format(full_path))
        try:
            loaded_model.load_model(full_path)
            self.logger.info("Model loaded successfully")
        except Exception as ex:
            self.logger.error("Error in loading model: {}".format(ex))
            raise
        return loaded_model


@model_decorator(supervised=True)
class XGboostRegression(object):
    def __init__(self, is_new=True, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.is_new = is_new
        self.enable_categorical = kwargs.get("enable_categorical", False)
        if self.enable_categorical:
            tree_method = "hist"
        else:
            tree_method = kwargs.get("tree_method", "auto")
        if is_new:
            self.model = xgboost.XGBRegressor(
                n_estimators=toolbox.parse_int(kwargs.get("n_estimators", 100)),
                max_depth=toolbox.parse_int(kwargs.get("max_depth", 3)),
                learning_rate=float(kwargs.get("learning_rate", 0.1)),
                objective=kwargs.get("objective", "reg:squarederror"),
                max_leaves=toolbox.parse_int(kwargs.get("max_leaves", 0)),
                enable_categorical=self.enable_categorical,
                tree_method=tree_method,
            )

    def fit_model(self, X_train, y_train):
        self.logger.info("Ebable categorical: {}".format(self.enable_categorical))
        self.model.fit(
            X_train,
            y_train,
        )
        return self.model

    def predict_model(self, model_obj, X):
        try:
            if self.is_new:
                self.logger.info("Predicting with new model")
                predictions = self.model.predict(X)
            else:
                self.logger.info(
                    "Predicting with old model, enable categorical: {}".format(
                        self.enable_categorical
                    )
                )
                dmatrix_x = xgboost.DMatrix(
                    X, enable_categorical=self.enable_categorical
                )
                self.logger.info("Converted to DMatrix")
                predictions = model_obj.predict(dmatrix_x)
            self.logger.info("Predictions made successfully")
        except Exception as ex:
            self.logger.error("Error in making predictions: {}".format(ex))
            raise
        return predictions

    def save_model(self, file_path, model_obj):
        return model_obj.save_model(
            path.join(file_path, am_sys_definitions.MODEL_JSON_FILE)
        )

    def load_model(self, model_path):
        loaded_model = xgboost.Booster()
        full_path = path.join(model_path, am_sys_definitions.MODEL_JSON_FILE)
        self.logger.info("Getting model from {}".format(full_path))
        try:
            loaded_model.load_model(full_path)
            self.logger.info("Model loaded successfully")
        except Exception as ex:
            self.logger.error("Error in loading model: {}".format(ex))
            raise
        return loaded_model


@model_decorator(supervised=False)
class ClusterKmeans(object):
    def __init__(self, is_new=True, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.is_new = is_new
        if is_new:
            self.model = KMeans(
                n_clusters=kwargs.get("n_clusters"),
                init=kwargs.get("init", "k-means++"),
                n_init=kwargs.get("n_init", 10),
                max_iter=kwargs.get("max_iter", 300),
                tol=kwargs.get("tol", 0.0001),
                verbose=kwargs.get("verbose", 0),
                random_state=kwargs.get("random_state"),
                copy_x=kwargs.get("copy_x", True),
                algorithm=kwargs.get("algorithm", "auto"),
            )

    def fit_model(self, X_train, **kwargs):
        model = self.model.fit(X_train)
        self.logger.info("Cluster centers: \n {}".format(model.cluster_centers_))
        return model

    def predict_model(self, model_obj, X):
        if self.is_new:
            predictions = self.model.predict(X).tolist()
        else:
            predictions = model_obj.predict(X).tolist()
        return predictions

    def save_model(self, file_path, model_obj):
        return joblib.dump(
            model_obj, path.join(file_path, am_sys_definitions.MODEL_PICKLE_FILE)
        )

    def load_model(self, model_path):

        return joblib.load(path.join(model_path, am_sys_definitions.MODEL_PICKLE_FILE))


@model_decorator(supervised=False)
class BG_NBD(object):
    def __init__(self, is_new=True, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.penalizer_coef = kwargs.get("penalizer_coef", 0)
        self.customer_id_col = kwargs.get("customer_id_col")
        self.freq = kwargs.get("freq")
        self.recency = "recency"
        self.frequency = "frequency"
        self.monetary_value = "monetary_value"
        self.T = "T"
        self.datetime_col = kwargs.get("datetime_col")
        self.monetary_value_col = kwargs.get("monetary_value_col")
        self.max_t = kwargs.get("max_t", 0)
        self.prediction_window = "prediction_window"

    def summary(self, X):
        data = summary_data_from_transaction_data(
            X,
            customer_id_col=self.customer_id_col,
            freq=self.freq,
            datetime_col=self.datetime_col,
            monetary_value_col=self.monetary_value_col,
        )
        self.logger.info("Summary data sample: \n {}".format(data.head(5)))
        if self.max_t > 0:
            data[self.T] = self.max_t
        return data

    def fit_model(self, X_train):
        summary_data = self.summary(X_train)
        bgf = BetaGeoFitter(self.penalizer_coef)
        return bgf.fit(
            summary_data[self.frequency],
            summary_data[self.recency],
            summary_data[self.T],
        )

    def save_model(self, file_path, model_obj):
        return model_obj.save_model(
            path.join(file_path, am_sys_definitions.MODEL_PICKLE_FILE)
        )

    def load_model(self, model_path):
        bgf = BetaGeoFitter(self.penalizer_coef)
        bgf.load_model(path.join(model_path, am_sys_definitions.MODEL_PICKLE_FILE))
        return bgf

    def predict_model(self, model_obj, X: pd.DataFrame):
        try:
            visits = model_obj.conditional_expected_number_of_purchases_up_to_time(
                X[self.prediction_window],
                X[self.frequency],
                X[self.recency],
                X[self.T],
            )
        except Exception as ex:
            self.logger.error("Error in making visits predictions: {}".format(ex))
            raise

        try:
            unit_visits = model_obj.conditional_expected_number_of_purchases_up_to_time(
                1,
                X[self.frequency],
                X[self.recency],
                X[self.T],
            )
        except Exception as ex:
            self.logger.error("Error in making unit visits predictions: {}".format(ex))
            raise

        try:
            p_alive_global = model_obj.conditional_probability_alive(
                X[self.frequency],
                X[self.recency],
                X[self.T],
            )
        except Exception as ex:
            self.logger.error(
                "Error in making p_alive_global predictions: {}".format(ex)
            )
            raise

        p_alive = 1 - 1 / (np.exp(X[self.prediction_window] * unit_visits))

        return [
            [
                {"measure": "visits", "value": val[0]},
                {"measure": "p_alive", "value": val[1]},
                {"measure": "p_alive_global", "value": val[2]},
            ]
            for val in zip(visits, p_alive, p_alive_global)
        ]


@model_decorator(supervised=False)
class GG_NBD(object):
    def __init__(self, is_new=True, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.penalizer_coef = kwargs.get("penalizer_coef", 0)
        self.customer_id_col = kwargs.get("customer_id_col")
        self.freq = kwargs.get("freq")
        self.recency = "recency"
        self.frequency = "frequency"
        self.monetary_value = "monetary_value"
        self.T = "T"
        self.datetime_col = kwargs.get("datetime_col")
        self.monetary_value_col = kwargs.get("monetary_value_col")
        self.max_t = kwargs.get("max_t", 0)
        self.prediction_window = "prediction_window"

    def fit_model(self, X_train):
        summary_data = summary_data_from_transaction_data(
            X_train[X_train[self.monetary_value_col] > 0],
            customer_id_col=self.customer_id_col,
            freq=self.freq,
            datetime_col=self.datetime_col,
            monetary_value_col=self.monetary_value_col,
        )
        returning_customers_summary = summary_data[summary_data[self.frequency] > 0]
        ggf = GammaGammaFitter(self.penalizer_coef)
        return ggf.fit(
            returning_customers_summary[self.frequency],
            returning_customers_summary[self.monetary_value],
        )

    def save_model(self, file_path, model_obj):
        return model_obj.save_model(
            path.join(file_path, am_sys_definitions.MODEL_PICKLE_FILE)
        )

    def load_model(self, model_path):
        ggf = GammaGammaFitter(self.penalizer_coef)
        ggf.load_model(path.join(model_path, am_sys_definitions.MODEL_PICKLE_FILE))
        return ggf

    def predict_model(self, model_obj, X):
        avg_profit = model_obj.conditional_expected_average_profit(
            X[self.frequency], X[self.monetary_value]
        )

        return [
            [
                {"measure": "avg_profit", "value": val},
            ]
            for val in avg_profit
        ]
