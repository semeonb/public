import logging
from sklearn.metrics import (
    log_loss,
    mean_squared_error,
    mean_absolute_error,
    silhouette_score,
)
from airmelt_data.gcp import Bigquery


class Model(object):
    """
    Abstract representation of ML model
    """

    def __init__(self, model_class, is_new=True, **kwargs: dict) -> None:
        """
        model_class: one of the model classes in the SUPPORTED_MODELS file
        is_new: indication if the model is a new or existing one
        """
        self.logger = logging.getLogger(__name__)
        self.loss_function_name = kwargs.get("loss_function")
        if self.loss_function_name:
            self.logger.info("Loss function name: {}".format(self.loss_function_name))
        try:
            self.logger.info("Initializing model: {}".format(model_class))
            self.model = model_class(is_new, **kwargs)
            self.logger.info("Model initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize model: {}".format(e))
            self.logger.error(model_class)
            print(e)
            raise

    def fit(self, X_train, y_train, **kwargs):
        """
        X_train: Training dataset
        y_train: Labels dataset. Is null when the model is unsupervised
        """
        if len(y_train) > 0:
            try:
                res = self.model.fit_model(X_train, y_train, **kwargs)
                self.logger.info("Model fitted successfully")
            except Exception as e:
                self.logger.error("Failed to fit model: {}".format(e))
                raise
        else:
            res = self.model.fit_model(X_train)
        return res

    def loss_function(self):
        if self.loss_function_name in ["mse", "mean_squared_error"]:
            return mean_squared_error
        elif self.loss_function_name in ["mae", "mean_absolute_error"]:
            return mean_absolute_error
        elif self.loss_function_name in ["log_loss", "binary_crossentropy", "log"]:
            return log_loss
        elif self.loss_function_name in ["silhouette_score"]:
            return silhouette_score
        elif not self.loss_function_name:
            return lambda x, y: None
        else:
            raise Exception("unknown loss function name")

    def predict(self, model_obj, X):
        """
        model_obj: Model pickled object
        X: Prediction dataset
        """
        return self.model.predict_model(model_obj, X)

    def loss(self, y_true, y_pred):
        """
        y_true: true labels
        y_pred: predicted labels
        """
        loss_fuct_obj = self.loss_function()
        self.logger.info("Loss function: {}".format(loss_fuct_obj))
        self.logger.info("y_true: {}".format(y_true[0:4]))
        self.logger.info("y_pred: {}".format(y_pred[0:4]))
        return loss_fuct_obj(y_true, y_pred)

    def save(self, file_path, client: Bigquery, dest_bucket, dest_blob_name, model):
        """
        file_path: OS path to pull the data from
        client: BQ client object
        dest_bucket: GS bucket name
        dest_blob_name: destination GS object name
        model: Model picked object
        """
        self.model.save_model(file_path, model)
        return client.upload_from_directory(
            directory_path=file_path,
            dest_bucket_name=dest_bucket,
            dest_blob_name=dest_blob_name,
        )

    def load(self, model_path):
        """
        model_path: Full model path
        """
        return self.model.load_model(model_path)
