from os import path
import joblib
import logging

from airmelt_data import ml, am_sys_definitions, supported_models
from airmelt_system import toolbox


class Model(object):
    def __init__(self, model_name) -> None:
        self.model = model_name


class Submodel(Model):
    def __init__(
        self,
        model_config: dict,
        submodel_name: str,
        model_name: str,
        prod_model_folder: str,
    ) -> None:
        """
        model_config: model cofiguration dictionary
        submodel_name: Submodel Name
        model_name: Model Name
        prod_model_folder: Local production model folder
        """
        super().__init__(model_name)
        self.logger = logging.getLogger(__name__)
        self.model_config = model_config
        mcl = supported_models.SUPPORTED_MODEL_TYPES[self.model_config["type"]]
        self.submodel_name = submodel_name
        self.prod_model_folder = prod_model_folder
        self.prod_model_binary_loc = path.join(
            self.prod_model_folder, am_sys_definitions.MODEL_OBJECT_NAME
        )

        if self.model_config.get("primary_components", 0) > 0:
            self.logger.info(
                "Loading primary components file, PC count: {}".format(
                    self.model_config.get("primary_components")
                )
            )
            try:
                pca_path = path.join(
                    self.prod_model_folder, am_sys_definitions.PCA_PICKLE_FILE
                )
                self.pca_model = joblib.load(filename=pca_path)
            except Exception as ex:
                self.logger.error(
                    "No PCA transformation file was found at {pp}: {ex}".format(
                        pp=pca_path, ex=ex
                    )
                )
        else:
            self.pca_model = None

        if self.model_config.get("skip_preprocessing") == "enable_categorical":
            self.enable_categorical = True
        else:
            self.enable_categorical = False

        model_params = self.model_config.copy()
        model_params["enable_categorical"] = self.enable_categorical

        # Existing model is an abstract representation of the model
        self.existing_model = ml.Model(model_class=mcl, is_new=False, **model_params)

        try:
            self.logger.info(
                "Loading model object from: {}".format(self.prod_model_binary_loc)
            )
            self.obj = self.existing_model.load(self.prod_model_binary_loc)
            self.logger.info("Model object {} loaded successfully".format(self.obj))
        except Exception as ex:
            self.logger.error("Failed to load model object: {}".format(ex))
            raise
        try:
            self.expanded_columns = toolbox.open_txt_file(
                path.join(
                    self.prod_model_folder,
                    am_sys_definitions.EXPANDED_COLUMNS_FILE,
                )
            )
        except OSError as ex:
            self.logger.warning("Expanded columns file was not found!")
            self.expanded_columns = []

    def _import_columns_file(self, filename=am_sys_definitions.COLUMNS_FILE):
        return toolbox.open_json_file(path.join(self.prod_model_folder, filename))

    def get_column_definition(self):
        prod_model_columns = dict()
        columns = self._import_columns_file()
        for i, v in columns.items():
            column_definition = toolbox.open_json_file(
                path.join(self.prod_model_folder, v)
            )
            if column_definition.get("type") == am_sys_definitions.REGULAR:
                column_definition["scaler"] = joblib.load(
                    filename=path.join(
                        self.prod_model_folder,
                        am_sys_definitions.COLUMNS_FOLDER,
                        column_definition.get("scaler_file"),
                    )
                )
            elif column_definition.get("type") == am_sys_definitions.MULTILABEL:
                column_definition["mlb"] = joblib.load(
                    filename=path.join(
                        self.prod_model_folder,
                        am_sys_definitions.COLUMNS_FOLDER,
                        column_definition.get("mlb_file"),
                    )
                )
            else:
                if column_definition.get("type") != am_sys_definitions.CATEGORIAL:
                    raise Exception("Unknown data type")
            prod_model_columns[i] = column_definition
        return prod_model_columns

    def predict(self, X):
        """
        X: independent dataset
        """
        return self.existing_model.predict(self.obj, X)

    def loss(self, y_true, y_pred):
        """
        y_true: true values
        y_pred: predicted values
        """
        return self.existing_model.loss(y_true, y_pred)

    def predict_loss(self, X, y_true):
        """
        X: independent dataset
        y_true: true values
        """
        predicted_values = self.predict(X)
        loss = self.loss(y_true, predicted_values)
        return loss
