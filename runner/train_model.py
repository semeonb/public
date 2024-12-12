#!/usr/bin/env python3
"""
train_model.py
Author: Semeon Balagula @ Airmelt
The script will run the model and/or its submodels, save the results in Google Storage, also will compare
the new model performance to the one in production and replace the production model if needed.


Usage:
    train_model.py [--help] (--model_name <value>) [--submodels <value>] 
                   [--train_pct <value>] [--prod] [--skip_copy_gs_files] 
                   [--config_file_location <value>]

                       

Options:
    --help                          Show this screen and exit
    --model_name <value>            Specify model name
    --submodels <value>             Submodels in the format: "[x, y, z]", if not specified, all submodels will be run [default: []]
    --train_pct <value>             Percent of data to leave out for training [default: 0.8]
    --prod                          If specified, prod environment will be invoked
    --skip_copy_gs_files            If scpecified, existing files will not be copied
    --config_file_location <value>  Specify the location of the config file [default: /home/prod/config.json]

"""

from docopt import docopt
from os import path
import logging
import json
import joblib
from datetime import datetime
import pandas as pd

from core import (
    db,
    gcp,
    dataprep,
    ml,
    am_sys_definitions,
    supported_models,
    existing,
)
from system import toolbox

args = docopt(__doc__)

try:
    config_file = toolbox.open_json_file(
        config_file=toolbox.open_json_file(args["--config_file_location"])
    )
    if args["--prod"]:
        SYSTEM_CONFIG: dict = config_file["prod"]
    else:
        SYSTEM_CONFIG: dict = config_file["dev"]
except Exception as ex:
    print("System config file was not found!")
    raise

DB_TYPE = SYSTEM_CONFIG.get("db_type")
WORK_DIR = SYSTEM_CONFIG.get("work_dir")
BQ_PROJECT_ID = SYSTEM_CONFIG.get("bq_project_id")
GS_PROJECT_ID = SYSTEM_CONFIG.get("gs_project_id")
GS_BUCKET_NAME = SYSTEM_CONFIG.get("gs_bucket_name")
GS_FOLDER_NAME = SYSTEM_CONFIG.get("gs_folder_name")


class TempModel(object):
    def __init__(
        self,
        global_config: dict,
        model_name: str,
        train_pct: float,
        **kwargs,
    ) -> None:
        """
        Creates a temporary model
        global_config: Dictionary containing model metadata
        model_name: Model Name
        train_pct: value between 0 and 1 to allocate for training purposes
        """
        self.logger = logging.getLogger(__name__)
        self.global_config = global_config
        self.model_name = model_name
        self.train_pct = train_pct
        self.queries_folder = path.join(
            WORK_DIR, model_name, am_sys_definitions.QUERIES_FOLDER
        )
        self.temp_model_folder = path.join(
            WORK_DIR, model_name, am_sys_definitions.TEMPORARY_MODELS_FOLDER
        )
        self.bq = gcp.Bigquery(bq_project_id=BQ_PROJECT_ID, gs_project_id=GS_PROJECT_ID)
        self.model_class = supported_models.SUPPORTED_MODEL_TYPES[
            self.global_config["type"]
        ]

        if self.global_config["purpose"] == "classification":
            self.continious_label = False
        else:
            self.continious_label = True
        self.logger.info("Continious label: {}".format(self.continious_label))


class TempSubModel(TempModel):
    def __init__(
        self,
        global_config: dict,
        model_name: str,
        train_pct: float,
        submodel_name: str,
        values,
    ) -> None:
        """
        global_config: Dictionary containing model metadata
        model_name: Model Name
        train_pct: value between 0 and 1 to allocate for training purposes
        submodel_name: Submodel Name
        value: Dictionary with model-specific
        """
        super().__init__(global_config, model_name, train_pct)
        self.submodel_name = submodel_name
        self.values = values
        self.model = None
        self.model_config = self.global_config
        if self.model_config.get("skip_preprocessing") == "enable_categorical":
            self.model_config["enable_categorical"] = True
        else:
            self.model_config["enable_categorical"] = False
        self.submodel_folder = path.join(self.temp_model_folder, self.submodel_name)
        self.gs_path = path.join(
            GS_FOLDER_NAME,
            self.model_name,
            am_sys_definitions.TEMPORARY_MODELS_FOLDER,
            submodel_name,
        )
        self.submodel_gs_loc = path.join(self.gs_path, self.model_config["model_file"])
        self.logger.info("GS submodel location: {}".format(self.submodel_gs_loc))

        self.submodel_local_loc = path.join(
            self.submodel_folder, self.model_config["model_file"]
        )

        # Create local folder if needed
        toolbox.create_local_dir(self.submodel_folder)

        self.logger.info("Local submodel location: {}".format(self.submodel_local_loc))
        self.model_config["work_dir"] = self.submodel_folder

        self.logger.info("Getting raw data")
        queries = db.QueryList(
            db_type=DB_TYPE,
            query_list=toolbox.gen_queries(self.queries_folder, self.values),
            bq_project_id=BQ_PROJECT_ID,
        )
        self.df = queries.to_dataframe(key_list=self.model_config.get("key_list"))
        self.logger.info("Ended getting raw data")

        self.loss = None
        self.item_embedding_layer = None
        self.train_x_shape = None
        self.test_x_shape = None
        self.train_y_shape = None
        self.test_y_shape = None
        self.mu = None
        self.pi = None

    def run_dataprep(self, prep_data, **kwargs):
        if kwargs.get("expanded_columns_list"):
            expanded_columns_list = kwargs.get("expanded_columns_list")
        else:
            expanded_columns_list = prep_data.prod_columns
        self.logger.info("saving the original columns")
        saved_loc = self.save_text(
            self.submodel_folder,
            {
                i: path.join(am_sys_definitions.COLUMNS_FOLDER, i + ".json")
                for i in prep_data.column_dict.keys()
            },
            self.gs_path,
            am_sys_definitions.COLUMNS_FILE,
        )
        self.logger.info("The originals columns file was saved at {}".format(saved_loc))
        self.logger.info("saving final columns list")
        self.save_text(
            self.submodel_folder,
            expanded_columns_list,
            self.gs_path,
            am_sys_definitions.EXPANDED_COLUMNS_FILE,
        )
        self.logger.info("Saving column definitions")
        self.save_column_definitions(prep_data, self.gs_path, self.submodel_folder)
        return prep_data.X, prep_data.Y

    def save_decoded_item_embedding(self, item_encoder, bq_client: gcp.Bigquery):
        item_embedding_name = self.model_config.get("item_embedding_name")
        self.logger.info("Saving item embedding")
        item_embedding_list = self.model.model.item_embedding_layer.tolist()
        all_item_encodings = item_encoder.inverse_transform(
            range(len(item_embedding_list))
        ).tolist()
        decoded_dict = dict(zip(all_item_encodings, item_embedding_list))

        output_list = []
        for k, v in decoded_dict.items():
            output_list.append(
                {
                    "item": k,
                    "embedding": [{"dim": i, "value": j} for i, j in enumerate(v)],
                }
            )

        decoded_dict_file = "{}.json".format(item_embedding_name)
        decoded_dict_file_loc = path.join(WORK_DIR, decoded_dict_file)

        with open(decoded_dict_file_loc, "w") as outfile:
            for item in output_list:
                outfile.write(json.dumps(item) + "\n")

        gs_destination_path = path.join(self.gs_path, decoded_dict_file)
        bq_client.localfile2gs(
            bucket=GS_BUCKET_NAME,
            source_path=decoded_dict_file_loc,
            dest_path=gs_destination_path,
            delete_local=True,
        )
        self.logger.info("Item embedding saved at {}".format(gs_destination_path))
        # LOAD TO BIGQUERY
        try:
            dataset = self.model_config["pred_destination_dataset"]
            table = self.model_config["pred_destination_table"]
            bq_client.load(
                dataset=dataset,
                table=table + self.submodel_name,
                path="gs://" + path.join(GS_BUCKET_NAME, gs_destination_path),
                source_format="NEWLINE_DELIMITED_JSON",
                write_flag="WRITE_TRUNCATE",
            )
        except Exception as ex:
            self.logger.error("Error loading item embedding to Bigquery: {}".format(ex))
        return gs_destination_path

    def run_train_test_split(self, X, Y):
        tts = dataprep.TrainTestSplit(x=X, y=Y, train_pct=self.train_pct)
        self.model_config["input_dim"] = tts.train_x.shape[1]
        if self.model_config.get("purpose") == "classification":
            self.model_config["labels_dim"] = tts.train_y.shape[1]
        else:
            self.model_config["labels_dim"] = 1

        self.train_x_shape = list(tts.train_x.shape)
        self.train_y_shape = list(tts.train_y.shape)
        self.test_x_shape = list(tts.test_x.shape)
        self.test_y_shape = list(tts.test_y.shape)

        shapes_msg = """
        X Train: {xtr},
        Y Train: {ytr},
        X Test: {xts},
        Y Test: {yts}
        """.format(
            xtr=self.train_x_shape,
            ytr=self.train_y_shape,
            xts=self.test_x_shape,
            yts=self.test_y_shape,
        )
        self.logger.info("Shapes: \n {}".format(shapes_msg))
        return tts

    def run(self):
        label_column = self.model_config.get("label_column")

        if self.model_config.get("skip_preprocessing") == "True":
            self.logger.info("Skipping preprocessing")
            if label_column:
                X = self.df.loc[:, self.df.columns != label_column]
                Y = self.df[label_column]
            else:
                X = self.df
                Y = []
            prep_data = None
        elif self.model_config.get("skip_preprocessing") == "enable_categorical":
            """
            Keep the categorical columns as is, and only preprocess the numerical columns
            """
            self.logger.info("Staring converting eligible columns to categorical")
            prep_data = dataprep.ProcessedData(
                df=self.df,
                label_column=label_column,
                continious_label=self.continious_label,
                key_list=self.model_config.get("key_list"),
                balance_ratio=self.model_config.get("balance_ratio", 0),
                low_info_thresh=self.model_config.get("low_info_thresh", 1),
                primary_components=self.model_config.get("primary_components", 0),
                purpose=self.model_config.get("purpose"),
            )
            self.mu, self.pi = prep_data.mu, prep_data.pi
            X = prep_data.pred
            Y = prep_data.Y
            self.logger.info("Saving column definitions")
            self.run_dataprep(prep_data, expanded_columns_list=list(X.columns.values))
            for col, col_detail in prep_data.column_dict.items():
                if col_detail["type"] == am_sys_definitions.CATEGORIAL:
                    X[col] = X[col].astype("category")
            self.logger.info("Ended converting eligible columns to categorical")
        else:
            self.logger.info("Starting preprocessing")
            prep_data = dataprep.ProcessedData(
                df=self.df,
                label_column=label_column,
                continious_label=self.continious_label,
                key_list=self.model_config.get("key_list"),
                balance_ratio=self.model_config.get("balance_ratio", 0),
                low_info_thresh=self.model_config.get("low_info_thresh", 1),
                primary_components=self.model_config.get("primary_components", 0),
                purpose=self.model_config.get("purpose"),
            )
            self.mu, self.pi = prep_data.mu, prep_data.pi
            X, Y = self.run_dataprep(prep_data)
            self.logger.info("Ended preprocessing")

        if (
            self.train_pct < 1
            and getattr(self.model_class, "__supervised__")
            and label_column
        ):
            self.logger.info("Running train-test split")
            tts = self.run_train_test_split(X, Y)
            train_x, train_y = tts.train_x, tts.train_y
            test_x, test_y = tts.test_x, tts.test_y

        else:
            train_x, train_y = X, Y
            test_x, test_y = X, Y

        self.logger.info("Data types: \n {}".format(X.dtypes))
        # Getting the model class
        try:
            self.model = ml.Model(model_class=self.model_class, **self.model_config)
        except Exception as ex:
            self.logger.error("Error getting the model class: {}".format(ex))
            raise
        try:
            model_obj = self.model.fit(train_x, train_y)
        except Exception as ex:
            self.logger.error("Error fitting the model class: {}".format(ex))
            raise
        try:
            bq = gcp.Bigquery(bq_project_id=BQ_PROJECT_ID, gs_project_id=GS_PROJECT_ID)
            self.logger.info("Saving train and test data")
            train_x_file = "{model_name}_x.csv".format(model_name=self.model_name)
            self.train_x_local_loc = path.join(WORK_DIR, train_x_file)
            train_y_file = "{model_name}_y.csv".format(model_name=self.model_name)
            self.train_y_local_loc = path.join(WORK_DIR, train_y_file)
            X.to_csv(self.train_x_local_loc, index=False)
            self.logger.info("Shape of X: {}".format(X.shape))
            bq.localfile2gs(
                bucket=GS_BUCKET_NAME,
                source_path=self.train_x_local_loc,
                dest_path=path.join(self.gs_path, train_x_file),
                delete_local=True,
            )
            y_df = pd.DataFrame(Y)
            y_df.to_csv(self.train_y_local_loc, index=False)
            # Upload to GS
            bq.localfile2gs(
                bucket=GS_BUCKET_NAME,
                source_path=self.train_y_local_loc,
                dest_path=path.join(self.gs_path, train_y_file),
                delete_local=True,
            )
        except Exception as ex:
            self.logger.error("Error saving train and test data: {}".format(ex))

        temp_model = ml.Model(model_class=self.model_class, is_new=False)
        try:
            temp_model.save(
                self.submodel_local_loc,
                client=self.bq,
                dest_bucket=GS_BUCKET_NAME,
                dest_blob_name=self.submodel_gs_loc,
                model=model_obj,
            )
            self.logger.info("saved model at {}".format(self.submodel_gs_loc))
        except Exception as ex:
            self.logger.error("Could not save the model: \{}".format(ex))
            raise
        if (
            self.model_config.get("loss_function")
            and self.model_config.get("purpose") != "embedding"
        ):
            self.logger.info("Calculating loss and predictions for the created model")
            try:
                if getattr(self.model_class, "__supervised__") and self.train_pct < 1:
                    predicted = self.model.predict(model_obj, X=test_x)
                    self.loss = self.model.loss(y_true=test_y, y_pred=predicted)
                else:
                    predicted = self.model.predict(model_obj, X=test_x)
                    self.loss = self.model.loss(y_true=test_x, y_pred=predicted)
            except Exception as ex:
                self.logger.error(
                    "Error calculating loss and predictions: {}".format(ex)
                )
                raise
        elif self.model_config.get("purpose") == "embedding":
            try:
                self.loss = self.model.model.history.history["loss"][-1]
            except Exception as ex:
                self.logger.warning("Error getting loss from history: {}".format(ex))
                self.loss = -1

            item_embed_column = self.model_config.get("item_input_name")
            item_embed_column_properties = prep_data.column_dict.get(item_embed_column)
            item_encoder = item_embed_column_properties.get("encoder")
            if not item_encoder:
                self.logger.error(
                    "Could not find item encoder for column {}".format(
                        item_embed_column
                    )
                )
                raise Exception("Could not find item encoder")
            try:
                self.save_decoded_item_embedding(item_encoder, bq_client=bq)
            except Exception as ex:
                self.logger.error("Error getting item embedding layer: {}".format(ex))
                raise
        else:
            self.loss = -1
        self.logger.info(
            "Submodel {sm} loss is: {ls}".format(sm=self.submodel_name, ls=self.loss)
        )
        return prep_data

    def save_statistics(
        self, statistics_file=am_sys_definitions.STATISTICS_FILE, **kwargs
    ):
        full_local_path = path.join(self.submodel_folder, statistics_file)
        statistics = dict()
        statistics["utc_timestamp"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        statistics["mu"] = self.mu
        statistics["pi"] = self.pi
        statistics["loss"] = self.loss
        statistics["train_x_shape"] = str(self.train_x_shape)
        statistics["train_y_shape"] = str(self.train_y_shape)
        statistics["test_x_shape"] = str(self.test_x_shape)
        statistics["test_y_shape"] = str(self.test_y_shape)
        for i, v in kwargs.items():
            statistics[i] = v
        json_object = json.dumps(statistics, indent=4)
        with open(full_local_path, "w") as outfile:
            outfile.write(json_object)
        save_loc = self.bq.localfile2gs(
            bucket=GS_BUCKET_NAME,
            source_path=full_local_path,
            dest_path=path.join(self.gs_path, statistics_file),
        )
        return save_loc

    def save_text(
        self,
        local_folder: str,
        values,
        gs_path: str,
        file_name: str,
    ):
        full_path = path.join(local_folder, file_name)
        full_gs_path = path.join(gs_path, file_name)
        toolbox.create_local_dir(local_folder)
        if file_name.endswith("txt"):
            list_values = ",".join(values)
            with open(full_path, "w") as text_file:
                print((list_values), file=text_file)
        else:
            json_object = json.dumps(values, indent=4)
            with open(full_path, "w") as outfile:
                outfile.write(json_object)
        save_loc = self.bq.localfile2gs(
            bucket=GS_BUCKET_NAME,
            source_path=full_path,
            dest_path=full_gs_path,
        )
        return save_loc

    def save_column_definitions(self, prep_data, gs_path: str, local_path: str):
        toolbox.create_local_dir(
            path.join(local_path, am_sys_definitions.COLUMNS_FOLDER)
        )
        # saving PCA model if exists
        if prep_data.pca_model:
            joblib.dump(
                prep_data.pca_model,
                path.join(local_path, am_sys_definitions.PCA_PICKLE_FILE),
            )
            self.bq.localfile2gs(
                bucket=GS_BUCKET_NAME,
                source_path=path.join(local_path, am_sys_definitions.PCA_PICKLE_FILE),
                dest_path=path.join(gs_path, am_sys_definitions.PCA_PICKLE_FILE),
            )

        for name, val in prep_data.column_dict.items():
            column_definition = dict()
            filename = name + ".json"
            local_path_full = path.join(
                local_path, am_sys_definitions.COLUMNS_FOLDER, filename
            )
            column_definition["type"] = val.get("type")
            column_definition["values"] = val.get("values")
            for prep_type in ["scaler", "mlb", "encoder"]:
                if val.get(prep_type):
                    self.logger.info(
                        "Saving preprocessing file for {nm}: {pp}".format(
                            nm=name, pp=prep_type
                        )
                    )
                    prep_file = "{nm}.{ext}".format(nm=name, ext=prep_type)
                    column_definition["{}_file".format(prep_type)] = prep_file
                    joblib.dump(
                        val.get(prep_type),
                        path.join(
                            local_path, am_sys_definitions.COLUMNS_FOLDER, prep_file
                        ),
                    )
            json_object = json.dumps(column_definition, indent=4)
            with open(local_path_full, "w") as outfile:
                outfile.write(json_object)
        self.bq.upload_from_directory(
            path.join(local_path, am_sys_definitions.COLUMNS_FOLDER),
            GS_BUCKET_NAME,
            path.join(gs_path, am_sys_definitions.COLUMNS_FOLDER),
        )
        self.logger.info(
            "Uploaded column data to GS {}".format(
                path.join(gs_path, am_sys_definitions.COLUMNS_FOLDER)
            )
        )

    def copy_submodel_to_production(self, submodel_name):
        source_folder = path.join(
            GS_FOLDER_NAME,
            self.model_name,
            am_sys_definitions.TEMPORARY_MODELS_FOLDER,
            submodel_name,
        )
        for blob in self.bq.list_blobs(GS_BUCKET_NAME, source_folder):
            dest_blob = blob.replace(
                am_sys_definitions.TEMPORARY_MODELS_FOLDER,
                am_sys_definitions.PRODUCTION_MODELS_FOLDER,
            )
            self.bq.copy_blob(
                bucket_name=GS_BUCKET_NAME,
                blob_name=blob,
                destination_bucket_name=GS_BUCKET_NAME,
                destination_blob_name=dest_blob,
            )


def get_old_model_loss(
    prep_data,
    model_config,
    submodel_name,
    model_name,
    prod_submodel_folder,
    train_pct,
    model_config_global,
):
    logger = logging.getLogger(__name__)
    prod_submodel = existing.Submodel(
        model_config,
        submodel_name,
        model_name,
        prod_submodel_folder,
    )
    # import existing production model
    prod_columns_definition = prod_submodel.get_column_definition()
    logger.info("Loaded production model")
    logger.info("Getting original data scaled with production columns")
    # If the model has a categorical variables enabled, we don't need to transform the data
    if model_config_global.get("skip_preprocessing") == "enable_categorical":
        df_x_sample_transformed = prep_data.pred[list(prod_submodel.expanded_columns)]
    elif model_config_global.get("skip_preprocessing") == "True":
        df_x_sample_transformed = prep_data.pred
    else:
        df_x_sample_transformed = prep_data.transform_x(
            prod_columns_definition=prod_columns_definition,
            prod_columns=prod_submodel.expanded_columns,
        )
    # taking a test subset for model comparison
    try:
        tts_prod = dataprep.TrainTestSplit(
            x=df_x_sample_transformed, y=prep_data.Y, train_pct=train_pct
        )
    except Exception as ex:
        logger.error("Error splitting train and test data \n: {}".format(ex))
        raise

    try:
        prod_model_loss = prod_submodel.predict_loss(
            X=tts_prod.test_x,
            y_true=tts_prod.test_y,
        )
    except Exception as ex:
        logger.error("Error getting model loss: \n {}".format(ex))
        prod_model_loss = -1
    return prod_model_loss


def run_submodel(submodel_name, model_name, model_config_global, train_pct, values):
    logger = logging.getLogger(__name__)
    prod_submodel_folder = path.join(
        WORK_DIR,
        model_name,
        am_sys_definitions.PRODUCTION_MODELS_FOLDER,
        submodel_name,
    )
    # Create model folder in the working directory
    temp_submodel = TempSubModel(
        model_config_global, model_name, train_pct, submodel_name, values
    )
    logger.info("Started training sub-model {}".format(submodel_name))
    prep_data = temp_submodel.run()
    logger.info("Ended training sub-model {}".format(submodel_name))

    if (
        not path.exists(prod_submodel_folder)
        or not getattr(temp_submodel.model_class, "__supervised__")
        or not prep_data
        or model_config_global.get("purpose") == "embedding"
    ):
        logger.info(
            """No production model found or supervised == {}.
               copying the temporary model to production""".format(
                getattr(temp_submodel.model_class, "__supervised__")
            )
        )
        logger.info("Saving statistics")
        loc = temp_submodel.save_statistics()
        logger.info("The file saved at {}".format(loc))
        temp_submodel.copy_submodel_to_production(submodel_name)
        msg = "The loss is {}".format(str(round(temp_submodel.loss, 2)))
    else:
        old_model_loss = get_old_model_loss(
            prep_data=prep_data,
            model_config=temp_submodel.model_config,
            submodel_name=submodel_name,
            model_name=model_name,
            prod_submodel_folder=prod_submodel_folder,
            train_pct=train_pct,
            model_config_global=model_config_global,
        )
        logger.info("Saving statistics, loss is {}".format(old_model_loss))
        loc = temp_submodel.save_statistics(production_loss=old_model_loss)
        logger.info("The file saved at {}".format(loc))

        if temp_submodel.loss < old_model_loss or old_model_loss == -1:
            temp_submodel.copy_submodel_to_production(submodel_name)
            msg = """New loss ({nl}) is LESS than the old one ({ol}), 
                   REPLACING the model with the new one""".format(
                nl=str(round(temp_submodel.loss, 2)), ol=str(round(old_model_loss, 2))
            )
        else:
            msg = """New loss ({nl}) is GREATER than the old one ({ol}), 
                   KEEPING the model with the old one""".format(
                nl=str(round(temp_submodel.loss, 2)), ol=str(round(old_model_loss, 2))
            )
    return msg


def main():
    args = docopt(__doc__)
    model_name = args["--model_name"]
    nc = toolbox.NotificationCenter(**SYSTEM_CONFIG)
    nc.send_info("Started training model: {}".format(model_name))
    toolbox.InitLogger(logger_name="model_train_{}".format(model_name))
    logger = logging.getLogger(__name__)
    logger.info("Started training model {}".format(model_name))
    try:
        if args["--submodels"] == "[]":
            submodels = []
        else:
            submodels = args["--submodels"].strip("][").split(", ")
    except Exception:
        submodels = []
    train_pct = float(args["--train_pct"])
    if DB_TYPE == "bigquery":
        if not BQ_PROJECT_ID:
            raise Exception("bq_project_id is required when bigquery is selected")

    if not args["--skip_copy_gs_files"]:
        logger.info("copying all submodels and config from GS")
        ld = dataprep.LocalDirectories(
            model_name=model_name,
            bq=gcp.Bigquery(bq_project_id=BQ_PROJECT_ID, gs_project_id=GS_PROJECT_ID),
        )
        ld.copy_from_gs(
            gs_bucket_name=GS_BUCKET_NAME,
            gs_folder_name=GS_FOLDER_NAME,
            work_dir=WORK_DIR,
        )

    logger.info("Getting configuration file")
    try:
        model_config_global, submodels_to_run = dataprep.get_model_configuration(
            model_name, submodels, WORK_DIR
        )
    except Exception as ex:
        logger.error("Error getting configuration file {}".format(ex))
        raise
    logger.info("Finished loading configuration file")

    # If model purpose is in ["embedding", "clustering"]
    # then train pct = 1
    if model_config_global["purpose"] in ["embedding", "clustering"]:
        train_pct = 1
    for submodel_name, values in submodels_to_run.items():
        for s_key, s_value in values.items():
            model_config_global[s_key] = s_value
        nc.send_info(
            "Started training sub model {sn} of model {md}".format(
                sn=submodel_name, md=model_name
            )
        )
        try:
            msg = run_submodel(
                submodel_name=submodel_name,
                model_name=model_name,
                model_config_global=model_config_global,
                train_pct=train_pct,
                values=values,
            )
            nc.send_info(msg)
        except Exception as ex:
            nc.send_alert(
                "Error training submodel {sm} of the model {md}".format(
                    sm=submodel_name, md=model_name
                )
            )
            logger.error("Error training submodel: {}".format(ex))
            continue
        nc.send_info(
            "Ended training submodel {sn} of model {md}".format(
                sn=submodel_name, md=model_name
            )
        )
    nc.send_info("Ended training model: {}".format(model_name))


if __name__ == "__main__":
    main()
