#!/usr/bin/env python3
"""
predict_model.py
Author: Semeon Balagula
The script will run the model predictions and/or its submodels, save the results in Google Storage and 


Usage:
    predict_model.py [--help] (--model_name <value>) [--submodels <value>] [--prod] 
                     [--skip_copy_gs_files] [--keep_results_in_gs]
                     [--config_file_location <value>]

                       

Options:
    --help                          Show this screen and exit
    --model_name <value>            Specify model name
    --submodels <value>             Submodels in the format: "[x, y, z]", if not specified, all submodels will be run [default: []]
    --prod                          If specified, prod environment will be invoked
    --skip_copy_gs_files            If scpecified, existing files will not be copied
    --keep_results_in_gs            If specified, results will be kept in GS
    --config_file_location <value>  Specify the location of the config file [default: /home/prod/config.json]

"""

from docopt import docopt
from os import path
import logging
from datetime import datetime
import time

from core import (
    db,
    gcp,
    dataprep,
    am_sys_definitions,
    existing,
)
from system import toolbox

args = docopt(__doc__)


try:
    config_file = toolbox.open_json_file(args["--config_file_location"])
    print("System config file was found!")
    if args["--prod"]:
        SYSTEM_CONFIG = config_file["prod"]
    else:
        SYSTEM_CONFIG = config_file["dev"]
except Exception as ex:
    print("System config file was not found!")
    raise

DB_TYPE = SYSTEM_CONFIG.get("db_type")
WORK_DIR = SYSTEM_CONFIG.get("work_dir")
BQ_PROJECT_ID = SYSTEM_CONFIG.get("bq_project_id")
DEST_BQ_PROJECT_ID = SYSTEM_CONFIG.get("dest_bq_project_id")
GS_PROJECT_ID = SYSTEM_CONFIG.get("gs_project_id")
GS_BUCKET_NAME = SYSTEM_CONFIG.get("gs_bucket_name")
GS_FOLDER_NAME = SYSTEM_CONFIG.get("gs_folder_name")
DEST_GS_RESULTS = SYSTEM_CONFIG.get("dest_gs_results")
DEST_BQ_DATASET = SYSTEM_CONFIG.get("dest_bq_dataset")
TEMP_BQ_DATASET = SYSTEM_CONFIG.get("temp_bq_dataset")


class ProdModel(object):
    def __init__(self, global_config, model_name) -> None:
        self.logger = logging.getLogger(__name__)
        self.logger.info("Global config: \n {}".format(global_config))
        self.global_config = global_config
        self.model_name = model_name
        if global_config.get("purpose") == "classification":
            self.bq_table_schema = am_sys_definitions.CATEGORIAL_OUTPUT_SCHEMA
        elif global_config.get("purpose") == "regression":
            self.bq_table_schema = am_sys_definitions.CONT_OUTPUT_SCHEMA
        elif global_config.get("purpose") == "clustering":
            self.bq_table_schema = am_sys_definitions.CONT_OUTPUT_SCHEMA
        elif global_config.get("purpose") == "survival":
            self.bq_table_schema = am_sys_definitions.GENERAL_OUTPUT_SCHEMA
        else:
            raise Exception(
                "No schema defined for the purpose {}".format(
                    global_config.get("purpose")
                )
            )
        self.queries_folder = path.join(
            WORK_DIR, model_name, am_sys_definitions.PROD_QUERIES_FOLDER
        )
        self.bq = gcp.Bigquery(bq_project_id=BQ_PROJECT_ID, gs_project_id=GS_PROJECT_ID)
        self.logger.info("BQ project id: {}".format(BQ_PROJECT_ID))
        self.bq_dest = gcp.Bigquery(
            bq_project_id=DEST_BQ_PROJECT_ID, gs_project_id=GS_PROJECT_ID
        )
        self.logger.info("Destination BQ project id: {}".format(DEST_BQ_PROJECT_ID))


class ProdSubModel(ProdModel):
    """
    The purpose of this class is to execute and save the model results
    """

    def __init__(self, global_config, model_name, values, submodel_name) -> None:
        super().__init__(global_config, model_name)
        self.values = values
        self.submodel_name = submodel_name
        self.logger.info("Starting loading the query")
        queries = db.QueryList(
            db_type=DB_TYPE,
            query_list=toolbox.gen_queries(self.queries_folder, self.values),
            bq_project_id=BQ_PROJECT_ID,
        )
        self.logger.info("The queries were loaded successfully")
        self.df = queries.to_dataframe(key_list=self.global_config.get("key_list"))
        self.pred_key = list(self.df[global_config["pred_key"]])
        self.rd = dataprep.RawData(
            df=self.df,
            key_list=self.global_config.get("key_list"),
        )
        self.logger.info("The dataframe was loaded successfully")

    def remove_keys(self):
        df_tmp = self.df.copy()
        for col in self.global_config.get("key_list"):
            try:
                # drop the key columns
                df_tmp.drop(col, axis=1, inplace=True)
            except Exception:
                self.logger.warning(
                    "Column {} was not found in the dataframe".format(col)
                )
        return df_tmp

    def load_results_to_gcp(self, predicted_dict, execution_id, keep_results_in_gs):
        filename = "{eid}.json".format(eid=execution_id)
        source_path = path.join(WORK_DIR, filename)
        toolbox.save_json_file(predicted_dict, source_path)
        gs_path = path.join(
            DEST_GS_RESULTS,
            self.model_name,
            self.submodel_name,
            filename,
        )
        self.bq.localfile2gs(
            bucket=GS_BUCKET_NAME,
            source_path=source_path,
            dest_path=gs_path,
            delete_local=True,
        )
        if self.global_config.get("bq_write_method") == "merge":
            temp_tablename = "{mn}_{sm}_{eid}".format(
                mn=self.model_name, sm=self.submodel_name, eid=execution_id
            )
            self.bq_dest.load(
                dataset=TEMP_BQ_DATASET,
                table=temp_tablename,
                path="gs://" + path.join(GS_BUCKET_NAME, gs_path),
                source_format="NEWLINE_DELIMITED_JSON",
                write_flag="WRITE_APPEND",
                schema=self.bq_table_schema,
            )
            output_columns = [i.get("name") for i in self.bq_table_schema]
            try:
                self.merge_with_target_table(
                    output_cols=output_columns,
                    target_table=self.global_config.get("pred_destination_table"),
                    source_table=temp_tablename,
                    target_dataset=DEST_BQ_DATASET,
                    source_dataset=TEMP_BQ_DATASET,
                    key_column=self.global_config["pred_key"],
                )
            except Exception as ex:
                self.logger.error("could not merge the table!: \n {}".format(ex))
                print("could not merge the target table!")
            # Drop temp table
            self.bq_dest.delete_bq_table(dataset=TEMP_BQ_DATASET, table=temp_tablename)
        elif self.global_config.get("bq_write_method") == "truncate":
            self.bq_dest.load(
                dataset=DEST_BQ_DATASET,
                table=self.global_config.get("pred_destination_table"),
                path="gs://" + path.join(GS_BUCKET_NAME, gs_path),
                source_format="NEWLINE_DELIMITED_JSON",
                write_flag="WRITE_TRUNCATE",
                schema=self.bq_table_schema,
            )
        else:
            raise Exception("Unknown bq_write_method")
        self.logger.info("Results saved to BQ, removing cloud files")
        if not keep_results_in_gs:
            try:
                self.bq.delete_blobs(bucket_name=GS_BUCKET_NAME, path=gs_path)
            except Exception as ex:
                self.logger.error("could not delete blobs!: \n {}".format(ex))
                print("could not delete blobs!")

    def merge_with_target_table(
        self,
        output_cols,
        target_table,
        source_table,
        target_dataset,
        source_dataset,
        key_column,
    ):
        # if table not exists, just copy it
        if target_table not in self.bq_dest.list_tables(target_dataset):
            self.logger.info(
                "The table {tbl} was not found in {ds}, creating new one".format(
                    tbl=target_table, ds=target_dataset
                )
            )
            self.bq_dest.copy_bq_table(
                source_dataset=source_dataset,
                source_table=source_table,
                target_dataset=target_dataset,
                target_table=target_table,
            )
        else:
            query = """
                MERGE {target_dataset}.{target_table} T
                USING {source_dataset}.{source_table} S
                ON
                T.{key_column} = S.{key_column} AND T.submodel_name = S.submodel_name
                WHEN MATCHED THEN
                UPDATE SET
                    {update_statement}
                WHEN NOT MATCHED BY TARGET THEN
                INSERT ({insert_statement})
                VALUES ({values_statement})
                """.format(
                update_statement=",".join(
                    [
                        "{col} = S.{col}".format(col=a)
                        for a in output_cols
                        if a != self.pred_key
                    ]
                ),
                insert_statement=",".join(output_cols),
                values_statement=",".join(
                    ["S.{col}".format(col=a) for a in output_cols]
                ),
                target_dataset=target_dataset,
                target_table=target_table,
                source_dataset=source_dataset,
                source_table=source_table,
                key_column=key_column,
            )
            self.logger.info("The merge query is: \n {}".format(query))
            self.bq_dest.execute_query(query)


def rescale_classes(pi, mu, value):
    """
    pi - probability of the event in the real dataset
    mu - probability of the event in the manufactured dataset
    p_manufuctured - probability calculated based on the manufactured dataset
    """
    if pi and mu:
        p_manufuctured = value[1]
        rescale_1 = (pi * (1 - mu) * p_manufuctured) / (
            (pi - mu) * p_manufuctured + mu * (1 - pi)
        )
        return [1 - rescale_1, rescale_1]
    else:
        return value


def classification_pred_output(predicted_results, pi, mu):
    return [
        [
            {"class": int(key), "value": str(val)}
            for key, val in enumerate(rescale_classes(pi, mu, i))
        ]
        for i in predicted_results
    ]


def regression_prod_output(predicted_results):
    output = []
    for i in predicted_results:
        if type(i) == list:
            output.append(float(i[0]))
        else:
            output.append(float(i))
    return output


def predict_submodel(
    values,
    submodel_name,
    model_name,
    model_config_gobal,
    execution_id,
    keep_results_in_gs=True,
):
    """
    values: the input data for making predictions
    submodel_name: the name of the submodel that will be used for making predictions
    model_name: the name of the model that contains the submodel
    model_config_global: a dictionary containing the configuration for the model
    execution_id: an execution ID for the prediction
    """
    logger = logging.getLogger(__name__)
    # Create model folder in the working directory
    logger.info("Started predicting the sub-model {}".format(submodel_name))
    prod_statistics = toolbox.open_json_file(
        path.join(
            WORK_DIR,
            model_name,
            am_sys_definitions.PRODUCTION_MODELS_FOLDER,
            submodel_name,
            am_sys_definitions.STATISTICS_FILE,
        )
    )
    prod_submodel_folder = path.join(
        WORK_DIR,
        model_name,
        am_sys_definitions.PRODUCTION_MODELS_FOLDER,
        submodel_name,
    )
    logger.info("Statistics: \n {}".format(prod_statistics))
    # Getting existing prod model object, configuration and input columns
    prod_submodel = existing.Submodel(
        model_config=model_config_gobal,
        submodel_name=submodel_name,
        model_name=model_name,
        prod_model_folder=prod_submodel_folder,
    )
    # Getting data for the prediction
    psm = ProdSubModel(
        global_config=model_config_gobal,
        model_name=model_name,
        values=values,
        submodel_name=submodel_name,
    )
    if model_config_gobal.get("skip_preprocessing") == "True":
        logger.info("Skipping preprocessing")
        df_x_transformed = psm.df
    elif model_config_gobal.get("skip_preprocessing") == "enable_categorical":
        """
        Keep the categorical columns as is, and only preprocess the numerical columns
        """
        prod_columns_definition = prod_submodel.get_column_definition()
        logger.info("Starting converting eligible columns to categorical")
        df_x_transformed = psm.remove_keys()
        df_x_transformed = df_x_transformed[prod_submodel.expanded_columns]
        for col, col_detail in prod_columns_definition.items():
            if col_detail["type"] == am_sys_definitions.CATEGORIAL:
                df_x_transformed[col] = df_x_transformed[col].astype("category")
                logger.info("Converted column {} to categorical".format(col))
        logger.info("Ended converting eligible columns to categorical")
    else:
        logger.info("Getting original data scaled with production columns")
        try:
            df_x_transformed = psm.rd.transform_x(
                prod_columns_definition=prod_submodel.get_column_definition(),
                prod_columns=prod_submodel.expanded_columns,
                pca_model=prod_submodel.pca_model,
            )
        except Exception as ex:
            logger.error("Could not get original data sample: \n {}".format(ex))
            raise
    logger.info("Transformed dataset size is {}".format(df_x_transformed.shape))
    logger.info("Transformed data sample: \n {}".format(df_x_transformed.head(5)))
    logger.info("Predicting the results")
    try:
        predicted_results = prod_submodel.predict(X=df_x_transformed)
    except Exception as ex:
        logger.error("Error predicting results: \ {}".format(ex))
        raise
    if model_config_gobal.get("purpose") == "classification":
        predicted_results = classification_pred_output(
            predicted_results, prod_statistics.get("pi"), prod_statistics.get("mu")
        )
    elif model_config_gobal.get("purpose") == "survival":
        pass
    else:
        predicted_results = regression_prod_output(predicted_results)

    logger.info("results sample: {}".format(predicted_results[:5]))
    logger.info("Finished predicting the results")

    logger.info("Getting target column values")
    try:
        target_column_name = model_config_gobal.get("target_column_name")
        if target_column_name in psm.rd.pred.columns.values:
            target_column_current_values = list(psm.rd.pred[target_column_name])
        else:
            logger.info("No target column found, creating null list")
            target_column_current_values = [None] * len(predicted_results)
        logger.info("Loaded target column values")
    except Exception as ex:
        target_column_current_values = [None] * len(predicted_results)
        logger.warning("Could not get target column values: \ {}".format(ex))

    predicted_dict = toolbox.output_builder(
        predicted_results=predicted_results,
        target_column_values=target_column_current_values,
        keys=psm.pred_key,
        model_name=model_name,
        submodel_name=submodel_name,
        execution_id=execution_id,
        submodel_rank=values.get("rank"),
    )

    logger.info("Sample output: \n {}".format(predicted_dict[0]))
    # loading to GCP
    try:
        psm.load_results_to_gcp(predicted_dict, execution_id, keep_results_in_gs)
    except Exception as ex:
        logger.error("Error loading results to gcp: \n {}".format(ex))
        raise


def main():
    args = docopt(__doc__)
    model_name = args["--model_name"]
    if args["--keep_results_in_gs"]:
        keep_results_in_gs = True
    else:
        keep_results_in_gs = False
    nc = toolbox.NotificationCenter(**SYSTEM_CONFIG)
    nc.send_info("Started predicting model: {}".format(model_name))
    execution_id = int(time.time())
    toolbox.InitLogger(logger_name="model_predict_{}".format(model_name))
    logger = logging.getLogger(__name__)
    logger.info(
        "Started model {model} at {dt}, execution id {eid}".format(
            model=model_name, dt=datetime.utcnow(), eid=execution_id
        )
    )
    try:
        if args["--submodels"] == "[]":
            submodels = []
        else:
            submodels = args["--submodels"].strip("][").split(", ")
    except Exception:
        submodels = []
    if DB_TYPE == "bigquery":
        if not BQ_PROJECT_ID:
            raise Exception("bq_project_id is required when bigquery is selected")
    ld = dataprep.LocalDirectories(
        model_name=model_name,
        bq=gcp.Bigquery(bq_project_id=BQ_PROJECT_ID, gs_project_id=GS_PROJECT_ID),
    )

    if not args["--skip_copy_gs_files"]:
        ld.copy_from_gs(
            gs_bucket_name=GS_BUCKET_NAME,
            gs_folder_name=GS_FOLDER_NAME,
            work_dir=WORK_DIR,
        )

    # Getting model configuration
    model_config_gobal, submodels_to_run = dataprep.get_model_configuration(
        model_name, submodels, WORK_DIR
    )
    for submodel_name, values in submodels_to_run.items():
        nc.send_info(
            "Started predicting sub model {sn} of model {md}".format(
                sn=submodel_name, md=model_name
            )
        )
        for s_key, s_value in values.items():
            model_config_gobal[s_key] = s_value
        try:
            predict_submodel(
                values=values,
                submodel_name=submodel_name,
                model_name=model_name,
                model_config_gobal=model_config_gobal,
                execution_id=execution_id,
                keep_results_in_gs=keep_results_in_gs,
            )
        except Exception as ex:
            nc.send_alert(
                "Error predicting submodel {sm} of the model {md}. See logs".format(
                    sm=submodel_name, md=model_name
                )
            )
            logger.error("Error: {}".format(ex))
            continue
        nc.send_info(
            "Ended predicting sub model {sn} of model {md}".format(
                sn=submodel_name, md=model_name
            )
        )
    nc.send_info("Ended predicting model: {}".format(model_name))


if __name__ == "__main__":
    main()
