from os import path
import json
import logging
import joblib
import tensorflow as tf

from core.gcp import Bigquery
from core import am_sys_definitions


class Structure(object):
    def __init__(
        self, model_name, submodel_name, gs_bucket, gs_folder_name, bq_project_id
    ):
        self.model_name = model_name
        self.submodel_name = submodel_name
        self.gs_bucket = gs_bucket
        self.gs_folder_name = gs_folder_name
        self.submodel_folder = path.join(
            gs_folder_name,
            model_name,
            am_sys_definitions.PRODUCTION_MODELS_FOLDER,
            submodel_name,
        )
        self.columns_blob = path.join(
            self.submodel_folder, am_sys_definitions.COLUMNS_FILE
        )
        self.all_columns_blob = path.join(
            self.submodel_folder, am_sys_definitions.EXPANDED_COLUMNS_FILE
        )
        self.bq = Bigquery(bq_project_id=bq_project_id)

    def _read_raw(self):
        file_contents = self.bq.read_from_gs(self.gs_bucket, self.columns_blob)
        return json.loads(file_contents)

    def _read_file(self, file_name, transform_json=False):
        file_blob = path.join(self.submodel_folder, file_name)
        file_contents = self.bq.read_from_gs(self.gs_bucket, file_blob)
        if transform_json:
            return json.loads(file_contents)
        else:
            return file_contents

    def _read_all_columns(self):
        file_contents = self.bq.read_from_gs(self.gs_bucket, self.all_columns_blob)
        return str(file_contents)  # json.loads(file_contents)

    def build(self):
        logger = logging.getLogger(__name__)
        structure = dict()
        columns = self._read_raw()
        for col, file_name in columns.items():
            column_properties = self._read_file(file_name, transform_json=True)
            if column_properties.get("type") == am_sys_definitions.REGULAR:
                try:
                    column_properties["scaler"] = self.bq.read_gs_model(
                        self.gs_bucket,
                        path.join(
                            self.submodel_folder,
                            am_sys_definitions.COLUMNS_FOLDER,
                            column_properties.get("scaler_file"),
                        ),
                    )
                except Exception as ex:
                    logger.warning("No scaler was found: {}".format(ex))
            structure[col] = column_properties
        return structure


class InputData(Structure):
    def __init__(
        self, model_name, submodel_name, gs_bucket, gs_folder_name, bq_project_id
    ):
        super().__init__(
            model_name, submodel_name, gs_bucket, gs_folder_name, bq_project_id
        )
        self.structure = self.build()
        self.all_columns = self._read_all_columns()

    # def transform(self, )


input = InputData(
    model_name="click_prediction",
    submodel_name="1_ad_1st",
    gs_bucket="airtrade",
    gs_folder_name="am_models/models_definition",
    bq_project_id="air-trade-2020",
)

print(input.all_columns.split(","))
